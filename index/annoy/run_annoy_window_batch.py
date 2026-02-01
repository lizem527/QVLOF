#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import argparse
from typing import List, Tuple

import numpy as np

# faiss 用于真值（L2 exact）
import faiss

from annoy import AnnoyIndex


# -----------------------------
# 全局计时与日志
# -----------------------------
T_GLOBAL0 = time.perf_counter()


def _fmt_elapsed() -> str:
    return f"{(time.perf_counter() - T_GLOBAL0):.3f}s"


def log(msg: str):
    # flush=True：长任务时实时输出日志
    print(f"[LOG] [elapsed={_fmt_elapsed()}] {msg}", flush=True)


# -----------------------------
# 工具：解析逗号分隔列表
# -----------------------------
def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


# -----------------------------
# 读取 CSV（id + 128维）
# -----------------------------
def load_csv_id_vec(path: str, d: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSV: id,v0,v1,...,v(d-1)
    返回:
      ids: (N,) int64
      vecs: (N,d) float32
    """
    t0 = time.perf_counter()
    log(f"[STEP 1] Loading CSV: {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != d + 1:
        raise ValueError(f"[CSV DIM ERROR] {path}: expect {d+1} cols (id+{d}), got {data.shape[1]}")

    ids = data[:, 0].astype(np.int64)
    vecs = data[:, 1:].astype(np.float32)

    t1 = time.perf_counter()
    vec0_preview = vecs[0, :8].tolist() if vecs.shape[0] > 0 else []
    log(
        f"[STEP 1] Loaded CSV done. rows={vecs.shape[0]} cols={data.shape[1]} vec_dim={vecs.shape[1]} "
        f"time={(t1 - t0):.3f}s ids_preview={ids[:3].tolist()} vec0_preview={vec0_preview}"
    )
    return ids, vecs


# -----------------------------
# 构建 Annoy 索引（用行号 idx 作为 annoy_id）
# -----------------------------
def build_annoy_index(
        base_vecs: np.ndarray,
        metric: str,
        n_trees: int,
        n_jobs: int,
        seed: int,
) -> Tuple[AnnoyIndex, float]:
    """
    返回：annoy_index, build_ms
    build_ms 计入 add_item + build 的总耗时（常见实验口径）
    """
    n, d = base_vecs.shape
    log(f"[STEP 2] Build Annoy start: N={n} D={d} metric={metric} n_trees={n_trees} n_jobs={n_jobs} seed={seed}")

    a = AnnoyIndex(d, metric)
    a.set_seed(seed)

    t0 = time.perf_counter_ns()

    # add_item: annoy_id = 行号 idx（0..N-1）
    # 进度日志：每 100k 打一次（或小数据每 10% 打一次）
    prog_every = 100000 if n >= 100000 else max(1, n // 10)
    for idx in range(n):
        a.add_item(idx, base_vecs[idx])
        if (idx + 1) % prog_every == 0:
            log(f"[STEP 2] add_item progress: {idx+1}/{n}")

    log("[STEP 2] add_item done. start build() ...")
    a.build(n_trees, n_jobs=n_jobs)
    log("[STEP 2] build() done.")

    t1 = time.perf_counter_ns()
    build_ms = (t1 - t0) / 1e6
    log(f"[STEP 2] Build Annoy finished. build_ms={build_ms:.3f} ms")
    return a, build_ms


# -----------------------------
# 计算真值：Faiss FlatL2（一次算 maxK）
# -----------------------------
def compute_ground_truth_faiss_l2(
        base_vecs: np.ndarray,
        query_vecs: np.ndarray,
        max_k: int,
) -> np.ndarray:
    """
    返回 true_indices: (Q, max_k) int64
    """
    n, d = base_vecs.shape
    q = query_vecs.shape[0]
    log(f"[STEP 3] Ground truth (Faiss FlatL2) start: N={n} D={d} Q={q} maxK={max_k}")

    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(d)
    index.add(base_vecs)  # exact brute-force
    _, true_indices = index.search(query_vecs, max_k)
    t1 = time.perf_counter()

    log(f"[STEP 3] Ground truth done. true_indices.shape={true_indices.shape} time={(t1 - t0):.3f}s")
    return true_indices.astype(np.int64)


# -----------------------------
# 运行实验：对 search_k × topK 网格
# -----------------------------
def run_experiment(
        loadcsv: str,
        point_csv: str,
        log_dir: str,
        topKs: List[int],
        search_ks: List[int],
        metric: str,
        n_trees: int,
        n_jobs: int,
        seed: int,
        use_auto_search_k: bool,
        window_size: int,  # [MOD] NEW: per-window aggregation size
):
    log("[STEP 0] Experiment start")
    log(f"[CONFIG] loadcsv={loadcsv}")
    log(f"[CONFIG] point_csv={point_csv}")
    log(f"[CONFIG] log_dir(base)={log_dir}")
    log(f"[CONFIG] topKs={topKs}")
    log(f"[CONFIG] search_ks(input)={search_ks}")
    log(f"[CONFIG] use_auto_search_k={use_auto_search_k}")
    log(f"[CONFIG] window_size={window_size}")  # [MOD] NEW
    log(f"[CONFIG] metric={metric} n_trees={n_trees} n_jobs={n_jobs} seed={seed}")

    # =============================
    # 创建时间戳子目录
    # =============================
    # ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # log_dir_ts = os.path.join(log_dir, ts)
    log_dir_ts = log_dir
    os.makedirs(log_dir_ts, exist_ok=True)
    log(f"[STEP 0] Created timestamp log dir: {log_dir_ts}")

    base_name = os.path.splitext(os.path.basename(loadcsv))[0]

    # [MOD] NEW: per-window metrics CSV (replaces old qps_csv usage)
    per_window_csv = os.path.join(log_dir_ts, f"per_window_annoy_{base_name}.csv")

    # keep per-query id list CSV
    ids_csv = os.path.join(log_dir_ts, f"blockids_annnoy_{base_name}.csv")

    log(f"[STEP 0] Output files:")
    log(f"         per_window_csv={per_window_csv}")  # [MOD] NEW
    log(f"         ids_csv={ids_csv}")

    # 1) 读数据
    base_ids, base_vecs = load_csv_id_vec(loadcsv, d=128)
    query_ids, query_vecs = load_csv_id_vec(point_csv, d=128)

    if base_vecs.shape[1] != 128 or query_vecs.shape[1] != 128:
        raise ValueError("Only support 128D in this script. Please modify d if needed.")

    log(f"[STEP 1] Loaded base/query OK. base_vecs={base_vecs.shape}, query_vecs={query_vecs.shape}")
    log(f"[STEP 1] base_ids range: min={int(base_ids.min())} max={int(base_ids.max())} n={len(base_ids)}")
    log(f"[STEP 1] query_ids range: min={int(query_ids.min())} max={int(query_ids.max())} n={len(query_ids)}")

    # 2) 构建 Annoy
    annoy_index, build_ms = build_annoy_index(
        base_vecs=base_vecs,
        metric=metric,
        n_trees=n_trees,
        n_jobs=n_jobs,
        seed=seed,
    )

    # 3) Faiss 真值（一次算 maxK）
    topKs_sorted = sorted(set(int(k) for k in topKs))
    if not topKs_sorted:
        raise ValueError("topKs is empty.")
    maxK = max(topKs_sorted)
    log(f"[STEP 3] Compute ground truth once with maxK={maxK} (reuse prefixes for smaller K).")
    true_indices = compute_ground_truth_faiss_l2(base_vecs, query_vecs, max_k=maxK)  # (Q, maxK)

    # 4) 写 CSV 头
    log("[STEP 4] Writing CSV headers ...")

    # [MOD] NEW: per-window header
    with open(per_window_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "search_k", "k", "n_trees", "build_ms",
            "window_id", "qb", "qe", "window_q",
            "lat_ms_mean", "qps", "recall_mean",
            "window_search_sec",  # [MOD] NEW: scheme-A aligned (search+refetch only)
        ])

    with open(ids_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["search_k", "k", "query", "id_list"])

    log("[STEP 4] CSV headers written.")

    # 5) 主循环：对每个 topK 分轮
    Q = query_vecs.shape[0]

    # [MOD] NEW: window settings
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    nwin = (Q + window_size - 1) // window_size

    log(f"[STEP 5] Start grid search. Q={Q}, topKs_sorted={topKs_sorted}, window_size={window_size}, nwin={nwin}")

    # 进度打印频率（不影响计时段）
    query_prog_every = 1000 if Q >= 5000 else max(1, Q // 10)

    for k in topKs_sorted:
        log(f"[STEP 5] ===== Begin topK={k} =====")

        # 自动 search_k：search_k ≈ n_trees * k（Annoy README 默认语义）
        if use_auto_search_k:
            search_ks_eff = [int(n_trees * k)]
            log(f"[AUTO] search_k not specified => use search_k = n_trees * k = {n_trees} * {k} = {search_ks_eff[0]}")
        else:
            search_ks_eff = search_ks
            log(f"[STEP 5] Using user-provided search_ks: {search_ks_eff}")

        t_cache0 = time.perf_counter()
        true_sets = [set(true_indices[qi, :k].tolist()) for qi in range(Q)]
        t_cache1 = time.perf_counter()
        log(f"[STEP 5] Prepared true_sets for k={k}. time={(t_cache1 - t_cache0):.3f}s")

        for sk in search_ks_eff:
            log(f"[STEP 5] ---- Grid start: k={k}, search_k={sk} ----")
            grid_t0 = time.perf_counter()

            # [MOD] NEW: per-window loop (Scheme A aligned: qps and latency share same timing scope)
            for w_id in range(nwin):
                qb = w_id * window_size
                qe = min(qb + window_size, Q)
                wn = qe - qb
                if wn <= 0:
                    continue

                win_query_ms_sum = 0.0
                win_recalls = []
                id_rows = []  # buffer ids for this window, write after timing

                for qi in range(qb, qe):
                    q = query_vecs[qi]
                    q_list = q.tolist()  # 放在计时外（避免污染）

                    # -----------------------------
                    # 严格计时段：只包含 annoy 查询 + 回表
                    # -----------------------------
                    t0 = time.perf_counter_ns()
                    nn = annoy_index.get_nns_by_vector(
                        q_list,
                        k,
                        search_k=sk,
                        include_distances=False,
                    )
                    # 回表：取回向量（模拟从索引命中到取出记录）
                    for idx in nn:
                        _ = annoy_index.get_item_vector(idx)
                    t1 = time.perf_counter_ns()
                    # -----------------------------

                    query_ms = (t1 - t0) / 1e6
                    win_query_ms_sum += query_ms

                    # recall 计算不计时
                    hit = len(set(nn) & true_sets[qi])
                    win_recalls.append(hit / float(k))

                    # per-query topK id list（写盘放在计时外）
                    id_list_str = " ".join(str(int(x)) for x in nn)
                    id_rows.append((sk, k, qi, id_list_str))

                    if (qi + 1) % query_prog_every == 0:
                        log(f"[STEP 5] progress (k={k}, search_k={sk}): {qi+1}/{Q}")

                # [MOD] NEW: Scheme A aligned per-window metrics:
                # - latency is mean of per-query (search+refetch) times
                # - qps uses SAME scope: window_q / sum(search+refetch seconds)
                lat_ms_mean = win_query_ms_sum / float(wn)
                window_search_sec = win_query_ms_sum / 1000.0
                qps = (wn / window_search_sec) if window_search_sec > 0 else 0.0
                recall_mean = float(np.mean(win_recalls)) if win_recalls else 0.0

                # write per-window row (one line per window)
                with open(per_window_csv, "a", newline="") as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow([
                        sk, k, n_trees, f"{build_ms:.3f}",
                        w_id, qb, qe, wn,
                        f"{lat_ms_mean:.6f}", f"{qps:.6f}", f"{recall_mean:.6f}",
                        f"{window_search_sec:.6f}",
                    ])

                # write per-query ids (one line per query), outside timing
                with open(ids_csv, "a", newline="") as f:
                    wcsv = csv.writer(f)
                    for row in id_rows:
                        wcsv.writerow(row)

                log(
                    f"[STEP 5] window done: k={k}, search_k={sk}, window_id={w_id}, qb={qb}, qe={qe}, window_q={wn} "
                    f"lat_ms_mean={lat_ms_mean:.6f} qps={qps:.3f} recall_mean={recall_mean:.6f} "
                    f"window_search_sec={window_search_sec:.3f}s"
                )

            grid_t1 = time.perf_counter()
            log(
                f"[STEP 5] ---- Grid done: k={k}, search_k={sk} "
                f"grid_wall_time={(grid_t1 - grid_t0):.3f}s ----"
            )

        log(f"[STEP 5] ===== End topK={k} =====")

    log("[STEP 6] Experiment finished.")
    log("[OUTPUT] Generated files:")
    log(f" - {per_window_csv}")
    log(f" - {ids_csv}")


# -----------------------------
# CLI + 默认参数
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Annoy SIFT-1M (L2) multi-round ANN benchmark")

    # 文件路径
    parser.add_argument("--loadcsv", type=str, default="load_csv/chiristmas/vectors_base.csv",
                        help="Base vectors CSV (id,v0..v127)")
    parser.add_argument("--point_csv", type=str, default="query-csv/selected_query_50.csv",
                        help="Query vectors CSV (id,v0..v127)")
    parser.add_argument("--log_dir", type=str, default="results_log",
                        help="Base output directory for csv logs (will append timestamp subdir)")

    # 参数
    parser.add_argument("--topKs", type=str, default="10,100,200",
                        help="Comma-separated topKs, e.g. 10,100,200")

    # 重要：默认留空 => 自动 search_k = n_trees * k
    parser.add_argument("--search_ks", type=str, default="",
                        help="Comma-separated search_k list. If empty, auto uses search_k = n_trees * k.")

    # [MOD] NEW: per-window aggregation size
    parser.add_argument("--window_size", type=int, default=200,
                        help="Window size for per-window latency/qps/recall aggregation")

    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["angular", "euclidean", "manhattan", "hamming", "dot"],
                        help="Annoy metric (SIFT L2 => euclidean)")
    parser.add_argument("--n_trees", type=int, default=100, help="Number of trees for Annoy build")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Annoy build threads (-1 uses all cores)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Annoy build")

    args = parser.parse_args()

    topKs = parse_int_list(args.topKs)

    # 用户是否显式指定 search_ks？
    search_ks = parse_int_list(args.search_ks)
    use_auto_search_k = False
    if not search_ks:
        use_auto_search_k = True

    log("[MAIN] Parsed args:")
    log(f"       loadcsv={args.loadcsv}")
    log(f"       point_csv={args.point_csv}")
    log(f"       log_dir(base)={args.log_dir}")
    log(f"       topKs={topKs}")
    log(f"       search_ks(user)={search_ks}")
    log(f"       use_auto_search_k={use_auto_search_k}")
    log(f"       window_size={args.window_size}")  # [MOD] NEW
    log(f"       metric={args.metric} n_trees={args.n_trees} n_jobs={args.n_jobs} seed={args.seed}")

    run_experiment(
        loadcsv=args.loadcsv,
        point_csv=args.point_csv,
        log_dir=args.log_dir,
        topKs=topKs,
        search_ks=search_ks,  # 可能为空
        metric=args.metric,
        n_trees=args.n_trees,
        n_jobs=args.n_jobs,
        seed=args.seed,
        use_auto_search_k=use_auto_search_k,
        window_size=args.window_size,  # [MOD] NEW
    )


# -----------------------------
# 使用示例：
#
# 1) 让 search_k 默认采用 n_trees*k（不要传 --search_ks）
# python run_annoy_index_search.py --loadcsv ... --point_csv ... --log_dir ... --topKs 10,100,200 --n_trees 300 --window_size 200
#
# 2) 显式指定 search_k 网格（会覆盖默认）
# python run_annoy_index_search.py --loadcsv ... --point_csv ... --log_dir ... --topKs 10,100,200 --search_ks 30000,60000,120000 --n_trees 300 --window_size 200
# -----------------------------
if __name__ == "__main__":
    main()
