#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import csv
import argparse
import numpy as np
import faiss

# =========================================================
# 数据读取
# =========================================================
def load_csv_with_id(path, dtype=np.float32, delimiter=",", skiprows=1):
    """
    CSV 格式: id, v0, v1, ...
    默认跳过首行表头（skiprows=1）。
    返回:
      ids: (N,)
      vecs: (N, d)
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, dtype=dtype)
    if data.ndim == 1:
        # 只有一行时，np.loadtxt 可能返回 1D，强制转 2D
        data = data.reshape(1, -1)

    ids = data[:, 0].astype(np.int64)
    vecs = data[:, 1:].astype(dtype)
    return ids, vecs


# =========================================================
# LSH 投影与动态编码
# =========================================================
class LSHProjector:
    def __init__(self, K=16, L=4, d=128, seed=42):
        np.random.seed(seed)
        self.K = K
        self.L = L
        self.d = d
        self.projections = [np.random.randn(d, K).astype(np.float32) for _ in range(L)]

    def project(self, vec):
        # 返回 L 个投影，每个投影是长度 K 的向量
        return [vec @ proj for proj in self.projections]


class DynamicEncoder:
    def __init__(self, n_bins=256, seed=42):
        self.breakpoints = None
        self.n_bins = n_bins
        self.seed = seed

    def fit(self, values):
        # values: 1D array
        values = np.asarray(values)
        if len(values) == 0:
            raise ValueError("DynamicEncoder.fit received empty values")

        rng = np.random.default_rng(self.seed)
        sample_size = min(100000, len(values))

        # 如果 values 太小，replace=False 也可；如果 sample_size==len(values)，等价全取
        sampled = rng.choice(values, size=sample_size, replace=False)

        # 生成 n_bins+1 个分位点
        self.breakpoints = np.quantile(sampled, np.linspace(0, 1, self.n_bins + 1))

        # 防止breakpoints全部相同导致 searchsorted 异常
        if np.all(self.breakpoints == self.breakpoints[0]):
            # 退化处理：人为构造一个微小区间
            eps = 1e-6
            self.breakpoints = self.breakpoints + np.linspace(-eps, eps, self.n_bins + 1)

    def encode(self, val):
        if self.breakpoints is None:
            raise RuntimeError("DynamicEncoder.encode called before fit()")
        # searchsorted 返回插入点 [0..n_bins+1]，减 1 得到 [ -1 .. n_bins ]
        code = int(np.searchsorted(self.breakpoints, val, side="right") - 1)
        # clip 到 [0, n_bins-1]
        return max(0, min(self.n_bins - 1, code))


# =========================================================
# DETree（动态编码树）
# =========================================================
class DETreeNode:
    def __init__(self, depth=0, max_leaf_size=1000):
        self.children = {}
        self.points = []      # 只在叶子存 idx
        self.depth = depth
        self.max_leaf_size = max_leaf_size
        self.split_dim = None

    def insert(self, code, idx):
        """
        code: tuple[int] 长度为 K
        idx:  数据行号
        """
        node = self
        while True:
            # 叶子条件：容量未满 或 depth 已经到尽头
            if len(node.points) < node.max_leaf_size or node.depth >= len(code):
                node.points.append(idx)
                return

            if node.split_dim is None:
                node.split_dim = node.depth % len(code)

            split_val = code[node.split_dim]
            if split_val not in node.children:
                node.children[split_val] = DETreeNode(depth=node.depth + 1, max_leaf_size=node.max_leaf_size)
            node = node.children[split_val]


class DETLSHIndex:
    def __init__(self, data, K=16, L=4, n_bins=256, max_leaf_size=1000, seed=42):
        """
        data: (N, d) numpy array
        """
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError("data must be 2D array (N, d)")

        self.K = int(K)
        self.L = int(L)
        self.n_bins = int(n_bins)
        self.max_leaf_size = int(max_leaf_size)
        self.seed = int(seed)

        self.projector = LSHProjector(K=self.K, L=self.L, d=self.data.shape[1], seed=self.seed)
        self.encoders = [[DynamicEncoder(n_bins=self.n_bins, seed=self.seed + 7 + l * 997 + k * 131)
                          for k in range(self.K)]
                         for l in range(self.L)]
        self.trees = [DETreeNode(depth=0, max_leaf_size=self.max_leaf_size) for _ in range(self.L)]

    def build(self, log_every=100000):
        """
        构建索引：拟合每个(l,k)编码器的分位点，并插入 DETree
        """
        N = self.data.shape[0]

        # 先对所有向量做投影（会占内存：N * L * K）
        projected_data = [self.projector.project(vec) for vec in self.data]

        # 拟合每个 encoder 的 breakpoints
        for l in range(self.L):
            for k in range(self.K):
                dim_values = np.array([proj[l][k] for proj in projected_data], dtype=np.float32)
                self.encoders[l][k].fit(dim_values)

        # 插入树
        for idx, projs in enumerate(projected_data):
            for l in range(self.L):
                code = tuple(self.encoders[l][k].encode(projs[l][k]) for k in range(self.K))
                self.trees[l].insert(code, idx)

            if log_every and idx % log_every == 0:
                print(f"[BUILD] Inserted {idx}/{N}")

    def query(self, q_vec, epsilon=0.1, top_k=20):
        """
        q_vec: (d,)
        epsilon: 桶扩展比例，实际阈值 = epsilon * n_bins
        返回: [(idx, dist), ...] 按 dist 升序
        """
        q_vec = np.asarray(q_vec, dtype=np.float32)
        if q_vec.ndim != 1 or q_vec.shape[0] != self.data.shape[1]:
            raise ValueError(f"q_vec shape must be (d,) with d={self.data.shape[1]}")

        q_projs = self.projector.project(q_vec)
        candidates = set()

        thr = float(epsilon) * float(self.n_bins)

        for l in range(self.L):
            q_code = tuple(self.encoders[l][k].encode(q_projs[l][k]) for k in range(self.K))

            stack = [self.trees[l]]
            while stack:
                node = stack.pop()
                if node.children:
                    split_dim = node.split_dim
                    query_val = q_code[split_dim]
                    for child_key, child_node in node.children.items():
                        if abs(child_key - query_val) <= thr:
                            stack.append(child_node)
                else:
                    candidates.update(node.points)

        # 计算候选集的真实 L2 距离
        if not candidates:
            return []

        cand_list = list(candidates)
        # 向量化加速距离计算
        cand_vecs = self.data[cand_list]
        diffs = cand_vecs - q_vec
        dists = np.linalg.norm(diffs, axis=1)

        pairs = list(zip(cand_list, dists.tolist()))
        pairs.sort(key=lambda x: x[1])
        return pairs[:int(top_k)]


# =========================================================
# 输出 TopK 到 CSV
# =========================================================
def dump_topk_for_queries_to_csv(index: DETLSHIndex,
                                 base_ids: np.ndarray,
                                 query_ids: np.ndarray,
                                 queries: np.ndarray,
                                 out_csv: str,
                                 top_k=50,
                                 epsilon=0.1,
                                 preview_dims=8):
    """
    输出 CSV: 每行是 (qid, rank, idx(row), base_id, dist, vec_preview...)
    qid 使用 query_ids（真实id）
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    preview_dims = int(preview_dims)
    preview_dims = max(0, min(preview_dims, index.data.shape[1]))

    fieldnames = ["qid", "rank", "idx", "base_id", "dist"] + [f"v{i}" for i in range(preview_dims)]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for qi in range(len(queries)):
            q_vec = queries[qi]
            hits = index.query(q_vec, epsilon=epsilon, top_k=top_k)  # [(idx, dist), ...]

            for rank, (idx, dist) in enumerate(hits, start=1):
                row = {
                    "qid": int(query_ids[qi]),
                    "rank": int(rank),
                    "idx": int(idx),                 # base 行号
                    "base_id": int(base_ids[idx]),   # base 的真实 id
                    "dist": float(dist),
                }
                if preview_dims > 0:
                    vec_preview = index.data[idx][:preview_dims]
                    for i in range(preview_dims):
                        row[f"v{i}"] = float(vec_preview[i])

                writer.writerow(row)

    print(f"[DUMP] dumped top{top_k} for {len(queries)} queries -> {out_csv}")


# =========================================================
# 参数：默认值在代码结尾定义，命令行可覆盖
# =========================================================
def parse_args(defaults: dict):
    parser = argparse.ArgumentParser(description="DET-LSH runner (params defined at end, CLI overrides)")

    parser.add_argument("--base_csv", type=str, default=defaults["base_csv"],
                        help="Base vectors CSV (id,v0,...)")
    parser.add_argument("--query_csv", type=str, default=defaults["query_csv"],
                        help="Query vectors CSV (id,v0,...)")
    parser.add_argument("--out_csv", type=str, default=defaults["out_csv"],
                        help="Output CSV path")

    parser.add_argument("--K", type=int, default=defaults["K"], help="LSH K (projection dim per table)")
    parser.add_argument("--L", type=int, default=defaults["L"], help="Number of LSH tables")
    parser.add_argument("--n_bins", type=int, default=defaults["n_bins"], help="Number of dynamic bins")
    parser.add_argument("--max_leaf_size", type=int, default=defaults["max_leaf_size"], help="Max leaf size")

    parser.add_argument("--top_k", type=int, default=defaults["top_k"], help="Top-K for ANN search & eval")
    parser.add_argument("--epsilon", type=float, default=defaults["epsilon"], help="Bucket expansion ratio")
    parser.add_argument("--preview_dims", type=int, default=defaults["preview_dims"], help="Preview dims in dump CSV")

    parser.add_argument("--skiprows", type=int, default=defaults["skiprows"],
                        help="CSV skiprows (1 means has header).")
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed")

    return parser.parse_args()


# =========================================================
# 主流程
# =========================================================
def main(args):
    print("========== DET-LSH CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("===================================")

    # 1) load base / query
    base_ids, sift_base = load_csv_with_id(args.base_csv, skiprows=args.skiprows)
    query_ids, queries = load_csv_with_id(args.query_csv, skiprows=args.skiprows)

    if sift_base.shape[1] != queries.shape[1]:
        raise ValueError(f"Dim mismatch: base dim={sift_base.shape[1]}, query dim={queries.shape[1]}")

    # 2) build DET-LSH index
    start_build = time.time()
    index = DETLSHIndex(
        sift_base,
        K=args.K,
        L=args.L,
        n_bins=args.n_bins,
        max_leaf_size=args.max_leaf_size,
        seed=args.seed,
    )
    index.build()
    build_time_s = time.time() - start_build

    # 3) exact ground truth (faiss flat)
    exact_index = faiss.IndexFlatL2(sift_base.shape[1])
    exact_index.add(sift_base)
    true_distances, true_knn_indices = exact_index.search(queries, k=args.top_k)

    # 4) ANN query & metrics
    recalls, overall_ratios = [], []
    total_query_time_ms = 0.0

    for q_vec, true_indices in zip(queries, true_knn_indices):
        start_query = time.time()
        det_result = index.query(q_vec, epsilon=args.epsilon, top_k=args.top_k)
        total_query_time_ms += (time.time() - start_query) * 1000

        det_indices = [i for i, _ in det_result]

        # recall@k
        true_set = set(true_indices.tolist())
        det_set = set(det_indices)
        recalls.append(len(det_set & true_set) / (len(true_set) if len(true_set) else 1))

        # overall ratio
        ann_distance_sum = sum(dist for _, dist in det_result)

        exact_distances = np.linalg.norm(sift_base[true_indices] - q_vec, axis=1)
        exact_distance_sum = float(np.sum(exact_distances))

        overall_ratios.append(ann_distance_sum / (exact_distance_sum + 1e-12))

    # 5) print stats
    print(f"\n[Index Build Time] {build_time_s:.2f} s")
    print(f"[Total ANN Query Time] {total_query_time_ms:.2f} ms")
    print(f"[Recall@{args.top_k}] max={max(recalls):.4f} min={min(recalls):.4f} avg={np.mean(recalls):.4f}")
    print(f"[Overall Ratio] max={max(overall_ratios):.4f} min={min(overall_ratios):.4f} avg={np.mean(overall_ratios):.4f}")

    # 6) dump topk results
    dump_topk_for_queries_to_csv(
        index=index,
        base_ids=base_ids,
        query_ids=query_ids,
        queries=queries,
        out_csv=args.out_csv,
        top_k=args.top_k,
        epsilon=args.epsilon,
        preview_dims=args.preview_dims,
    )


# =========================================================
# ✅ 代码 默认参数，命令行可覆盖
# =========================================================
DEFAULT_CONFIG = {
    # 文件路径：base / query / out
    "base_csv": "/hd1/student/lzm/FANNS/Chiristmas/results_5w/f7_ag_verti_sorted.csv",
    "query_csv": "/hd1/student/lzm/FANNS/Chiristmas/selected_query.csv",
    "out_csv": "results/5w_f7.csv",

    # 索引参数
    "K": 16,
    "L": 4,
    "n_bins": 256,
    "max_leaf_size": 1000,

    # 查询与输出参数
    "top_k": 50,
    "epsilon": 0.1,
    "preview_dims": 8,

    # CSV 读取（1 表示有表头）
    "skiprows": 1,

    # 随机种子
    "seed": 42,
}


if __name__ == "__main__":
    args = parse_args(DEFAULT_CONFIG)
    main(args)

