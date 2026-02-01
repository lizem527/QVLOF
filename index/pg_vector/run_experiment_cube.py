# run_experiment_cube.py
import os
import csv
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from config_cube import (
    db_config,
    point_csv,
    log_dir,
    image_dir,
    query_table,
    topKs,
    groundtruth_table,
    work_mem,
    maintenance_work_mem,
    gist_fillfactor,
    gist_index_scan_cost,
)

SCHEMA = "public"


def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def read_header_and_dim(csv_path: str):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header or len(header) < 2:
        raise ValueError(f"CSV header 不合法（至少 id+1维）：{csv_path}")
    if header[0].lower() != "id":
        raise ValueError(f"查询 CSV 第一列必须是 id，当前为 {header[0]}：{csv_path}")
    return header, len(header) - 1


def load_query_vectors(file_path: str):
    """
    point_csv 格式：
      id, feature_1, feature_2, ..., feature_d
    d 动态推断
    """
    header, dim = read_header_and_dim(file_path)
    log(f"🔹 加载查询向量: {file_path} | dim={dim}")
    vectors = []

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            vec_part = row[1:1 + dim]
            if len(vec_part) != dim:
                raise ValueError(
                    f"第 {row_idx} 行维度不一致：期望 {dim}，实际 {len(vec_part)}"
                )
            vectors.append(list(map(float, vec_part)))

    log(f"  -> 加载完成，共 {len(vectors)} 个查询向量")
    return vectors, dim


def compute_recall(result_ids, gt_ids):
    if not gt_ids:
        return 0.0
    return len(set(result_ids) & set(gt_ids)) / len(gt_ids)


def find_groundtruth_for_table(q_table: str):
    """
    在 groundtruth_table 中找到 q_table 属于哪个 gt
    """
    for gt, tables in groundtruth_table.items():
        if q_table in tables:
            return gt
    return None


def get_first_col_name_and_value(table_name: str):
    """
    只对 _gist 表写 KNN CSV
    第一列写 fillfactor（构建参数）
    """
    if table_name.endswith("_gist"):
        return "fillfactor", int(gist_fillfactor)
    return None, None


def get_cube_dim_from_db(cursor, table_name: str) -> int:
    """
    cube 维度：cube_dim(vec)
    """
    cursor.execute(f'SELECT cube_dim(vec) FROM "{SCHEMA}"."{table_name}" LIMIT 1;')
    r = cursor.fetchone()
    if not r or r[0] is None:
        raise ValueError(
            f"无法从表 {SCHEMA}.{table_name} 推断 cube_dim(vec)：表为空或 vec 列异常"
        )
    return int(r[0])


def explain_analyze(cursor, sql: str):
    cursor.execute("EXPLAIN ANALYZE " + sql)
    rows = cursor.fetchall()
    plan_lines = [r[0] for r in rows]
    exec_time = -1.0
    for line in plan_lines:
        if "Execution Time" in line:
            # e.g. "Execution Time: 0.123 ms"
            exec_time = float(line.split(":")[1].strip().split()[0])
            break
    return plan_lines, exec_time


def execute_knn_query(cursor, table: str, vec, k: int):
    """
    cube KNN 查询：
      ORDER BY vec <-> cube(ARRAY[...]) LIMIT k
    """
    arr = "ARRAY[" + ",".join(map(str, vec)) + "]"
    sql = f'''
        SELECT id
        FROM "{SCHEMA}"."{table}"
        ORDER BY vec <-> cube({arr})
        LIMIT {k}
    '''
    # EXPLAIN ANALYZE
    plan_lines, exec_time = explain_analyze(cursor, sql.strip())

    # 实际查询拿 id
    cursor.execute(sql)
    ids = [r[0] for r in cursor.fetchall()]
    return ids, exec_time, plan_lines


def run():
    log("=== 🚀 cube + GiST 实验启动（动态维度；仅 _gist 写 KNN CSV；写入 plans） ===")
    ensure_dir(log_dir)
    ensure_dir(image_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cur_log_dir = os.path.join(log_dir, timestamp)
    cur_img_dir = os.path.join(image_dir, timestamp)
    cur_plan_dir = os.path.join(cur_log_dir, "plans")
    ensure_dir(cur_log_dir)
    ensure_dir(cur_img_dir)
    ensure_dir(cur_plan_dir)

    log(f"📁 日志路径: {cur_log_dir}")
    log(f"🖼️ 图片路径: {cur_img_dir}")
    log(f"🧠 EXPLAIN plans: {cur_plan_dir}")

    vectors, q_dim = load_query_vectors(point_csv)

    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # cube 扩展
    cursor.execute("CREATE EXTENSION IF NOT EXISTS cube;")
    conn.commit()

    # session 参数（每次查询前也会 set 一次，这里先 set 一次）
    cursor.execute(f"SET work_mem = '{work_mem}';")
    cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")
    cursor.execute(f"SET cpu_index_tuple_cost = {float(gist_index_scan_cost)};")

    # ✅ 维度校验：对每个 query_table 对应的 gt 表检查一致性
    for t in query_table:
        gt = find_groundtruth_for_table(t)
        if gt is None:
            raise ValueError(
                f"query_table 里的表 {t} 没有出现在 groundtruth_table 的任何列表里，请修正配置。"
            )
        db_dim = get_cube_dim_from_db(cursor, gt)
        log(f"🔎 维度校验：table={t} gt={gt} | query_dim={q_dim}, db_dim={db_dim}")
        if q_dim != db_dim:
            raise ValueError(
                f"维度不一致：point_csv={q_dim}维，但 groundtruth表 {SCHEMA}.{gt} 的 vec 是 {db_dim} 维。"
            )

    for k in topKs:
        log(f"\n=== 🔍 Top{k} 开始 ===")

        recalls_per_table = {t: [] for t in query_table}
        times_per_table = {t: [] for t in query_table}

        detail_txt = os.path.join(cur_log_dir, f"top{k}_details.txt")
        knn_csv = os.path.join(cur_log_dir, f"top{k}_knn_ids.csv")

        with open(detail_txt, "w", encoding="utf-8") as dlog, open(knn_csv, "w", newline="", encoding="utf-8") as cfile:
            dlog.write("QueryIndex,Table,ExecTime(ms),Recall\n")
            writer = csv.writer(cfile)
            writer.writerow(["fillfactor", "k", "query", "id_list"])

            for qi, vec in enumerate(vectors):
                log("\n" + "-" * 70)
                log(f"➡️ query={qi}（0-based）TopK={k}")

                for t in query_table:
                    gt = find_groundtruth_for_table(t)

                    try:
                        # 每次查询前设置 session 参数（保持一致性）
                        cursor.execute(f"SET work_mem = '{work_mem}';")
                        cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")
                        cursor.execute(f"SET cpu_index_tuple_cost = {float(gist_index_scan_cost)};")

                        gt_ids, _, gt_plan = execute_knn_query(cursor, gt, vec, k)
                        ids, ms, plan = execute_knn_query(cursor, t, vec, k)

                        # 写 plan（每条查询一个文件，便于你确认是否 Index Scan）
                        plan_path = os.path.join(cur_plan_dir, f"q{qi}_k{k}_{t}.plan.txt")
                        with open(plan_path, "w", encoding="utf-8") as pf:
                            pf.write("\n".join(plan))

                    except Exception as e:
                        log(f"❌ 查询失败 table={t}: {e}")
                        cursor.execute("ROLLBACK;")
                        continue

                    recall = compute_recall(ids, gt_ids)

                    log(f"📌 {t} (gt={gt}) ⏱️ {ms:.2f} ms | recall={recall:.4f}")
                    recalls_per_table[t].append(recall)
                    times_per_table[t].append(ms)
                    dlog.write(f"{qi},{t},{ms:.2f},{recall:.4f}\n")

                    # ✅ 仅 _gist 表写 KNN CSV
                    col, val = get_first_col_name_and_value(t)
                    if col is not None:
                        writer.writerow([val, k, qi, " ".join(map(str, ids))])

        log(f"🧾 Top{k} KNN CSV（仅 _gist 表）: {knn_csv}")

        # summary md
        summary_md = os.path.join(cur_log_dir, f"top{k}_summary.md")
        with open(summary_md, "w", encoding="utf-8") as s:
            s.write(f"# Top{k} 查询汇总（cube+GiST）\n\n")
            s.write("| Table | Total Time (ms) | Avg Time (ms) | Avg Recall |\n")
            s.write("|-------|----------------|----------------|-------------|\n")
            for t in query_table:
                total = sum(times_per_table[t])
                avg = total / len(times_per_table[t]) if times_per_table[t] else 0.0
                r = sum(recalls_per_table[t]) / len(recalls_per_table[t]) if recalls_per_table[t] else 0.0
                s.write(f"| {t} | {total:.2f} | {avg:.2f} | {r:.4f} |\n")

        # plot
        try:
            x = np.arange(len(query_table))
            totals = [sum(times_per_table[t]) for t in query_table]
            ravg = [
                (sum(recalls_per_table[t]) / len(recalls_per_table[t])) if recalls_per_table[t] else 0.0
                for t in query_table
            ]

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.bar(x, totals)
            ax1.set_ylabel("Total Query Time (ms)")

            ax2 = ax1.twinx()
            ax2.plot(x, ravg, marker="o")
            ax2.set_ylabel("Average Recall")

            plt.xticks(x, query_table, rotation=20, ha="right")
            plt.title(f"Top{k} (cube+GiST)")
            fig.tight_layout()

            out_png = os.path.join(cur_img_dir, f"top{k}_summary.png")
            plt.savefig(out_png)
            plt.close()
            log(f"📊 图片保存: {out_png}")
        except Exception as e:
            log(f"⚠️ 绘图失败: {e}")

    cursor.close()
    conn.close()
    log("✅ 完成")
    log(f"📁 日志目录: {cur_log_dir}")
    log(f"🧠 Plan 目录: {cur_plan_dir}")
    log(f"🖼️ 图片目录: {cur_img_dir}")


if __name__ == "__main__":
    run()
