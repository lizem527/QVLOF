import os
import csv
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from config import (
    db_config,
    point_csv,
    log_dir,
    image_dir,
    query_table,
    topKs,
    ivfflat_probesList,
    hnsw_ef_search,
    groundtruth_table,
    work_mem,
    maintenance_work_mem
)

# ivfflat probes 找不到匹配项时的默认值
ivfflat_probes_default = 35


def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_query_vectors(file_path: str):
    """
    point_csv: 默认第一行是表头
    每行取 row[:128] 作为向量（保持你原逻辑）
    """
    log(f"🔹 加载查询向量: {file_path}")
    vectors = []
    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            vec = list(map(float, row[:128]))
            vectors.append(vec)
    log(f"  -> 加载完成，共 {len(vectors)} 个查询向量")
    return vectors


def compute_recall(result_ids, gt_ids):
    if not gt_ids:
        return 0.0
    return len(set(result_ids) & set(gt_ids)) / len(gt_ids)


def find_groundtruth_table(query_table_name: str):
    for gt_table, derived_tables in groundtruth_table.items():
        if query_table_name in derived_tables:
            return gt_table
    return None


def infer_ivfflat_probes_for_table(table_name: str) -> int:
    """
    严格按表名包含 '_ivfflat' 判断 IVF 表，然后再从 ivfflat_probesList 推断 probes：
    - 若 table_name 是 xxx_ivfflat，则 base_name=xxx
    - 优先 base_name 精确匹配 ivfflat_probesList
    - 否则做子串匹配
    - 找不到就用默认 ivfflat_probes_default
    """
    base_name = table_name
    if base_name.endswith("_ivfflat"):
        base_name = base_name[: -len("_ivfflat")]

    if base_name in ivfflat_probesList:
        return int(ivfflat_probesList[base_name])

    for key, val in ivfflat_probesList.items():
        if key in base_name or base_name in key:
            return int(val)

    return int(ivfflat_probes_default)


def get_first_col_name_and_value(table_name: str):
    """
    严格按表名后缀判断：
      - 包含 '_hnsw'    -> 第一列含义是 ef，值取 config.hnsw_ef_search
      - 包含 '_ivfflat' -> 第一列含义是 ivfflat_probesList，值取 probes
      - 非索引表 -> 返回 (None, None)（不写 CSV）
    """
    if "_hnsw" in table_name:
        return "ef", int(hnsw_ef_search)
    if "_ivfflat" in table_name:
        return "ivfflat_probesList", int(infer_ivfflat_probes_for_table(table_name))
    return None, None


def execute_query(cursor, table: str, vec, k: int, ivfflat_probes_value=None):
    """
    保持 EXPLAIN ANALYZE + 实际查询 逻辑不变
    仅将参数设置判断改为严格 '_ivfflat' / '_hnsw'
    """
    emb_str = ",".join(map(str, vec))

    try:
        cursor.execute(f"SET work_mem = '{work_mem}';")
        cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")

        if "_ivfflat" in table and ivfflat_probes_value is not None:
            cursor.execute(f"SET ivfflat.probes = {ivfflat_probes_value};")
        elif "_hnsw" in table:
            cursor.execute(f"SET hnsw.ef_search = {hnsw_ef_search};")
    except Exception as e:
        log(f"❌ 参数设置失败: {e}")
        cursor.execute("ROLLBACK;")

    query = f"""
        SELECT id FROM {table}
        ORDER BY embedding <-> '[{emb_str}]'
        LIMIT {k};
    """

    try:
        explain_query = f"EXPLAIN ANALYZE {query.strip()}"
        cursor.execute(explain_query)
        plan_lines = cursor.fetchall()
        time_line = [line[0] for line in plan_lines if "Execution Time" in line[0]]
        exec_time = float(time_line[0].split(":")[1].strip().split()[0]) if time_line else -1.0

        cursor.execute(query)
        ids = [row[0] for row in cursor.fetchall()]
        return ids, exec_time
    except Exception as e:
        log(f"❌ 查询失败: {e}")
        cursor.execute("ROLLBACK;")
        return [], -1.0


def run():
    log("=== 🚀 启动查询评估脚本（写 CSV：仅索引表 _ivfflat/_hnsw；不再写 JSON） ===")
    ensure_dir(log_dir)
    ensure_dir(image_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cur_log_dir = os.path.join(log_dir, timestamp)
    cur_img_dir = os.path.join(image_dir, timestamp)
    ensure_dir(cur_log_dir)
    ensure_dir(cur_img_dir)

    log(f"📁 日志保存路径: {cur_log_dir}")
    log(f"🖼️ 图像保存路径: {cur_img_dir}")

    vectors = load_query_vectors(point_csv)

    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    for k in topKs:
        log(f"\n=== 🔍 Top{k} 查询评估开始 ===")

        recalls_per_table = {table: [] for table in query_table}
        times_per_table = {table: [] for table in query_table}

        # 明细 txt（保留）
        detail_txt_path = os.path.join(cur_log_dir, f"top{k}_details.txt")
        with open(detail_txt_path, "w", encoding="utf-8") as detail_log:
            detail_log.write("QueryIndex,Table,ExecTime(ms),Recall\n")

            # ✅ KNN id_list CSV：仅写索引表（_ivfflat/_hnsw）
            knn_csv_path = os.path.join(cur_log_dir, f"top{k}_knn_ids.csv")
            with open(knn_csv_path, "w", newline="", encoding="utf-8") as knn_csv:
                writer = csv.writer(knn_csv)
                writer.writerow(["ef/ivfflat_probesList", "k", "query", "id_list"])

                for i, vec in enumerate(vectors):
                    log("\n" + "-" * 70)
                    log(f"➡️ 查询向量 query={i}（0-based）🔹 当前 TopK: {k}")

                    for table in query_table:
                        gt_table = find_groundtruth_table(table)
                        if gt_table is None:
                            log(f"⚠️ 表 {table} 没有对应 groundtruth，跳过")
                            continue

                        # groundtruth（按你原逻辑：gt_table 通常是 base/flat）
                        gt_ids, _ = execute_query(cursor, gt_table, vec, k)

                        # 当前表的 probes（仅 ivfflat 需要）
                        current_ivfflat_probes = None
                        if "_ivfflat" in table:
                            current_ivfflat_probes = infer_ivfflat_probes_for_table(table)

                        # 实际查询
                        result_ids, exec_time = execute_query(
                            cursor,
                            table,
                            vec,
                            k,
                            ivfflat_probes_value=current_ivfflat_probes
                        )

                        recall = compute_recall(result_ids, gt_ids)

                        log(f"📌 表: {table} (gt={gt_table}) ⏱️ {exec_time:.2f} ms | ✅ recall={recall:.4f}")

                        recalls_per_table[table].append(recall)
                        times_per_table[table].append(exec_time)
                        detail_log.write(f"{i},{table},{exec_time:.2f},{recall:.4f}\n")

                        # ✅ 只写索引表：表名必须包含 _ivfflat 或 _hnsw
                        col_name, col_value = get_first_col_name_and_value(table)
                        if col_name is None:
                            continue  # 非索引表不写入 CSV

                        id_list_str = " ".join(map(str, result_ids))
                        writer.writerow([col_value, k, i, id_list_str])

        log(f"🧾 Top{k} KNN CSV 已保存（仅索引表）: {os.path.join(cur_log_dir, f'top{k}_knn_ids.csv')}")

        # summary md（保留）
        summary_md_path = os.path.join(cur_log_dir, f"top{k}_summary.md")
        with open(summary_md_path, "w", encoding="utf-8") as summary:
            summary.write(f"# Top{k} 查询汇总\n\n")
            summary.write("| Table | Total Time (ms) | Avg Time (ms) | Avg Recall |\n")
            summary.write("|-------|----------------|----------------|-------------|\n")
            for table in query_table:
                total_time = sum(times_per_table[table])
                avg_time = total_time / len(times_per_table[table]) if times_per_table[table] else 0.0
                avg_recall = (
                    sum(recalls_per_table[table]) / len(recalls_per_table[table])
                    if recalls_per_table[table] else 0.0
                )
                summary.write(f"| {table} | {total_time:.2f} | {avg_time:.2f} | {avg_recall:.4f} |\n")

        # 绘图（保留）
        try:
            x = np.arange(len(query_table))
            total_times = [sum(times_per_table[table]) for table in query_table]
            avg_recalls = [
                (sum(recalls_per_table[table]) / len(recalls_per_table[table])) if recalls_per_table[table] else 0.0
                for table in query_table
            ]

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.bar(x, total_times, color="skyblue", label="Total Query Time (ms)")
            ax1.set_ylabel("Total Query Time (ms)", color="skyblue")
            ax1.tick_params(axis="y", labelcolor="skyblue")

            ax2 = ax1.twinx()
            ax2.plot(x, avg_recalls, color="orange", marker="o", label="Average Recall")
            ax2.set_ylabel("Average Recall", color="orange")
            ax2.tick_params(axis="y", labelcolor="orange")

            plt.xticks(x, query_table, rotation=30, ha="right")
            plt.title(f"Top{k} Query - Total Time & Avg Recall")
            fig.tight_layout()

            img_path = os.path.join(cur_img_dir, f"top{k}_summary.png")
            plt.savefig(img_path)
            plt.close()
            log(f"📊 图像保存: {img_path}")
        except Exception as e:
            log(f"⚠️ 绘图失败: {e}，继续执行后续代码")

    cursor.close()
    conn.close()
    log("✅ 所有查询完成，连接已关闭")
    log(f"📁 日志存储在: {cur_log_dir}")
    log(f"🖼️ 图像存储在: {cur_img_dir}")


if __name__ == "__main__":
    run()
