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
    work_mem, maintenance_work_mem

)

def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_query_vectors(file_path):
    log(f"�� 加载查询向量: {file_path}")
    vectors = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            vec = list(map(float, row[:128]))
            vectors.append(vec)
    log(f"  -> 加载完成，共 {len(vectors)} 个查询向量")
    return vectors

def compute_recall(result_ids, gt_ids):
    if not gt_ids:
        return 0.0
    return len(set(result_ids) & set(gt_ids)) / len(gt_ids)

def execute_query(cursor, table, vec, k, verbose=False, ivfflat_probes_value=None):
    emb_str = ','.join(map(str, vec))
    try:
        cursor.execute(f"SET work_mem = '{work_mem}';")
        cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")
        if 'ivfflat' in table and ivfflat_probes_value is not None:
            cursor.execute(f"SET ivfflat.probes = {ivfflat_probes_value};")
        elif 'hnsw' in table:
            cursor.execute(f"SET hnsw.ef_search = {hnsw_ef_search};")
    except Exception as e:
        log(f"❌ 参数设置失败: {e}")
        cursor.execute("ROLLBACK;")

    query = f"""
        SELECT id FROM {table}
        ORDER BY embedding <-> '[{emb_str}]'
        LIMIT {k};
    """
    # 保证计算查询时间的时候要保持与原来代码用的sql一致，要触发回表
    try:
        explain_query = f"EXPLAIN ANALYZE {query.strip()}"
        cursor.execute(explain_query)
        plan_lines = cursor.fetchall()
        time_line = [line[0] for line in plan_lines if "Execution Time" in line[0]]
        exec_time = float(time_line[0].split(':')[1].strip().split()[0]) if time_line else -1.0
        cursor.execute(query)
        ids = [row[0] for row in cursor.fetchall()]
        return ids, exec_time
    except Exception as e:
        log(f"❌ 查询失败: {e}")
        cursor.execute("ROLLBACK;")
        return [], -1.0

def find_groundtruth_table(query_table_name):
    for gt_table, derived_tables in groundtruth_table.items():
        if query_table_name in derived_tables:
            return gt_table
    return None

def run():
    log("=== �� 启动查询评估脚本（含召回率） ===")
    ensure_dir(log_dir)
    ensure_dir(image_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cur_log_dir = os.path.join(log_dir, timestamp)
    cur_img_dir = os.path.join(image_dir, timestamp)
    ensure_dir(cur_log_dir)
    ensure_dir(cur_img_dir)

    log(f"�� 日志保存路径: {cur_log_dir}")
    log(f"��️ 图像保存路径: {cur_img_dir}")

    vectors = load_query_vectors(point_csv)
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    for k in topKs:
        log(f"\n=== �� Top{k} 查询评估开始 ===")
        recalls_per_table = {table: [] for table in query_table}
        times_per_table = {table: [] for table in query_table}

        detail_txt_path = os.path.join(cur_log_dir, f"top{k}_details.txt")
        with open(detail_txt_path, "w") as detail_log:
            detail_log.write("QueryIndex,Table,ExecTime(ms),Recall\n")

            for i, vec in enumerate(vectors):
                log("\n" + "-"*70)
                log(f"\n➡️ 查询第 {i + 1} 个向量 �� 当前 TopK: {k}")

                for table in query_table:
                    gt_table = find_groundtruth_table(table)
                    if gt_table is None:
                        log(f"⚠️ 表 {table} 没有对应的 groundtruth，跳过")
                        continue

                    # 根据表名匹配 ivfflat_probes
                    current_ivfflat_probes = None
                    for key, val in ivfflat_probesList.items():
                        if key in table:
                            current_ivfflat_probes = val
                            break
                    if current_ivfflat_probes is None:
                        current_ivfflat_probes = 25  # 默认值

                    log(f"\n�� 表: {table} (groundtruth 表: {gt_table})")
                    log(f"�� 使用 ivfflat_probes={current_ivfflat_probes}，point_csv={point_csv}")

                    gt_ids, _ = execute_query(cursor, gt_table, vec, k, verbose=True)
                    result_ids, exec_time = execute_query(cursor, table, vec, k, verbose=True, ivfflat_probes_value=current_ivfflat_probes)
                    recall = compute_recall(result_ids, gt_ids)

                    log(f"⏱️ 耗时: {exec_time:.2f} ms")
                    log(f"✅ 召回率: {recall:.4f}")

                    recalls_per_table[table].append(recall)
                    times_per_table[table].append(exec_time)
                    detail_log.write(f"{i},{table},{exec_time:.2f},{recall:.4f}\n")

        # 写 summary
        summary_md_path = os.path.join(cur_log_dir, f"top{k}_summary.md")
        with open(summary_md_path, "w") as summary:
            summary.write(f"# Top{k} 查询汇总\n\n")
            summary.write("| Table | Total Time (ms) | Avg Time (ms) | Avg Recall |\n")
            summary.write("|-------|----------------|----------------|-------------|\n")
            for table in query_table:
                total_time = sum(times_per_table[table])
                avg_time = total_time / len(times_per_table[table]) if times_per_table[table] else 0.0
                avg_recall = sum(recalls_per_table[table]) / len(recalls_per_table[table])
                summary.write(f"| {table} | {total_time:.2f} | {avg_time:.2f} | {avg_recall:.4f} |\n")

        # 绘图部分
        try:
            x = np.arange(len(query_table))
            total_times = [sum(times_per_table[table]) for table in query_table]
            avg_recalls = [sum(recalls_per_table[table]) / len(recalls_per_table[table]) for table in query_table]

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.bar(x, total_times, color='skyblue', label='Total Query Time (ms)')
            ax1.set_ylabel('Total Query Time (ms)', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')

            ax2 = ax1.twinx()
            ax2.plot(x, avg_recalls, color='orange', marker='o', label='Average Recall')
            ax2.set_ylabel('Average Recall', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            plt.xticks(x, query_table, rotation=30, ha='right')
            plt.title(f"Top{k} Query - Total Time & Avg Recall")
            fig.tight_layout()

            img_path = os.path.join(cur_img_dir, f"top{k}_summary.png")
            plt.savefig(img_path)
            plt.close()
            log(f"�� 图像保存: {img_path}")
        except Exception as e:
            log(f"⚠️ 绘图失败: {e}，继续执行后续代码")


    cursor.close()
    conn.close()
    log("✅ 所有查询完成，连接已关闭")
    log(f"�� 日志存储在: {cur_log_dir}")
    log(f"��️ 图像存储在: {cur_img_dir}")

if __name__ == "__main__":
    run()


