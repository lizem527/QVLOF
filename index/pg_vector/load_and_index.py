import os
import time
import psycopg2
import pandas as pd
from datetime import timedelta
from io import StringIO
from config import (
    db_config,
    fanncsv_dir,
    lists,
    ivfflat_probes,
    m,
    ef_construction,
    hnsw_ef_search,
    work_mem,                 # 从 config.py 导入
    maintenance_work_mem,     # 从 config.py 导入
)

# 表特定 ivfflat_probes 设置
ivfflat_probes_table = {
    'vectors_base': 35,
    'baseline_vectors': 30,
    'method2_vectors': 30,
    'method3_vectors': 25,
    'method6_vectors_2': 30,
    'method6_vectors_3': 2,
}

def format_time(seconds):
    return str(timedelta(seconds=round(seconds)))

def execute_sql(cursor, sql, description):
    print(f"[SQL执行] {description} ...")
    start = time.time()
    cursor.execute(sql)
    elapsed = time.time() - start
    print(f"[SQL完成] {description}，耗时：{format_time(elapsed)}\n")

def process_csv_file(csv_path, table_name, conn):
    print(f"\n==== 开始处理文件：{csv_path} 对应表名：{table_name} ====")
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        print(f"[跳过] 文件为空：{csv_path}")
        return

    embedding_dim = df.shape[1] - 1
    print(f"[读取数据] 完成，读取 {len(df)} 条向量，推断向量维度为 {embedding_dim}")

    #table_ivfflat_probes = ivfflat_probes_table.get(table_name, ivfflat_probes)
    table_ivfflat_probes = 35

    with conn.cursor() as cursor:
        # 删除旧表并创建新表
        execute_sql(cursor, f"DROP TABLE IF EXISTS {table_name};", f"删除旧表 {table_name}")
        create_sql = f"""
            CREATE TABLE {table_name} (
                id BIGINT,
                embedding VECTOR({embedding_dim}),
                num BIGINT
            );
        """
        execute_sql(cursor, create_sql, f"创建表 {table_name}")

        # COPY 批量插入
        start_insert = time.time()
        buffer = StringIO()
        for i, row in df.iterrows():
            vec = [str(float(v)) for v in row.iloc[1:1+embedding_dim].tolist()]
            vec_str = "[" + ",".join(vec) + "]"
            buffer.write(f"{int(row.iloc[0])}\t{vec_str}\t{i}\n")
        buffer.seek(0)
        cursor.copy_from(buffer, table_name, sep="\t", columns=("id", "embedding", "num"))
        conn.commit()  # 每表提交
        elapsed_insert = time.time() - start_insert
        print(f"[插入数据] COPY 完成 {len(df)} 条，耗时：{format_time(elapsed_insert)}")

        # IVFFlat 索引
        ivfflat_table = table_name + "_ivfflat"
        execute_sql(cursor, f"DROP TABLE IF EXISTS {ivfflat_table};", f"删除旧表 {ivfflat_table}")
        execute_sql(cursor, f"CREATE TABLE {ivfflat_table} AS TABLE {table_name};", f"复制表为 {ivfflat_table}")

        cursor.execute(f"SET work_mem = '{work_mem}';")
        cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")
        execute_sql(cursor, f"""
            CREATE INDEX ON {ivfflat_table}
            USING ivfflat (embedding vector_l2_ops)
            WITH (lists = {lists});
        """, f"创建 IVFFlat 索引 (lists={lists})")
        cursor.execute(f"SET ivfflat.probes = {table_ivfflat_probes};")
        print(f"[设置参数] ivfflat.probes = {table_ivfflat_probes}")
        conn.commit()

        # HNSW 索引
        hnsw_table = table_name + "_hnsw"
        execute_sql(cursor, f"DROP TABLE IF EXISTS {hnsw_table};", f"删除旧表 {hnsw_table}")
        execute_sql(cursor, f"CREATE TABLE {hnsw_table} AS TABLE {table_name};", f"复制表为 {hnsw_table}")

        cursor.execute(f"SET work_mem = '{work_mem}';")
        cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")
        execute_sql(cursor, f"""
            CREATE INDEX ON {hnsw_table}
            USING hnsw (embedding vector_l2_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
        """, f"创建 HNSW 索引 (m={m}, ef_construction={ef_construction})")
        cursor.execute(f"SET hnsw.ef_search = {hnsw_ef_search};")
        print(f"[设置参数] hnsw.ef_search = {hnsw_ef_search}")
        conn.commit()

def main():
    conn = psycopg2.connect(
        dbname=db_config.get("database") or db_config.get("dbname"),
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
    )

    csv_files = [f for f in os.listdir(fanncsv_dir) if f.endswith('.csv')]
    print(f"\n📁 共找到 {len(csv_files)} 个 CSV 文件：{csv_files}")

    for idx, filename in enumerate(csv_files, 1):
        csv_path = os.path.join(fanncsv_dir, filename)
        table_name = os.path.splitext(filename)[0]
        print(f"\n========== 开始处理第 {idx}/{len(csv_files)} 个文件: {filename} ==========")
        process_csv_file(csv_path, table_name, conn)
        print(f"========== 完成处理文件: {filename} ==========\n")

    conn.close()
    print("\n✅ 所有 CSV 文件处理完成。")

if __name__ == "__main__":
    main()

