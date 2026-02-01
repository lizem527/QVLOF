# load_index_cube.py
import time
import psycopg2
from datetime import timedelta

from config_cube import (
    db_config,
    work_mem,
    maintenance_work_mem,
    gist_fillfactor,
    gist_buffering,
    # 你 config 里还有 gist_parallel_workers，但 GiST 不支持索引参数 parallel_workers
    # 我们改为使用 max_parallel_maintenance_workers（session 参数）
    gist_parallel_workers,
    groundtruth_table,
)

SCHEMA = "public"


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def execute_sql(cursor, sql: str, desc: str):
    print(f"[SQL执行] {desc} ...")
    t0 = time.time()
    cursor.execute(sql)
    print(f"[SQL完成] {desc}，耗时：{format_time(time.time() - t0)}\n")


def ensure_base_table_has_cube_vec(cursor, base_table: str):
    """
    校验 base 表存在、且有 vec 列并且类型为 cube
    """
    cursor.execute(
        """
        SELECT t.typname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid
        JOIN pg_type t ON t.oid = a.atttypid
        WHERE n.nspname = %s
          AND c.relname = %s
          AND a.attname = 'vec'
          AND c.relkind = 'r';
        """,
        (SCHEMA, base_table),
    )
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(
            f"base 表 {SCHEMA}.{base_table} 不存在，或没有 vec 列。\n"
            f"请在 psql 执行：\\d {SCHEMA}.{base_table} 检查是否有 vec cube 列。"
        )
    if row[0] != "cube":
        raise RuntimeError(
            f"base 表 {SCHEMA}.{base_table} 的 vec 列类型不是 cube，而是 {row[0]}。"
        )


def main():
    conn = psycopg2.connect(
        dbname=db_config.get("database") or db_config.get("dbname"),
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
    )
    # 建索引 DDL 用 autocommit 更稳
    conn.autocommit = True

    base_tables = list(groundtruth_table.keys())
    print(f"✅ 将按 groundtruth_table 的 key 创建 GiST 表：{base_tables}")

    with conn.cursor() as cursor:
        # 确保 cube 扩展
        execute_sql(cursor, "CREATE EXTENSION IF NOT EXISTS cube;", "确保 cube 扩展已安装")

        # 会话内存参数
        cursor.execute(f"SET work_mem = '{work_mem}';")
        cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")

        # ✅ GiST 不支持 WITH(parallel_workers=...)
        # ✅ 正确做法：设置并行维护 worker（是否并行由 PG 决定）
        try:
            cursor.execute(f"SET max_parallel_maintenance_workers = {int(gist_parallel_workers)};")
            print(f"[设置参数] max_parallel_maintenance_workers = {int(gist_parallel_workers)}")
        except Exception as e:
            # 某些版本/权限可能不允许 set，这里不致命
            print(f"[警告] 无法设置 max_parallel_maintenance_workers：{e}")

        for idx, base in enumerate(base_tables, 1):
            gist_table = f"{base}_gist"
            gist_index = f"{gist_table}__vec_gist_idx"

            print(f"\n========== {idx}/{len(base_tables)}: {base} -> {gist_table} ==========")

            # 校验 base 表结构
            ensure_base_table_has_cube_vec(cursor, base)

            # 复制 base -> gist
            execute_sql(
                cursor,
                f'DROP TABLE IF EXISTS "{SCHEMA}"."{gist_table}";',
                f"删除旧表 {gist_table}"
            )
            execute_sql(
                cursor,
                f'CREATE TABLE "{SCHEMA}"."{gist_table}" AS TABLE "{SCHEMA}"."{base}";',
                f"复制 base 表为 {gist_table}"
            )

            # 建索引
            execute_sql(
                cursor,
                f'DROP INDEX IF EXISTS "{SCHEMA}"."{gist_index}";',
                f"删除旧索引 {gist_index}"
            )

            # ✅ GiST 兼容参数：fillfactor / buffering
            create_index_sql = f'''
                CREATE INDEX "{gist_index}"
                ON "{SCHEMA}"."{gist_table}"
                USING gist (vec)
                WITH (
                    fillfactor = {int(gist_fillfactor)},
                    buffering = {gist_buffering}
                );
            '''
            execute_sql(
                cursor,
                create_index_sql,
                f"创建 GiST 索引（fillfactor={gist_fillfactor}, buffering={gist_buffering}）"
            )

            print(f"[完成] base={base} gist={gist_table} index={gist_index}")

    conn.close()
    print("\n✅ 全部 GiST 索引表创建完成。")


if __name__ == "__main__":
    main()
