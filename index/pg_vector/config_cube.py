# config.py

# =========================
# 文件路径
# =========================
fanncsv_dir = 'loadcsvdir/load_csv/cube'            # 导入数据库的 CSV 文件夹
point_csv  = 'query-csv/uniform_query_10w_6dim.csv' # 查询向量 CSV（id + feature_*）

# 输出目录
log_dir   = 'querytime_log'
image_dir = 'image'

# =========================
# 查询设置
# =========================
topKs = [100]

# cube + GiST 对照表（base 是 cube 表；_gist 是 GiST 索引表）
query_table = [
    'data_vec6_gist',
]

# groundtruth 映射（按你原逻辑）
groundtruth_table = {
    'data_vec4': ['data_vec4', 'data_vec4_gist'],
    'data_vec6': ['data_vec6', 'data_vec6_gist'],
}

# =========================
# 数据库连接信息
# =========================
db_config = {
    "database": "gauss_cube",
    "user": "postgres",
    "password": "1234qwer",
    "host": "localhost",
    "port": "5432"
}

# =========================
# 内存参数
# =========================
work_mem = '8GB'
maintenance_work_mem = '8GB'

# =========================
# GiST（cube）参数：一个构建，一个查询
# =========================
# 构建参数：GiST 索引页填充率（70~100）
gist_fillfactor = 75

# 查询参数：planner 成本（影响是否更倾向走 GiST 索引；越小越倾向）
gist_index_scan_cost = 1.0

# （可选）构建并行与 buffering
gist_parallel_workers = 4
gist_buffering = 'on'
