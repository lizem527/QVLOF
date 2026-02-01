# 文件路径
loadcsv = 'load_csv/chiristmas/vectors_base.csv'

# 查询结果保存位置
log_dir= 'results_log'

#point 查询向量文件，根据每一行向量查询表中里该向量最近的k个向量
point_csv = 'query-csv/selected_query_50.csv'


# knn topk 查询表距离向量最近的k个向量
topKs = [10,100,200]  # 查询TopK


# Annoy 的索引是很多棵随机投影树的“森林”，构建参数n_trees、查询参数search_k
# L2 检索：metric="euclidean" ;  余弦检索：metric="angular"（需要向量先做 L2 normalize，才能真正对应 cosine）;  最大内积：metric="dot"
metric = "euclidean"
# 建 n_trees 棵树，越大召回率越高｜构建时间越长 ,  选用50～100
n_trees = 100
# search_k：查询时最多“检查/访问”多少个节点 ;越大越准（搜索更深）,查询更慢,默认大致是 n_trees * n（n 是要的近邻数）默认search_k = -1
# 再试：2x / 5x / 10x 默认值（例如 5k、12.5k、25k） 找到“recall 提升开始变慢”的拐点停住
search_k = 1000

# 默认 n_jobs=-1 build 用满 CPU
n_jobs=-1
seed=42
