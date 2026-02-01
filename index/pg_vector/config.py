# 文件路径
# fanncsv_dir = 'loadcsvdir/load_csv_vold'  # FANN 待加载到pg数据库的CSV文件夹路径
# fanncsv_dir = 'loadcsvdir/load_csv_vnew'  # FANN 待加载到pg数据库的CSV文件夹路径
# fanncsv_dir = 'loadcsvdir/load-vold-0914'
fanncsv_dir = 'loadcsvdir/load_chiristmas'

# 结果保存位置
log_dir= 'querytime_log'
image_dir= 'image'

#point
# point_csv = 'query-csv/Query_method3.csv'     # point向量 CSV文件路径
# point_csv = 'query-csv/selected_query.csv'     # point向量 CSV文件路径

# point_csv = 'query-csv/fann_a_nog/selected_query_trimmed.csv'     # point向量 CSV文件路径
# point_csv = f'/hd1/workspace/vector-demo/fanns/query-csv/vnew/uniform_query_4885_trimmed.csv'

amount = '100w'
dims = '128dims'
# hnsw  ivfflat
index_type = 'hnsw'
#point_csv = f'query-csv/0915/sampled_query_{amount}.csv'
point_csv = f'query-csv/chiristmas/selected_query_trimmed.csv'




#groundtruth
groundtruth_dir = 'groundtruth/groundtruth_universal'     #通用的groundtruth

topKs = [100]  # 查询TopK



#适配多数据量
#query_table = [f'base_vectors_{amount}_{index_type}',f'baseline_vectors_{amount}_{index_type}',f'method6_3_vectors_{amount}_{index_type}']
# query_table = [f'base_vectors_{amount}_{index_type}',f'baseline_vectors_{amount}_{index_type}',f'method6_3_vectors_{amount}_{index_type}']
# query_table = [f'base_vectors_{index_type}',f'baseline_vectors_{index_type}',f'improved_centroid_dist_vectors_{index_type}',f'improved_random_vectors_{index_type}',f'improved_size_desc_vectors_{index_type}']
# query_table = [f'base_vectors_{index_type}'] + [
#     f'improved_order_{i}_vectors_{index_type}' for i in range(31)
# ]

#hnsw
#query_table = ['base_vectors_100w_hnsw','baseline_vectors_100w_hnsw','method6_3_vectors_100w_hnsw','base_vectors_50w_hnsw','baseline_vectors_50w_hnsw','method6_3_vectors_50w_hnsw','base_vectors_20w_hnsw','baseline_vectors_20w_hnsw','method6_3_vectors_20w_hnsw','base_vectors_10w_hnsw','baseline_vectors_10w_hnsw','method6_3_vectors_10w_hnsw']
#query_table = [f'base_vectors_{amount}_hnsw','baseline_vectors_{amount}_hnsw','method6_3_vectors_{amount}_hnsw']

#IVFFlat
# query_table = ['base_vectors_100w_ivfflat','baseline_vectors_100w_ivfflat','method6_3_vectors_100w_ivfflat','base_vectors_50w_ivfflat','baseline_vectors_50w_ivfflat','method6_3_vectors_50w_ivfflat','base_vectors_20w_ivfflat','baseline_vectors_20w_ivfflat','method6_3_vectors_20w_ivfflat','base_vectors_10w_ivfflat','baseline_vectors_10w_ivfflat','method6_3_vectors_10w_ivfflat']
#query_table = ['base_vectors_100w_ivfflat']
#query_table = ['base_vectors_50w_ivfflat']
#query_table = ['base_vectors_20w_ivfflat']
#query_table = ['base_vectors_10w_ivfflat']

# chiristmas
# query_table = ['base_vectors_ivfflat','f2_zorder_sorted_ivfflat','f3_hilbert_sorted_ivfflat','f4_idistance_sorted_ivfflat','f5_kmeans_sorted_ivfflat','f6_nog_sorted_ivfflat']


# query_table = ['base_vectors_hnsw','baselinewithcov_vectors_hnsw','method3withcov_vectors_hnsw']
#query_table = ['base_vectors_ivfflat', 'f1_ag1_query_sorted_ivfflat', 'f1_ag2_query_sorted_ivfflat', 'f2_zorder_sorted_ivfflat',
#               'f3_hilbert_sorted_ivfflat', 'f4_idistance_sorted_ivfflat', 'f5_kmeans_sorted_ivfflat', 'f6_nog_sorted_ivfflat',
#               'f7_averti_sorted_ivfflat']

#query_table = ['base_vectors_hnsw', 'f1_ag1_query_sorted_hnsw', 'f1_ag2_query_sorted_hnsw', 'f2_zorder_sorted_hnsw',
#               'f3_hilbert_sorted_hnsw', 'f4_idistance_sorted_hnsw', 'f5_kmeans_sorted_hnsw', 'f6_nog_sorted_hnsw',
#               'f7_averti_sorted_hnsw']


#query_table = ['base_vectors_hnsw', 'f4_idistance_sorted_hnsw', 'f7_averti_sorted_hnsw','f_rabbitq_hnsw']
query_table = ['base_vectors_ivfflat', 'f4_idistance_sorted_ivfflat', 'f7_averti_sorted_ivfflat','f_rabbitq_ivfflat']


use_groundtruth_table = ['vectors_base','baseline_vectors_hnsw']
# groundtruth_table = {
    
#     'base_vectors_100w':['base_vectors_100w','base_vectors_100w_ivfflat','base_vectors_100w_hnsw'],
#     'baseline_vectors_100w':['baseline_vectors_100w','baseline_vectors_100w_ivfflat','baseline_vectors_100w_hnsw'],
#     'method6_3_vectors_100w':['method6_3_vectors_100w','method6_3_vectors_100w_ivfflat','method6_3_vectors_100w_hnsw'],
#     'base_vectors_50w':['base_vectors_50w','base_vectors_50w_ivfflat','base_vectors_50w_hnsw'],
#     'baseline_vectors_50w':['baseline_vectors_50w','baseline_vectors_50w_ivfflat','baseline_vectors_50w_hnsw'],
#     'method6_3_vectors_50w':['method6_3_vectors_50w','method6_3_vectors_50w_ivfflat','method6_3_vectors_50w_hnsw'],
#     'base_vectors_20w':['base_vectors_20w','base_vectors_20w_ivfflat','base_vectors_20w_hnsw'],
#     'baseline_vectors_20w':['baseline_vectors_20w','baseline_vectors_20w_ivfflat','baseline_vectors_20w_hnsw'],
#     'method6_3_vectors_20w':['method6_3_vectors_20w','method6_3_vectors_20w_ivfflat','method6_3_vectors_20w_hnsw'],
#     'base_vectors_10w':['base_vectors_10w','base_vectors_10w_ivfflat','base_vectors_10w_hnsw'],
#     'baseline_vectors_10w':['baseline_vectors_10w','baseline_vectors_10w_ivfflat','baseline_vectors_10w_hnsw'],
#     'method6_3_vectors_10w':['method6_3_vectors_10w','method6_3_vectors_10w_ivfflat','method6_3_vectors_10w_hnsw'],


#     'base_vectors':['base_vectors','base_vectors_ivfflat','base_vectors_hnsw'],
#     'baseline_vectors':['baseline_vectors','baseline_vectors_ivfflat','baseline_vectors_hnsw'],
#     'improved_centroid_dist_vectors':['improved_centroid_dist_vectors','improved_centroid_dist_vectors_ivfflat','improved_centroid_dist_vectors_hnsw'],
#     'improved_random_vectors':['improved_random_vectors','improved_random_vectors_ivfflat','improved_random_vectors_hnsw'],
#     'improved_size_desc_vectors':['improved_size_desc_vectors','improved_size_desc_vectors_ivfflat','improved_size_desc_vectors_hnsw'],
# }

# groundtruth_table = {
#     'base_vectors': [f'base_vectors{suffix}' for suffix in ['', '_ivfflat', '_hnsw']],
# }
# groundtruth_table.update({
#     f'improved_order_{i}_vectors': [
#         f'improved_order_{i}_vectors{suffix}' for suffix in ['', '_ivfflat', '_hnsw']
#     ] for i in range(31)
# })

groundtruth_table = {

    'base_vectors':['base_vectors','base_vectors_ivfflat','base_vectors_hnsw'],
    'f2_zorder_sorted':['f2_zorder_sorted','f2_zorder_sorted_ivfflat','f2_zorder_sorted_hnsw'],
    'f3_hilbert_sorted':['f3_hilbert_sorted','f3_hilbert_sorted_ivfflat','f3_hilbert_sorted_hnsw'],
    'f4_idistance_sorted':['f4_idistance_sorted','f4_idistance_sorted_ivfflat','f4_idistance_sorted_hnsw'],
    'f5_kmeans_sorted':['f5_kmeans_sorted','f5_kmeans_sorted_ivfflat','f5_kmeans_sorted_hnsw'],
    'f6_nog_sorted':['f6_nog_sorted','f6_nog_sorted_ivfflat','f6_nog_sorted_hnsw'],
    'f7_averti_sorted':['f7_averti_sorted','f7_averti_sorted_ivfflat','f7_averti_sorted_hnsw'],
    
    'method6_vectors_3':['method6_vectors_3','method6_vectors_3_ivfflat','method6_vectors_3_hnsw'],

    'baselinewithcov_vectors':['baselinewithcov_vectors','method6_vectors_3_ivfflat','baselinewithcov_vectors_hnsw'],
    'method3withcov_vectors':['method3withcov_vectors','method6_vectors_3_ivfflat','method3withcov_vectors_hnsw'],

    'f1_ag1_query_sorted':['f1_ag1_query_sorted','f1_ag1_query_sorted_ivfflat','f1_ag1_query_sorted_hnsw'],
    'f1_ag2_query_sorted':['f1_ag2_query_sorted','f1_ag2_query_sorted_ivfflat','f1_ag2_query_sorted_hnsw'],
    
    'f_rabbitq': ['f_rabbitq', 'f_rabbitq_hnsw', 'f_rabbitq_ivfflat'],
}



recall_verify_csv = 'recall_verify/DETLSH/result/method6_3_query.csv'     #通用的groundtruth
recall_query_csv = 'query-csv/fann_a_nog/selected_query_6_trimmed.csv'  # 用来查询的point向量 CSV文件路径
recall_query_table = 'method6_vectors_3'
recall_topKs = 20     #通用的groundtruth

# 数据库连接信息  fanns_db  fanns_a_notg fanns_exp_new  fanns_chiristmas
db_config = {
    
    "database": "fanns_chiristmas",
    "user": "postgres",
    "password": "1234qwer",
    "host": "localhost",
    "port": "5432"
}

geneIVF_table = ['vectors_base','baseline_vectors','method6_vectors_3']
# IVFFlat 索引参数
# IVFFlat 索引构建的参数
# 将所有向量划分为几个簇(倒排表)；值越大 查询时定位更精确
# lists = 100
lists = 100

"""
	原本都设置一样的ivfflat_probes = 2,现在控制平均召回率在接近99.99情况下，调整每一个表ivfflat_probes的值。
	现在选择能使召回率达到99%的ivfflat_probes最低值
    vectors_base_ivfflat				35
    baseline_vectors_ivfflat		30
    method2_vectors_ivfflat			30
    method3_vectors_ivfflat			25
    method6_vectors_2_ivfflat		30
    method6_vectors_3_ivfflat		2
    vectors_base_ivfflat				35
    baseline_vectors_ivfflat		30
    method2_vectors_ivfflat			30
    method3_vectors_ivfflat			25
    method6_vectors_2_ivfflat		30
    method6_vectors_3_ivfflat		
"""

# 最多从 lists 个列表中选出 ivfflat_probes 个列表去扫描；值越大越精准但是查询时间更长
ivfflat_probes = 35
ivfflat_probesList = {
    'base_vectors':25,
    'f2_zorder_sorted':25,
    'f3_hilbert_sorted':25,
    'f4_idistance_sorted':25,
    'f5_kmeans_sorted':25,
    'f6_nog_sorted':25,
    'method6_3_vectors':25
}


# HNSW 索引参数
# HNSW 索引构建的参数
# M每个节点最多与多少个其他节点连接，M 越大，图结构越密集，构建时间和空间开销越大，但查询质量更高。
m = 8
# 构建图时的搜索宽度（内部参数），越大图质量越高，但构建越慢
ef_construction = 60
# HNSW 索引查询设置的参数
# 查询时保留的候选节点数量。越大结果越准，但查询越慢
hnsw_ef_search = 200

# 内存参数
work_mem = '4GB'
maintenance_work_mem = '4GB'



