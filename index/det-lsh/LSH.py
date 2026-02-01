import faiss
import numpy as np
import json
import time
import random

# 加载真实数据
#sift_base = np.load("D:\\Python_Project\\Learned_Index\\dataset\\sift\\sift_base.npy")

# 参数定义
K = 16
L = 4
n_bins = 256
max_leaf_size = 1000

# LSH 投影函数
class LSHProjector:
    def __init__(self, K=16, L=4, d=128):
        np.random.seed(42)
        self.K, self.L = K, L
        self.projections = [np.random.randn(d, K) for _ in range(L)]

    def project(self, vec):
        return [vec @ proj for proj in self.projections]

# 动态编码器
class DynamicEncoder:
    def __init__(self, n_bins=256):
        self.breakpoints = None
        self.n_bins = n_bins

    def fit(self, values):
        sampled = np.random.choice(values, min(100000, len(values)), replace=False)
        self.breakpoints = np.quantile(sampled, np.linspace(0, 1, self.n_bins + 1))

    def encode(self, val):
        return np.searchsorted(self.breakpoints, val) - 1

class DETreeNode:
    def __init__(self, depth=0, max_leaf_size=1000):
        self.children = {}
        self.points = []
        self.depth = depth
        self.max_leaf_size = max_leaf_size
        self.split_dim = None

    def insert(self, code, idx):
        node = self
        while True:
            if len(node.points) < node.max_leaf_size or node.depth >= len(code):
                node.points.append(idx)
                return
            if node.split_dim is None:
                node.split_dim = node.depth % len(code)
            split_val = code[node.split_dim]
            if split_val not in node.children:
                node.children[split_val] = DETreeNode(node.depth + 1, node.max_leaf_size)
            node = node.children[split_val]

class DETLSHIndex:
    def __init__(self, data, K=16, L=4, n_bins=256, max_leaf_size=1000):
        self.projector = LSHProjector(K, L, d=data.shape[1])
        self.encoders = [[DynamicEncoder(n_bins) for _ in range(K)] for _ in range(L)]
        self.trees = [DETreeNode(max_leaf_size=max_leaf_size) for _ in range(L)]
        self.data = data

    def build(self):
        projected_data = [self.projector.project(vec) for vec in self.data]

        for l in range(L):
            for k in range(K):
                dim_values = np.array([proj[l][k] for proj in projected_data])
                self.encoders[l][k].fit(dim_values)

        for idx, projs in enumerate(projected_data):
            for l in range(L):
                code = tuple(self.encoders[l][k].encode(projs[l][k]) for k in range(K))
                self.trees[l].insert(code, idx)
            if idx % 100000 == 0:
                print(f"Inserted {idx}/{len(self.data)}")

    def query(self, q_vec, epsilon=0.1, top_k=20):
        q_projs = self.projector.project(q_vec)
        candidates = set()

        for l in range(L):
            q_code = tuple(self.encoders[l][k].encode(q_projs[l][k]) for k in range(K))
            stack = [self.trees[l]]
            while stack:
                node = stack.pop()
                if node.children:
                    split_dim = node.split_dim
                    query_val = q_code[split_dim]
                    for child_key, child_node in node.children.items():
                        if abs(child_key - query_val) <= epsilon * n_bins:
                            stack.append(child_node)
                else:
                    candidates.update(node.points)

        distances = [(idx, np.linalg.norm(self.data[idx] - q_vec)) for idx in candidates]
        return sorted(distances, key=lambda x: x[1])[:top_k]

#数据
num_queries= 100
data_path = '/home/west/vec_test/LSH/vector.csv'
sift_base = np.loadtxt(data_path, delimiter=',', dtype=np.float32, skiprows=1)
# 构建索引并计时
start_build = time.time()
index = DETLSHIndex(sift_base, K, L, n_bins, max_leaf_size)
index.build()
build_time_s = time.time() - start_build
'''
# 加载真实查询数据和20NN答案
with open('D:\Python_Project\Learned_Index\dataset\sift\SIFT_groundtruth.json', 'r') as f:
    groundtruth = json.load(f)
sample_indices = groundtruth['sampled_point_indices']
true_knn_indices = groundtruth['knn_indices']
'''
#sample_indices = random.sample(range(len(sift_base)), num_queries)
#queries=sift_base[sample_indices]
query_path='/home/west/vec_test/LSH/queries.csv'
queries = np.loadtxt(query_path, delimiter=',', dtype=np.float32, skiprows=1)
exact_index = faiss.IndexFlatL2(sift_base.shape[1])
exact_index.add(sift_base)
true_distances, true_knn_indices = exact_index.search(queries, k=20)

# 查询并计算指标
recalls, overall_ratios = [], []
total_query_time_ms = 0.0

top_k = 20

for query_vec, true_indices in zip(queries, true_knn_indices):
    # query_vec 本身就是一个向量，不要再拿它当 sift_base 的下标

    start_query = time.time()
    det_result = index.query(query_vec, top_k=top_k)  # det_result: [(idx, dist), ...]
    total_query_time_ms += (time.time() - start_query) * 1000

    det_indices = [i for i, _ in det_result]

    # recall@k
    true_set = set(true_indices.tolist())
    det_set = set(det_indices)
    recall = len(det_set & true_set) / len(true_set)  # len(true_set) == top_k
    recalls.append(recall)

    # ANN 距离和 / 精确距离和（按你的写法：精确的是 groundtruth 那 20 个点的真实距离）
    ann_distance_sum = sum(dist for _, dist in det_result)

    exact_distances = np.linalg.norm(sift_base[true_indices] - query_vec, axis=1)
    exact_distance_sum = float(np.sum(exact_distances))

    # 避免极端情况下除零（例如 query 恰好等于某个 base 向量）
    overall_ratios.append(ann_distance_sum / (exact_distance_sum + 1e-12))

# 统计召回率
print(f"Max Recall: {max(recalls):.4f}")
print(f"Min Recall: {min(recalls):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")

# 查询总时间和索引构建时间
print(f"Total ANN Query Time: {total_query_time_ms:.2f} ms")
print(f"Total Index Build Time: {build_time_s:.2f} s")

# Overall Ratio 统计
print(f"Max Overall Ratio: {max(overall_ratios):.4f}")
print(f"Min Overall Ratio: {min(overall_ratios):.4f}")
print(f"Average Overall Ratio: {np.mean(overall_ratios):.4f}")



def print_top3_for_query(index, queries, qid=0, epsilon=0.1):
    q_vec = queries[qid]  # 这是 query 向量本身（不是下标）
    top3 = index.query(q_vec, epsilon=epsilon, top_k=3)

    print(f"\n=== Query #{qid} ===")
    print("query_vec shape:", q_vec.shape)

    for rank, (idx, dist) in enumerate(top3, start=1):
        # idx 就是 self.data 的行号位置
        print(f"\nTop{rank}:")
        print("  self.data row idx =", idx)
        print("  distance =", float(dist))
        print("  vector preview (first 8 dims) =", index.data[idx][:8])

# 示例：打印第 0 个 query 的 top3
print_top3_for_query(index, queries, qid=0, epsilon=0.1)
