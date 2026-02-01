import argparse

import faiss
import numpy as np
import json
import time
import random
import csv

from tqdm import tqdm

# 加载真实数据
#sift_base = np.load("D:\\Python_Project\\Learned_Index\\dataset\\sift\\sift_base.npy")

# 参数定义
K = 16
L = 4
n_bins = 256
max_leaf_size = 1000

def load_csv_with_id(path, dtype=np.float32):
    """
    CSV 格式: id, v0, v1, ...
    返回:
      ids: (N,)
      vecs: (N, d)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=dtype)
    ids = data[:, 0].astype(np.int64)
    vecs = data[:, 1:].astype(dtype)
    return ids, vecs


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


def dump_topk_for_queries_to_csv(index, queries, out_csv, top_k=50, epsilon=0.1, preview_dims=8):
    """
    输出 CSV: 每行是 (qid, rank, idx, dist, vec_preview...)
    """
    fieldnames = ["qid", "rank", "idx", "dist"] + [f"v{i}" for i in range(preview_dims)]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for qid in range(len(queries)):
            q_vec = queries[qid]
            hits = index.query(q_vec, epsilon=epsilon, top_k=top_k)  # [(idx, dist), ...]

            for rank, (idx, dist) in enumerate(hits, start=1):
                row = {
                    "qid": qid,
                    "rank": rank,
                    "idx": int(idx),
                    "dist": float(dist),
                }
                vec_preview = index.data[idx][:preview_dims]
                for i in range(preview_dims):
                    row[f"v{i}"] = float(vec_preview[i])

                writer.writerow(row)

    print(f"dumped top{top_k} for {len(queries)} queries -> {out_csv}")


def main_func(data_name, top_k = 100):


    data_path = "/hd1/vec_index/Data/"
    result_path = "/hd1/vec_index/Results/DET-LSH/"
    query_path = "/hd1/vec_index/Query/selected_query.csv"

    base_ids, sift_base = load_csv_with_id(data_path + data_name + ".csv")
    # 构建索引并计时
    start_build = time.time()
    index = DETLSHIndex(sift_base, K, L, n_bins, max_leaf_size)
    index.build()
    build_time_s = time.time() - start_build

    query_ids, queries = load_csv_with_id(query_path)
    exact_index = faiss.IndexFlatL2(sift_base.shape[1])
    exact_index.add(sift_base)
    true_distances, true_knn_indices = exact_index.search(queries, k=top_k)

    # 查询并计算指标
    recalls, overall_ratios = [], []
    pc = time.perf_counter_ns
    total_query_time_ns = 0.0

    qid = 0
    ready_to_write = []
    for query_vec, true_indices in tqdm(zip(queries, true_knn_indices), total=len(queries), desc="Queries",):
    # for query_vec, true_indices in  zip(queries, true_knn_indices):
        # query_vec 本身就是一个向量，不要再拿它当 sift_base 的下标

        start = pc()
        # epsilon默认是0.1, 越大耗时越长, 精度越高
        det_result = index.query(query_vec, epsilon=0.15, top_k=top_k)  # det_result: [(idx, dist), ...]
        total_query_time_ns += pc() - start

        det_indices = [i for i, _ in det_result]

        # recall@k
        true_set = set(true_indices.tolist())
        det_set = set(det_indices)
        recall = len(det_set & true_set) / len(true_set)  # len(true_set) == top_k
        recalls.append(recall)

        # ANN 距离和 / 精确距离和 (精确的是 groundtruth 那 topk 个点的真实距离)
        ann_distance_sum = sum(dist for _, dist in det_result)

        exact_distances = np.linalg.norm(sift_base[true_indices] - query_vec, axis=1)
        exact_distance_sum = float(np.sum(exact_distances))

        # 避免极端情况下除零（例如 query 恰好等于某个 base 向量）
        overall_ratios.append(ann_distance_sum / (exact_distance_sum + 1e-12))

        id_list = ""
        for rank, (idx, dist) in enumerate(det_result, start=1):
            id_list = id_list + str(idx) + " "
        id_list = id_list.rstrip()
        ready_to_write.append([str(qid), id_list])
        qid += 1


    total_query_time_ms = total_query_time_ns / 1_000_000
    # 统计召回率
    print(f"Max Recall: {max(recalls):.4f}")
    print(f"Min Recall: {min(recalls):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")

    # 查询总时间和索引构建时间
    print(f"Total ANN Query Time: {total_query_time_ms:.2f} ms")
    print(f"Total Index Build Time: {build_time_s:.2f} s")
    print(f"QPS: {1000.0 /(total_query_time_ms/int(qid)):.2f}")

    # Overall Ratio 统计
    print(f"Max Overall Ratio: {max(overall_ratios):.4f}")
    print(f"Min Overall Ratio: {min(overall_ratios):.4f}")
    print(f"Average Overall Ratio: {np.mean(overall_ratios):.4f}")

    with open(result_path + data_name + "_summary.csv", "w", encoding="utf-8") as f:
        # 统计召回率
        f.write(f"Max Recall: {max(recalls):.4f}\n")
        f.write(f"Min Recall: {min(recalls):.4f}\n")
        f.write(f"Average Recall: {np.mean(recalls):.4f}\n\n")

        # 查询总时间和索引构建时间
        f.write(f"Total ANN Query Time: {total_query_time_ms:.2f} ms\n")
        f.write(f"Total Index Build Time: {build_time_s:.2f} s\n")
        f.write(f"QPS: {1000.0 / (total_query_time_ms / int(qid)):.2f}\n\n")

        # Overall Ratio 统计
        f.write(f"Max Overall Ratio: {max(overall_ratios):.4f}\n")
        f.write(f"Min Overall Ratio: {min(overall_ratios):.4f}\n")
        f.write(f"Average Overall Ratio: {np.mean(overall_ratios):.4f}\n")

    with open(result_path + data_name + "_ids.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "id_list"])
        w.writerows(ready_to_write)

    # dump_topk_for_queries_to_csv(
    #     index,
    #     queries,
    #     out_csv="results/5w_f7.csv",
    #     top_k=top_k,
    #     epsilon=0.1,
    #     preview_dims=8
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_name", help="dataset name")
    parser.add_argument("top_k", nargs="?", type=int, default=100, help="top k (default: 100)")
    args = parser.parse_args()

    main_func(args.data_name, args.top_k)

