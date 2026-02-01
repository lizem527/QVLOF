import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from AG import (
    f1_ag_query,
    f2_zorder,
    f3_hilbert,
    f4_idistance,
    f5_kmeans,
    f6_noA,
    f6_noA2,
    f6_noA3,
    f6_no_A4,
    f6_no_A5,
    f6_no_A6
    # f1_AG
    # f8_noAchange
)
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()  # Clear GPU memory at start
    
# ========= 基本工具函数 =========

def get_dataset_dir(dataset_root_dir, dataset):
    dataset_dir = dataset_root_dir
    for part in dataset.split('-'):
        dataset_dir = os.path.join(dataset_dir, part)
    return dataset_dir

def read_vecs(fname):
    data = np.fromfile(fname, dtype='int32')
    dim = data[0]
    vectors = data.reshape(-1, dim + 1)[:, 1:].copy()
    if fname.endswith(".fvecs"):
        vectors = vectors.view('float32')
    return vectors

def read_ivecs(fname):
    data = np.fromfile(fname, dtype='int32')
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:].copy()

def save_vectors_to_csv(vectors, ids, filename, is_selected=None, out_dir="result"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    if is_selected is None:
        data = np.hstack([ids.reshape(-1, 1), vectors])
        columns = ["id"] + [f"v{i}" for i in range(vectors.shape[1])]
    else:
        data = np.hstack([ids.reshape(-1, 1), vectors, is_selected.reshape(-1, 1)])
        columns = (
            ["id"]
            + [f"v{i}" for i in range(vectors.shape[1])]
            + ["is_selected"]
        )

    pd.DataFrame(data, columns=columns).to_csv(path, index=False)
    print(f"Saved: {path}")

def read_vectors_from_csv(path):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].to_numpy()
    vectors = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return ids, vectors
# ========= Query 选择（KMeans）=========

def select_query_kmeans(query, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=3407)
    kmeans.fit(query)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    max_cluster = -1
    max_avg_dist = -1

    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        dists = np.linalg.norm(query[idx] - centers[cid], axis=1)
        avg_dist = dists.mean()
        if avg_dist > max_avg_dist:
            max_avg_dist = avg_dist
            max_cluster = cid

    selected_indices = np.where(labels == max_cluster)[0].astype(np.int32)
    selected_query = query[selected_indices]

    return selected_query, selected_indices

# ========= 主流程 =========

if __name__ == "__main__":
    DATASET_ROOT = "../dataset"  # ← 改这里
    DATASET_NAME = "sift-1m"

    dataset_dir = get_dataset_dir(DATASET_ROOT, DATASET_NAME)
    sift_dir = os.path.join(dataset_dir, "sift")

    # ---- 读取 base / query / groundtruth ----
    vectors_base = read_vecs(os.path.join("/hd1/workspace/datasets/gist/gist", "gist_base.fvecs"))
    vectors_base = read_vecs(os.path.join(sift_dir, "sift_base.fvecs"))
    vectors_query = read_vecs(os.path.join(sift_dir, "sift_query.fvecs"))
    vectors_gt = read_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))

    # BASE_CSV = "./result/base_vectors.csv"
    # SELECTED_QUERY_CSV = "./selected_query.csv"

    # base_ids, vectors_base = read_vectors_from_csv(BASE_CSV)
    # query_ids, selected_query = read_vectors_from_csv(SELECTED_QUERY_CSV)

    print("Base shape :", vectors_base.shape)
    # print("Selected query shape:", selected_query.shape)

    base_ids = np.arange(len(vectors_base))
    #query_ids = np.arange(len(vectors_query))

    #print("Base shape :", vectors_base.shape)
    #print("Query shape:", vectors_query.shape)
    # print("GT shape   :", vectors_gt.shape)

    # ---- 选择一部分 query（非 random）----
    selected_query, selected_indices = select_query_kmeans(
        vectors_query,
    )

    print("Selected query shape:", selected_query.shape)

    # # ---- 保存 base / query / selected_query ----
    #save_vectors_to_csv(
    #     vectors_base,
    #     base_ids,
    #     "base_vectors.csv"
    # )

    #is_selected = np.zeros(len(vectors_query), dtype=np.int32)
    #is_selected[selected_indices] = 1

    # save_vectors_to_csv(
    #     vectors_query,
    #     query_ids,
    #     "query_vectors.csv",
    #     is_selected=is_selected
    # )

    SEED = 120  # 或你喜欢的任何数字
    rng = np.random.default_rng(SEED)

        # ========= 排序参数 =========
    cluster = 100
    out_dir = "../result"
    os.makedirs(out_dir, exist_ok=True)

    data = vectors_base
    query = selected_query
    print("begin!")
    # ========= 六种方法 =========
    methods = {
        #"random_perm": lambda: rng.permutation(len(data)),
        # "f1_ag2_query": lambda: f1_ag_query(data, query, k=cluster),
        #"f2_zorder":   lambda: f2_zorder(data, bits=64),
        #"f3_hilbert":  lambda: f3_hilbert(data, p=12),
        #"f4_idistance":lambda: f4_idistance(data, k=cluster),
        # "f5_kmeans":   lambda: f5_kmeans(data, k=cluster),
        # "f6_noA":      lambda: f6_noA(data, query, k=cluster),
        # "f8_noAchangelessverti": lambda : f8_noAchange(data, query, k=cluster),
        # "f1_AG": lambda: f6_noA2(data, query, k=cluster),
        # "f1_AG2": lambda: f6_no_A4(data, query, k=cluster),
        # "f1_AG3": lambda: f6_no_A5(data, query, k=cluster),
        "f1_AG_block10000": lambda: f6_no_A6(data, query, k=cluster),
        # "f1_AG": lambda: f1_AG(data, query, k=cluster),
    }
    # lgpf_k_list = [1, 2, 3, 4,5]
    # T_list = [0.5, 1.0]
    # # ========= 消融实验 =========
    # for lgpf_k in lgpf_k_list:
    #     for T in T_list:
    #         name = f"f1_AG_lgpfk{lgpf_k}_T{T}"

    #         print(f"\nRunning {name} ...")

    #         perm = f6_noA2(
    #             data=data,
    #             query=query,
    #             k=cluster,
    #             lgpf_k=lgpf_k,
    #             T=T
    #         )

    #         perm = np.asarray(perm).astype(np.int32)

    #         sorted_vectors = data[perm]
    #         sorted_ids = perm

    #         save_vectors_to_csv(
    #             sorted_vectors,
    #             sorted_ids,
    #             filename=f"{name}_sorted.csv",
    #             out_dir=out_dir
    #         )

    # print("All ablation experiments done.")


    # ========= 逐个计算、保存 =========
    for name, func in methods.items():
        print(f"\nRunning {name} ...")

        perm = func()                     # 排序后的 index
        perm = np.asarray(perm).astype(np.int32)

        sorted_vectors = data[perm]
        sorted_ids = perm

        save_vectors_to_csv(
            sorted_vectors,
            sorted_ids,
            filename=f"{name}_sorted.csv",
            out_dir=out_dir
        )

    print("All methods done.")
