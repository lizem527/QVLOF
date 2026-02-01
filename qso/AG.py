import numpy as np
import math
import torch
import COV
import LGPF

from sklearn.cluster import MiniBatchKMeans, KMeans
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.spatial.distance import cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========== UTILITIES ==========

def sort_within_cluster(data: np.ndarray, idx: np.ndarray, center: np.ndarray):
    d = np.linalg.norm(data[idx] - center, axis=1)
    return idx[np.argsort(d)]

# ========== f1: AG + query-based clustering ==========

def f1_ag_query(data: np.ndarray,
                query: np.ndarray,
                k: int,
                device: str = "cpu") -> np.ndarray:
    """
    AG mapping with query included and then removed from final ordering.
    :param data: N x D
    :param query: M x D
    :param k: number of clusters to use after transform
    :param device: torch device
    :return: permutation indices of data
    """
    # 1) concatenate and AG transform
    all_data = np.vstack([query, data]).astype(np.float32)

    cov, scale, rot = COV.covariance_eigen_decomposition(all_data)

    A = COV.transform_data(cov, scale, rot)
    A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A

    all_torch = torch.from_numpy(all_data).float().to(device)
    transformed = all_torch @ torch.from_numpy(A_np).float().to(device)

    transformed = LGPF.newdata(transformed).cpu().numpy()

    # remove query dimension in the transformed domain
    transformed_data = transformed[query.shape[0]:]

    # 2) KMeans clustering
    km = MiniBatchKMeans(n_clusters=k, batch_size=1024)
    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    # 3) sort clusters by global distance
    center_distances = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_distances)

    perm = []
    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        ordered_idx = sort_within_cluster(transformed_data, idx, centers[cid])
        perm.extend(ordered_idx.tolist())

    return np.array(perm)

# ========== f2: Z-order ==========

def f2_zorder(data: np.ndarray, bits: int = 16) -> np.ndarray:
    """
    True Z-order (Morton code) implementation via bit interleaving.

    Args:
        data: (N, D) float array
        bits: number of bits per dimension

    Returns:
        permutation indices (N,)
    """
    N, D = data.shape

    # 1) normalize to [0, 1]
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    norm = (data - data_min) / (data_max - data_min + 1e-12)

    # 2) quantize to integers
    max_int = (1 << bits) - 1
    ints = (norm * max_int).astype(np.uint32)  # shape (N, D)

    # 3) compute Morton code
    morton = np.zeros(N, dtype=np.uint64)

    for b in range(bits):
        for d in range(D):
            morton |= ((ints[:, d] >> b) & 1) << (b * D + d)

    # 4) sort by Morton code
    return np.argsort(morton)


# ========== f3: Hilbert ==========
def f3_hilbert(data: np.ndarray, p: int = 10) -> np.ndarray:
    """
    Hilbert curve-based 1D ordering.
    p: order of Hilbert curve (2^p per dimension)
    """
    d = data.shape[1]
    hil = HilbertCurve(p, d)

    # map to [0, 2^p - 1]
    data_min = data.min(axis=0)
    data_max = data.max(axis=0) + 1e-9
    normalized = (data - data_min) / (data_max - data_min)
    coords = (normalized * (2**p - 1)).astype(np.int64)  # 推荐用 int64

    dist_vals = []
    for coord in coords:
        dist_vals.append(hil.distance_from_point(coord.tolist()))

    dist_vals = np.array(dist_vals)
    return np.argsort(dist_vals)

# ========== f4: iDistance ==========
from sklearn.cluster import MiniBatchKMeans
def f4_idistance(data: np.ndarray, k: int) -> np.ndarray:
    """
    iDistance original style: choose k cluster centers as reference points,
    then assign each point a 1D key: offset + distance to nearest center.
    """
    # cluster to choose reference points
    km = MiniBatchKMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(data)
    centers = km.cluster_centers_

    # compute offset factors (e.g., gaps between centers)
    # simplest choice: sort centers by norm
    center_norms = np.linalg.norm(centers, axis=1)
    order = np.argsort(center_norms)
    offsets = np.zeros_like(center_norms)
    for i, cid in enumerate(order):
        offsets[cid] = i * (np.ptp(data) + 1e-6)

    # compute 1D iDistance keys
    keyvals = np.zeros(data.shape[0], dtype=float)
    for i in range(data.shape[0]):
        cid = labels[i]
        dist = np.linalg.norm(data[i] - centers[cid])
        keyvals[i] = offsets[cid] + dist

    # sort points by iDistance
    return np.argsort(keyvals)

# ========== f5: KMeans-based sorted points ==========

def f5_kmeans(data: np.ndarray, k: int) -> np.ndarray:
    """
    Simple KMeans ordering: sort by cluster, then by within-cluster distance.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=1024)
    labels = km.fit_predict(data)
    centers = km.cluster_centers_

    center_dists = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_dists)

    perm = []
    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        ordered_idx = sort_within_cluster(data, idx, centers[cid])
        perm.extend(ordered_idx.tolist())

    return np.array(perm)

from LGPF_q2d import newdata

def f6_noA(data: np.ndarray,
                query: np.ndarray,
                k: int,
                device: str = "cpu") -> np.ndarray:
    """
    AG mapping with query included and then removed from final ordering.
    :param data: N x D
    :param query: M x D
    :param k: number of clusters to use after transform
    :param device: torch device
    :return: permutation indices of data
    """
    # 1) concatenate and AG transform
    # all_data = np.vstack([query, data]).astype(np.float32)

    # transformed = LGPF.newdata(all_data).cpu().numpy()

    # # remove query dimension in the transformed domain
    # transformed_data = transformed[query.shape[0]:]
    transformed_data = newdata(
    query=query,
    data=data,
    k=3,
    T=0.3,
    chunk_size=4096
    ).cpu().numpy()

    # 2) KMeans clustering
    km = MiniBatchKMeans(n_clusters=k, batch_size=1024)
    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    # 3) sort clusters by global distance
    center_distances = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_distances)

    perm = []
    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        ordered_idx = sort_within_cluster(transformed_data, idx, centers[cid])
        perm.extend(ordered_idx.tolist())

    return np.array(perm)

def f6_noA2(
    data: np.ndarray,
    query: np.ndarray,
    k: int,
    lgpf_k: int = 10,        # newdata 里的 k
    T: float = 0.3,
    device: str = "cpu"
) -> np.ndarray:
    """
    AG mapping with query included and then removed from final ordering.
    使用 q2d LGPF + KMeans 聚类
    -------------------------------------------------
    :param data:  N x D
    :param query: M x D
    :param k:     number of clusters
    :param device: torch device (保留接口一致性)
    :return: permutation indices of data
    """

    # 1) q2d LGPF transform (only data updated)
    transformed_data = newdata(
        query=query,
        data=data,
        k=10,
        T=0.3,
        chunk_size=4096
    ).cpu().numpy()

    # 2) KMeans clustering (standard version)
    km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    # 3) sort clusters by global distance
    center_distances = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_distances)

    perm = []
    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        ordered_idx = sort_within_cluster(
            transformed_data,
            idx,
            centers[cid]
        )
        perm.extend(ordered_idx.tolist())

    return np.array(perm)
def f1_AG(
    data: np.ndarray,
    query: np.ndarray,
    k: int,
    device: str = "cpu"
) -> np.ndarray:
    """
    AG mapping with query included and then removed from final ordering.
    使用 q2d LGPF + KMeans 聚类
    -------------------------------------------------
    :param data:  N x D
    :param query: M x D
    :param k:     number of clusters
    :param device: torch device (保留接口一致性)
    :return: permutation indices of data
    """

    # 1. 只用 query 计算 cov/eigen 分解得到 A
    cov, scale, rot = COV.covariance_eigen_decomposition(query.astype(np.float32))
    A = COV.transform_data(cov, scale, rot)
    A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A

    # 2. 将 A 应用到 query + data 上
    all_data = np.vstack([query, data]).astype(np.float32)
    all_torch = torch.from_numpy(all_data).float().to(device)
    transformed_all = all_torch @ torch.from_numpy(A_np).float().to(device)

    # 分离变换后的 query 和 data
    M = query.shape[0]
    transformed_query = transformed_all[:M].cpu().numpy()
    transformed_data_np = transformed_all[M:].cpu().numpy()

    # 3. q2d LGPF 变换, 输入变换后的 query 和 data
    transformed_data = newdata(
        query=transformed_query,
        data=transformed_data_np,
        k=k,
        T=0.3,
        chunk_size=4096
    ).cpu().numpy()
    # 1) q2d LGPF transform (only data updated)
    # transformed_data = newdata(
    #     query=query,
    #     data=data,
    #     k=3,
    #     T=0.3,
    #     chunk_size=4096
    # ).cpu().numpy()

    # 2) KMeans clustering (standard version)
    km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    # 3) sort clusters by global distance
    center_distances = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_distances)

    perm = []
    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        ordered_idx = sort_within_cluster(
            transformed_data,
            idx,
            centers[cid]
        )
        perm.extend(ordered_idx.tolist())

    return np.array(perm)
# from k_means_constrained import KMeansConstrained
from k_means import EqualSizeKMeans
# from kmeans1 import EqualSizeKMeans
# from scipy.spatial.distance import cdist
def f6_noA3(
    data: np.ndarray,
    query: np.ndarray,
    k: int,
    lgpf_k: int = 3,        # newdata 里的 k
    T: float = 0.3,
    device: str = "cpu"
) -> np.ndarray:
    """
    AG mapping with query included and then removed from final ordering.
    使用 q2d LGPF + KMeans 聚类
    -------------------------------------------------
    :param data:  N x D
    :param query: M x D
    :param k:     number of clusters
    :param device: torch device (保留接口一致性)
    :return: permutation indices of data
    """

    # 1) q2d LGPF transform (only data updated)
    transformed_data = newdata(
        query=query,
        data=data,
        k=3,
        T=0.3,
        chunk_size=4096
    ).cpu().numpy()

    # 2) KMeans clustering (standard version)
    # km = KMeans(n_clusters=k, n_init=10)
    # labels = km.fit_predict(transformed_data)
    # centers = km.cluster_centers_

    # km = KMeansConstrained(
    #     n_clusters=k,
    #     size_min=1000,
    #     size_max=1000,
    #     n_init=10,
    #     random_state=42  # 建议加上随机种子以保证结果可复现
    # )
    km = EqualSizeKMeans(
        n_clusters=k,
        size=1000,
        n_init=10,
        random_state=42,
        # use_gpu=False,   # 可选
    )
    # labels = km.fit_predict(X)


    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_
    
    # 3) sort clusters by global distance
    center_distances = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_distances)

    perm = []
    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        ordered_idx = sort_within_cluster(
            transformed_data,
            idx,
            centers[cid]
        )
        perm.extend(ordered_idx.tolist())

    return np.array(perm)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def f6_no_A4(
    data: np.ndarray,
    query: np.ndarray,
    k: int,
    lgpf_k: int = 3,
    T: float = 0.3,
    device: str = "cpu"
) -> np.ndarray:
    """
    AG mapping with query included and then removed from final ordering.
    Cosine-distance version (A4)
    -------------------------------------------------
    :param data:   N x D
    :param query:  M x D
    :param k:      number of clusters
    :return: permutation indices of data
    """

    # --------------------------------------------------
    # 1) q2d LGPF transform
    # --------------------------------------------------
    transformed_data = newdata(
        query=query,
        data=data,
        k=lgpf_k,
        T=T,
        chunk_size=4096
    ).cpu().numpy()

    # --------------------------------------------------
    # 2) Equal-size clustering (unchanged)
    # --------------------------------------------------
    # km = EqualSizeKMeans(
    #     n_clusters=k,
    #     size=1000,
    #     n_init=10,
    #     random_state=42,
    # )
    km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    # --------------------------------------------------
    # 3) cluster ordering using cosine distance
    # --------------------------------------------------
    # cosine similarity in [-1, 1]
    center_sim = cosine_similarity(centers, centers)
    center_dist = 1.0 - center_sim

    # aggregate distance of each center to others
    center_scores = np.sum(center_dist, axis=1)
    ordered_clusters = np.argsort(center_scores)

    # --------------------------------------------------
    # 4) sort within each cluster using cosine distance
    # --------------------------------------------------
    perm = []

    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        Xc = transformed_data[idx]
        c = centers[cid].reshape(1, -1)

        # cosine distance to center
        sim = cosine_similarity(Xc, c).reshape(-1)
        dist = 1.0 - sim

        ordered_idx = idx[np.argsort(dist)]
        perm.extend(ordered_idx.tolist())

    return np.array(perm)


def f6_no_A5(
    data: np.ndarray,
    query: np.ndarray,
    k: int,
    lgpf_k: int = 3,
    T: float = 0.3,
    lam: float = 0.3,   # query attraction strength
    device: str = "cpu"
) -> np.ndarray:
    """
    no_A5a: Pure-angle + point-level query-attracted ordering
    """

    # --------------------------------------------------
    # 1) q2d LGPF transform
    # --------------------------------------------------
    transformed_data = newdata(
        query=query,
        data=data,
        k=lgpf_k,
        T=T,
        chunk_size=4096
    ).cpu().numpy()

    # L2 normalize (PURE ANGLE)
    transformed_data /= (
        np.linalg.norm(transformed_data, axis=1, keepdims=True) + 1e-8
    )

    q = query.mean(axis=0, keepdims=True)
    q /= np.linalg.norm(q) + 1e-8

    # --------------------------------------------------
    # 2) Equal-size clustering
    # --------------------------------------------------
    # km = EqualSizeKMeans(
    #     n_clusters=k,
    #     size=1000,
    #     n_init=10,
    #     random_state=42,
    # )
    km = KMeans(n_clusters=k, n_init=10)

    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    centers /= (
        np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    )

    # --------------------------------------------------
    # 3) cluster ordering (pure cosine geometry)
    # --------------------------------------------------
    center_sim = cosine_similarity(centers, centers)
    center_score = np.sum(1.0 - center_sim, axis=1)
    ordered_clusters = np.argsort(center_score)

    # --------------------------------------------------
    # 4) query-attracted ordering within clusters
    # --------------------------------------------------
    perm = []

    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        Xc = transformed_data[idx]
        c = centers[cid].reshape(1, -1)

        # pure angular distance to center
        d_center = 1.0 - cosine_similarity(Xc, c).reshape(-1)

        # query attraction
        sim_q = cosine_similarity(Xc, q).reshape(-1)

        score = d_center - lam * sim_q
        perm.extend(idx[np.argsort(score)].tolist())

    return np.array(perm)
def f6_no_A6(
    data: np.ndarray,
    query: np.ndarray,
    k: int,
    lgpf_k: int = 3,
    T: float = 0.3,
    device: str = "cpu"
) -> np.ndarray:
    """
    no_A5b: Pure-angle + block-level query pull
    """

    # --------------------------------------------------
    # 1) q2d LGPF transform
    # --------------------------------------------------
    #transformed_data = newdata(
    #    query=query,
    #    data=data,
    #    k=lgpf_k,
    #    T=T,
    #    chunk_size=4096
    #).cpu().numpy()
    transformed_data = data

    # L2 normalize
    transformed_data /= (
        np.linalg.norm(transformed_data, axis=1, keepdims=True) + 1e-8
    )

    q = query.mean(axis=0, keepdims=True)
    q /= np.linalg.norm(q) + 1e-8

    # --------------------------------------------------
    # 2) Equal-size clustering
    # --------------------------------------------------
    km = EqualSizeKMeans(
        n_clusters=k,
        size=10000,
        n_init=10,
        random_state=42,
    )
    # km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    centers /= (
        np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    )

    # --------------------------------------------------
    # 3) block-level query pull
    # --------------------------------------------------
    block_scores = np.full(k, np.inf)

    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        Xc = transformed_data[idx]
        sim_q = cosine_similarity(Xc, q).reshape(-1)
        block_scores[cid] = np.min(1.0 - sim_q)

    ordered_clusters = np.argsort(block_scores)

    # --------------------------------------------------
    # 4) intra-block ordering (pure cosine to center)
    # --------------------------------------------------
    perm = []

    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        Xc = transformed_data[idx]
        c = centers[cid].reshape(1, -1)

        dist = 1.0 - cosine_similarity(Xc, c).reshape(-1)
        perm.extend(idx[np.argsort(dist)].tolist())

    return np.array(perm)

def f7_noverti(data: np.ndarray,
                query: np.ndarray,
                k: int,
                device: str = "cpu") -> np.ndarray:
    """
    AG mapping with query included and then removed from final ordering.
    :param data: N x D
    :param query: M x D
    :param k: number of clusters to use after transform
    :param device: torch device
    :return: permutation indices of data
    """
    # 1) concatenate and AG transform
    all_data = np.vstack([query, data]).astype(np.float32)

    cov, scale, rot = COV.covariance_eigen_decomposition(all_data)

    A = COV.transform_data(cov, scale, rot)
    A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A

    all_torch = torch.from_numpy(all_data).float().to(device)
    transformed = all_torch @ torch.from_numpy(A_np).float().to(device)

    transformed = LGPF.newdata(transformed).cpu().numpy()

    # remove query dimension in the transformed domain
    transformed_data = transformed[query.shape[0]:]

    # 2) KMeans clustering
    km = MiniBatchKMeans(n_clusters=k, batch_size=1024)
    labels = km.fit_predict(transformed_data)
    centers = km.cluster_centers_

    # 3) sort clusters by global distance
    center_distances = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_distances)

    perm = []
    for cid in ordered_clusters:
        idx = np.where(labels == cid)[0]
        ordered_idx = sort_within_cluster(transformed_data, idx, centers[cid])
        perm.extend(ordered_idx.tolist())

    return np.array(perm)

def f8_noAchange(data: np.ndarray,
            query: np.ndarray,
            k: int,
            device: str = "cpu") -> np.ndarray:
    """
    AG mapping with query included during clustering,
    then removed from final ordering.
    :param data: N x D
    :param query: M x D
    :param k: number of clusters to use after transform
    :param device: torch device
    :return: permutation indices of data (0 ~ N-1)
    """

    # 1) concatenate and transform
    all_data = np.vstack([query, data]).astype(np.float32)
    transformed = LGPF.newdata(all_data).cpu().numpy()

    M = query.shape[0]   # number of query points
    N = data.shape[0]

    # 2) KMeans clustering (on ALL points)
    km = MiniBatchKMeans(n_clusters=k, batch_size=1024)
    labels_all = km.fit_predict(transformed)
    centers = km.cluster_centers_

    # 3) sort clusters by global distance
    center_distances = np.sum(cdist(centers, centers), axis=1)
    ordered_clusters = np.argsort(center_distances)

    perm = []

    for cid in ordered_clusters:
        # indices in ALL data space
        idx_all = np.where(labels_all == cid)[0]

        # 只保留 data 部分（去掉 query）
        idx_data = idx_all[idx_all >= M] - M
        if len(idx_data) == 0:
            continue

        ordered_idx = sort_within_cluster(
            transformed[M:],      # transformed_data
            idx_data,
            centers[cid]
        )
        perm.extend(ordered_idx.tolist())

    return np.array(perm)

# ========== test example ==========
def main():
    # small test
    np.random.seed(0)
    N, D = 1000000, 128
    data = np.random.randn(N, D).astype(np.float32)
    query = np.random.randn(5000, D).astype(np.float32)

    # set clustering hyperparameters
    k_clusters = 16

    # p1 = f1_ag_query(data, query, k=k_clusters)
    # p2 = f2_zorder(data)
    # p3 = f3_hilbert(data)
    # p4 = f4_idistance(data, k=k_clusters)
    # p5 = f5_kmeans(data, k=k_clusters)
    p6 = f6_noA(data, query, k=k_clusters)
    print("done!")
    # perms = [p1, p2, p3, p4, p5, p6]
    # names = ['f1_AG+Query', 'f2_Z-order', 'f3_Hilbert', 'f4_iDistance', 'f5_KMeans','f6_noG']

    # for name, perm in zip(names, perms):
       
    #     sorted_data = data[perm]

    #     print(f"   前5个点（{name} 排序后）：\n{np.sort(sorted_data.flatten())[:5]}")

if __name__ == "__main__":
    main()
