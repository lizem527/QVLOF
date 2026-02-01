import torch
import math

# =====================================================
# device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================================
# Farthest Point Sampling (FPS)
# =====================================================
def farthest_point_sampling(x: torch.Tensor, K: int):
    """
    x: [N, D]
    return:
        anchors: [K, D]
    """
    N, D = x.shape
    K = min(K, N)
    idx = torch.zeros(K, dtype=torch.long, device=x.device)

    idx[0] = torch.randint(0, N, (1,), device=x.device)
    dist = torch.full((N,), float("inf"), device=x.device)

    for i in range(1, K):
        last = x[idx[i - 1]].unsqueeze(0)
        d = torch.cdist(x, last).squeeze(1)
        dist = torch.minimum(dist, d)
        idx[i] = torch.argmax(dist)

    return x[idx]


# =====================================================
# Anchor-based get_dc (logN anchor)
# =====================================================
def get_dc_anchor_torch(
    data: torch.Tensor,
    m: int,
    C: float = 4.0,
    K_min: int = 16,
    K_max: int = 256,
):
    """
    Anchor 版 get_dc_torch
    K_anchor = C * log(N)

    返回:
        topk_pts:   [N, m+1, D]
        dc:         scalar
        topk_dists: [N, m+1]
    """
    assert data.dim() == 2
    N, D = data.shape

    # ---------- logN anchor 数 ----------
    K_anchor = int(math.ceil(C * math.log(max(N, 2))))
    K_anchor = max(K_min, min(K_anchor, K_max))
    K_anchor = min(K_anchor, N)

    # ---------- 选轴点 ----------
    anchors = farthest_point_sampling(data, K_anchor)

    # ---------- 距离 ----------
    dists = torch.cdist(data, anchors)  # [N, K]
    k_eff = min(m + 1, K_anchor)

    sorted_dists, idx = torch.topk(
        dists, k=k_eff, largest=False, dim=1
    )

    topk_pts = anchors[idx]  # [N, m+1, D]

    # ---------- dc ----------
    if k_eff > 1:
        dc = sorted_dists[:, 1].mean()
    else:
        dc = torch.tensor(0.0, device=data.device)

    return topk_pts, dc, sorted_dists


# =====================================================
# newdata (接口保持完全一致)
# =====================================================
def newdata(da, size=None, k=3, T=0.3):
    """
    接口与原版 newdata 完全一致
    """
    x = torch.as_tensor(da, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    N, D = x.shape
    if size is not None and size != D:
        raise ValueError(f"指定 size={size} 与输入维度 D={D} 不一致")

    # ---------- get_dc (logN anchor) ----------
    topk_pts, dc, topk_dists = get_dc_anchor_torch(
        x, m=k
    )

    # ---------- 原 G 力学结构 ----------
    k_plus_1 = topk_pts.shape[1]

    self_pts = topk_pts[:, 0:1, :]
    neighbors = topk_pts[:, 1:k_plus_1, :]

    if k_plus_1 > 1:
        nearest1 = topk_dists[:, 1]
        dist_js = topk_dists[:, 1:k_plus_1]
    else:
        nearest1 = torch.zeros(N, device=device)
        dist_js = torch.zeros(N, 0, device=device)

    eps = 1e-8
    denom = (dist_js ** 2).clamp_min(eps)

    if dist_js.numel() > 0:
        factor = (dc * nearest1.unsqueeze(1)) / denom
        factor = factor.unsqueeze(-1)
        delta = neighbors - self_pts
        F_all = factor * delta
        F_sum = F_all.sum(dim=1)
    else:
        F_sum = torch.zeros_like(x)

    b = torch.ones((N, D), device=device) + F_sum
    updated_self = topk_pts[:, 0, :] + T * b

    return updated_self

# import torch
# import math

# # =====================================================
# # device
# # =====================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


# # =====================================================
# # Farthest Point Sampling (FPS) for anchor seeds
# # =====================================================
# def farthest_point_sampling(x: torch.Tensor, K: int):
#     """
#     x: [N, D]
#     return:
#         seed_idx: [K]
#     """
#     N, D = x.shape
#     K = min(K, N)

#     idx = torch.zeros(K, dtype=torch.long, device=x.device)
#     idx[0] = torch.randint(0, N, (1,), device=x.device)

#     dist = torch.full((N,), float("inf"), device=x.device)

#     for i in range(1, K):
#         last = x[idx[i - 1]].unsqueeze(0)
#         d = torch.cdist(x, last).squeeze(1)
#         dist = torch.minimum(dist, d)
#         idx[i] = torch.argmax(dist)

#     return idx


# # =====================================================
# # Anchor-based get_dc with centroid & mass (log N)
# # =====================================================
# def get_dc_anchor_torch(
#     data: torch.Tensor,
#     m: int,
#     C: float = 4.0,
#     K_min: int = 16,
#     K_max: int = 256,
# ):
#     """
#     Anchor-based get_dc using centroid anchors

#     Returns:
#         topk_pts:   [N, m+1, D]   (centroid anchors)
#         dc:         scalar
#         topk_dists: [N, m+1]
#         topk_mass:  [N, m+1]
#     """
#     assert data.dim() == 2
#     N, D = data.shape

#     # ---------- number of anchors ----------
#     K_anchor = int(math.ceil(C * math.log(max(N, 2))))
#     K_anchor = max(K_min, min(K_anchor, K_max))
#     K_anchor = min(K_anchor, N)

#     # ---------- FPS seeds ----------
#     seed_idx = farthest_point_sampling(data, K_anchor)
#     seed_pts = data[seed_idx]                      # [K, D]

#     # ---------- assign points to nearest seed ----------
#     dists_seed = torch.cdist(data, seed_pts)       # [N, K]
#     assign = dists_seed.argmin(dim=1)               # [N]

#     # ---------- compute centroid & mass ----------
#     anchor_mass = torch.bincount(
#         assign, minlength=K_anchor
#     ).float().to(data.device)                       # [K]

#     anchor_centroid = torch.zeros(
#         (K_anchor, D), device=data.device
#     )
#     anchor_centroid.index_add_(0, assign, data)

#     anchor_centroid = anchor_centroid / (
#         anchor_mass.unsqueeze(1) + 1e-8
#     )                                               # [K, D]

#     # ---------- distances to centroids ----------
#     dists = torch.cdist(data, anchor_centroid)     # [N, K]

#     # ---------- top-k anchors ----------
#     k_eff = min(m + 1, K_anchor)
#     sorted_dists, idx = torch.topk(
#         dists, k=k_eff, largest=False, dim=1
#     )

#     topk_pts = anchor_centroid[idx]                 # [N, m+1, D]
#     topk_mass = anchor_mass[idx]                    # [N, m+1]

#     # ---------- dc ----------
#     if k_eff > 1:
#         dc = sorted_dists[:, 1].mean()
#     else:
#         dc = torch.tensor(0.0, device=data.device)

#     return topk_pts, dc, sorted_dists, topk_mass


# # =====================================================
# # newdata (INTERFACE UNCHANGED)
# # =====================================================
# def newdata(da, size=None, k=3, T=0.3):
#     """
#     Centroid-Mass Anchor LGPF
#     Interface is IDENTICAL to the original version
#     """
#     x = torch.as_tensor(da, dtype=torch.float32, device=device)
#     if x.dim() == 1:
#         x = x.unsqueeze(0)

#     N, D = x.shape
#     if size is not None and size != D:
#         raise ValueError(f"指定 size={size} 与输入维度 D={D} 不一致")

#     # ---------- get_dc ----------
#     topk_pts, dc, topk_dists, topk_mass = get_dc_anchor_torch(
#         x, m=k
#     )

#     k_plus_1 = topk_pts.shape[1]

#     self_pts = topk_pts[:, 0:1, :]            # [N,1,D]
#     neighbors = topk_pts[:, 1:k_plus_1, :]    # [N,k,D]

#     if k_plus_1 > 1:
#         nearest1 = topk_dists[:, 1]            # [N]
#         dist_js = topk_dists[:, 1:k_plus_1]    # [N,k]
#         mass = topk_mass[:, 1:k_plus_1]        # [N,k]
#     else:
#         nearest1 = torch.zeros(N, device=device)
#         dist_js = torch.zeros(N, 0, device=device)
#         mass = torch.zeros(N, 0, device=device)

#     # ---------- mass normalization ----------
#     mass = mass / (mass.mean() + 1e-8)
#     # 也可试：mass = mass.sqrt()

#     eps = 1e-8
#     denom = (dist_js ** 2).clamp_min(eps)

#     if dist_js.numel() > 0:
#         factor = (dc * nearest1.unsqueeze(1)) / denom
#         factor = factor.unsqueeze(-1)
#         mass = mass.unsqueeze(-1)

#         delta = neighbors - self_pts
#         F_all = mass * factor * delta
#         F_sum = F_all.sum(dim=1)
#     else:
#         F_sum = torch.zeros_like(x)

#     b = torch.ones((N, D), device=device) + F_sum
#     updated_self = topk_pts[:, 0, :] + T * b

#     return updated_self



# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 1000000, 128
    x = torch.randn(N, D)

    y = newdata(x, size=D, k=3, T=0.3)
    print("input:", x.shape)
    print("output:", y.shape)
