import torch
import math

# =====================================================
# device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================================
# FPS (for query anchor)
# =====================================================
def farthest_point_sampling(x: torch.Tensor, K: int):
    """
    x: [N, D]
    return: [K, D]
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
# Query anchor extractor (logQ)
# =====================================================
def get_query_anchors(
    query: torch.Tensor,
    C: float = 4.0,
    K_min: int = 8,
    K_max: int = 128,
):
    """
    query: [Q, D]
    return: anchors [Kq, D]
    """
    Q = query.shape[0]

    Kq = int(math.ceil(C * math.log(max(Q, 2))))
    Kq = max(K_min, min(Kq, K_max))
    Kq = min(Kq, Q)

    return farthest_point_sampling(query, Kq)


# =====================================================
# q2d newdata (OOM-safe)
# =====================================================
def newdata(
    query,
    data,
    size=None,
    k=3,
    T=0.3,
    chunk_size=4096,
):
    """
    接口风格 ≈ 原 newdata
    ----------------------------------
    query: [Q, D]
    data:  [N, D]
    返回:
        updated_data: [N, D]
    """

    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    x = torch.as_tensor(data, dtype=torch.float32, device=device)

    if q.dim() == 1:
        q = q.unsqueeze(0)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    Q, Dq = q.shape
    N, Dx = x.shape
    assert Dq == Dx, "query 和 data 维度不一致"

    if size is not None and size != Dx:
        raise ValueError(f"指定 size={size} 与输入维度 D={Dx} 不一致")

    # -------------------------------------------------
    # query -> anchor (logQ)
    # -------------------------------------------------
    q_anchor = get_query_anchors(q)

    # -------------------------------------------------
    # dc (query anchor 尺度)
    # -------------------------------------------------
    if q_anchor.shape[0] > 1:
        qd = torch.cdist(q_anchor, q_anchor)
        dc = qd[qd > 0].mean()
    else:
        dc = torch.tensor(1.0, device=device)

    # -------------------------------------------------
    # data 分块处理，避免 OOM
    # -------------------------------------------------
    updated_chunks = []
    eps = 1e-8

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_chunk = x[start:end]                     # [B, D]

        # 距离: [B, Kq]
        dist = torch.cdist(x_chunk, q_anchor)

        k_eff = min(k, q_anchor.shape[0])

        topk_dists, idx = torch.topk(
            dist, k=k_eff, largest=False, dim=1
        )

        q_sel = q_anchor[idx]                      # [B, k, D]

        denom = (topk_dists ** 2).clamp_min(eps)
        delta = q_sel - x_chunk.unsqueeze(1)
        factor = (dc / denom).unsqueeze(-1)

        F_sum = (factor * delta).sum(dim=1)

        b = torch.ones_like(x_chunk) + F_sum
        updated_chunk = x_chunk + T * b

        updated_chunks.append(updated_chunk)

    updated_data = torch.cat(updated_chunks, dim=0)
    return updated_data
