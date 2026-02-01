import torch

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_dc_torch(data: torch.Tensor, m: int):
    """
    Torch 版本的 get_dc：
    - data: [N, D] 张量
    - m: 取 m+1 个最近邻（包含自身）
    返回:
    - topk_pts: [N, m+1, D] 最近的 m+1 个点（按距离从小到大）
    - dc: 全体样本的平均最近邻距离 (scalar)
    - topk_dists: [N, m+1] 对应的距离值
    """
    assert data.dim() == 2, "data must be [N, D]"
    N, D = data.shape
    # 保证 k <= N-1（除自身外最多有 N-1 个别点）
    k = min(m, max(0, N - 1))
    k_plus_1 = k + 1

    # 计算成对距离矩阵：[N, N]
    # torch.cdist 返回平方根后的欧氏距离
    dists = torch.cdist(data, data)  # [N, N]

    # 取每行最小的 k+1 个（包含自身距离 0）
    # torch.topk 的 largest=False 取最小
    values_topk, indices_topk = torch.topk(dists, k=k_plus_1, largest=False, dim=1)

    # 根据 indices 选出对应的点向量 -> [N, k+1, D]
    # indices_topk: [N, k+1], 使用 gather 需要扩展索引维度
    idx_expanded = indices_topk.unsqueeze(-1).expand(-1, -1, D)  # [N, k+1, D]
    topk_pts = torch.gather(data.unsqueeze(1).expand(-1, k_plus_1, -1), 0,
                            idx_expanded)  # 先尝试 gather，但为了保证正确性，使用 take_along_dim 下面替代

    # 上面 gather 方式在 shape 上不方便，改用 take_along_dim：
    topk_pts = torch.take_along_dim(data.unsqueeze(1).expand(-1, k_plus_1, -1), idx_expanded, dim=0)
    # NOTE: 上面两种 gather/take_along_dim 在某些 torch 版本上行为差异，下面使用更兼容的替代实现：

    # 更稳妥的实现：直接索引（构造 batch 索引）
    batch_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_plus_1)  # [N, k+1]
    topk_pts = data[indices_topk]  # PyTorch 支持这样的高级索引 -> [N, k+1, D]

    # topk_dists: [N, k+1]
    topk_dists = values_topk

    # 计算 dc = 全体样本的平均最近邻距离（跳过自身，取第二小元素 values_topk[:,1]）
    # 注意：若 k==0（没有邻居），则设 dc 为 0（或 eps）
    eps = 1e-8
    if k >= 1:
        nearest_vals = values_topk[:, 1]  # 最近邻（非自身）
        dc = nearest_vals.mean()
    else:
        nearest_vals = torch.zeros(N, device=device)
        dc = torch.tensor(0.0, device=device)

    return topk_pts.to(device), dc.to(device), topk_dists.to(device)


def newdata(da, size=None, k=3, T=0.3):
    """
    Torch 版本的 newdata（基于原 second 代码逻辑，但完全用 torch 向量化实现）：
    - da: 可为 numpy array / list / torch.Tensor，期望形状 [N, D] 或 [D] 单条
    - size: 如果给出，检查或用于兼容；如果 None 则按输入维度确定
    - k: 原代码中为 3（取 top k 最近邻）
    - T: 缩放系数
    返回:
    - data: [N, D] 更新后的点（torch.Tensor，位于 device）
    """
    # 转为 tensor 并放到 device
    x = torch.as_tensor(da, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1, D]
    N, D = x.shape
    if size is not None and size != D:
        # 如果用户指定 size，与输入维度不一致则抛错
        raise ValueError(f"指定 size={size} 与输入维度 D={D} 不一致")
    # 使用 get_dc_torch 获取 topk 点、dc、距离
    # 注意：get_dc_torch 的参数 m 等于原代码的 k（它内部取 m 或 N-1）
    topk_pts, dc, topk_dists = get_dc_torch(x, m=k)

    # topk_pts: [N, k+1, D], topk_dists: [N, k+1]
    # 原代码中：初始化 b = ones(N, D)，然后对 j=1..k 做 b[i] += F_j
    # 其中 F_j = dc * distance[i][1] * (topk[i][j+1]-topk[i][0]) / distance[i][j+1]^2
    # 向量化实现如下：
    k_plus_1 = topk_pts.shape[1]
    k_effective = k_plus_1 - 1  # 实际有效邻居数（若数据量小则可能 < k）

    # self_pts: [N, 1, D], neighbors: [N, k, D]
    self_pts = topk_pts[:, 0:1, :]             # [N,1,D]
    neighbors = topk_pts[:, 1:k_plus_1, :]     # [N,k,D]
    # distances:
    nearest1 = topk_dists[:, 1]                # [N] 最近邻（非自身）
    dist_js = topk_dists[:, 1:k_plus_1]        # [N,k]

    # 避免 0 距离导致除零
    eps = 1e-8
    denom = (dist_js ** 2).clamp_min(eps)      # [N,k]

    # 计算 F_j for all j: shape [N,k,D]
    # factor per pair: dc * nearest1 / denom  -> [N,k]
    factor = (dc * nearest1.unsqueeze(1)) / denom  # [N,k]
    factor = factor.unsqueeze(-1)  # -> [N,k,1]

    delta = (neighbors - self_pts)  # [N,k,D]
    F_all = factor * delta          # [N,k,D]

    # 初始化 b as ones（保持原算法偏置）
    b = torch.ones((N, D), device=device, dtype=x.dtype)

    # 累加所有邻居的 F
    b = b + F_all.sum(dim=1)  # [N, D]

    # 更新 topk[:,0]（参照点）: topk[:,0] = topk[:,0] + T * b
    updated_self = topk_pts[:, 0, :] + T * b  # [N, D]

    # 返回最终数据：将每个样本的 topk[i,0]（更新后的自点）作为输出
    data_out = updated_self

    return data_out


# -------------------------
# 简单示例 / 测试
# -------------------------
if __name__ == "__main__":
    # 随机测试
    N = 6
    D = 4
    torch.manual_seed(0)
    sample = torch.randn(N, D)

    out = newdata(sample, size=D, k=3, T=0.3)
    print("input:\n", sample)
    print("output:\n", out)
