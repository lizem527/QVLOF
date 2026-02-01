import torch
import numpy as np

# 全局设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def covariance_eigen_decomposition(input_data):

    # 转换NumPy数组为PyTorch张量，并确保在指定设备上
    if isinstance(input_data, np.ndarray):
        # 自动转换数据类型为float32以提升GPU计算效率
        tensor_data = torch.from_numpy(input_data.astype(np.float32)).to(device)
    elif isinstance(input_data, torch.Tensor):
        # 直接使用现有张量，确保设备一致
        tensor_data = input_data.to(device).float()
    else:
        raise TypeError("输入数据类型必须是np.ndarray或torch.Tensor")

    # 维度处理
    if tensor_data.dim() == 1:
        tensor_data = tensor_data.unsqueeze(0)  # [n_features] -> [1, n_features]
    assert tensor_data.dim() == 2, "输入应为二维数据 [样本数, 特征数]"

    X = tensor_data
    n_samples, n_features = X.shape

    # 数据标准化
    X_centered = X - torch.mean(X, dim=0)

    # 协方差矩阵计算
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    # 特征分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # 特征值平方根矩阵
    sqrt_eigen_matrix = torch.diag_embed(torch.sqrt(eigenvalues))

    return cov_matrix, sqrt_eigen_matrix, eigenvectors

def transform_data(cov_mat, scale_mat, rotation_mat):

    # 将NumPy数据转换为PyTorch张量并送至相同设备
    # data_tensor = cov_mat.to(device)

    # 确保所有矩阵在同一设备
    sqrt_mat = scale_mat.to(device)
    eigvecs = rotation_mat.to(device)

    # 中心化处理 (与协方差计算时一致)
    # mean = torch.mean(data_tensor, dim=0)
    # print(mean)
    # centered_data = data_tensor - mean

    # 执行变换: scale_mat @ rotation_mat
    transformed = torch.matmul(cov_mat, eigvecs)
    transformed = torch.matmul(transformed, sqrt_mat)

    return transformed.cpu().numpy()
# def transform_data(cov_mat, scale_mat, rotation_mat):

#     # 将NumPy数据转换为PyTorch张量并送至相同设备
#     data_tensor = cov_mat.to(device)

#     # 确保所有矩阵在同一设备
#     sqrt_mat = scale_mat.to(device)
#     eigvecs = rotation_mat.to(device)

#     # 中心化处理 (与协方差计算时一致)
#     mean = torch.mean(data_tensor, dim=0)
#     # print(mean)
#     centered_data = data_tensor - mean

#     # 执行变换: cov_mat @ scale_mat @ rotation_mat
#     transformed = torch.matmul(centered_data, eigvecs)
#     transformed = torch.matmul(transformed, sqrt_mat)

#     return transformed.cpu().numpy()