import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm

from offline import COV
from offline.COV import *
from offline.LGPF import *
# from cuml.cluster import HDBSCAN
import os
import random

class DataProcessor:
    """数据处理器类，用于处理向量数据，包括协方差分解、聚类和保存结果"""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
    
    def save_vectors_to_csv(self, vectors, ids, savename, output_path="results_enhanced", labels=None):
        """将向量和对应的 ID 保存到 CSV 文件，可选保存类别序号"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(vectors.shape[1])])
        df['id'] = ids
        if labels is not None:
            df['cluster_label'] = labels
        df.to_csv(os.path.join(output_path, savename), index=False)
        print(f"已将向量保存到 {os.path.join(output_path, savename)}")

    def minibatch_kmeans_clustering(self, data, n_clusters=10, batch_size=10000, max_iterations=100, random_seed=42):
        """
        GPU-accelerated MiniBatchKMeans clustering using PyTorch with fixed random seed.
        """
        print("Starting MiniBatchKMeans clustering...")
        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        data = np.array(data, dtype=np.float32)
        data_torch = torch.from_numpy(data).to(self.device)
        n_samples, n_features = data_torch.shape

        # 随机初始化聚类中心
        indices = torch.randperm(n_samples)[:n_clusters]
        centers = data_torch[indices].clone()

        print(f"GPU memory allocated before clustering: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved before clustering: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # MiniBatchKMeans 迭代
        for iteration in tqdm(range(max_iterations), desc="Running MiniBatchKMeans"):
            batch_indices = torch.randperm(n_samples)[:batch_size]
            batch_data = data_torch[batch_indices]
            distances = torch.cdist(batch_data, centers)
            batch_labels = torch.argmin(distances, dim=1)

            # 更新聚类中心
            for k in range(n_clusters):
                mask = batch_labels == k
                if mask.sum() > 0:
                    centers[k] = batch_data[mask].mean(dim=0)

        # 为所有数据分配最终标签
        with torch.no_grad():
            distances = torch.cdist(data_torch, centers)
            labels = torch.argmin(distances, dim=1).cpu().numpy()

        centers = centers.cpu().numpy()
        print(f"GPU memory allocated after clustering: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved after clustering: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()

        print(f"Number of clusters: {n_clusters}")
        return labels, centers
        
    # def hdbscan_clustering(self, data, min_cluster_size=100, min_samples=10):
    #     """执行 HDBSCAN 聚类，基于 GPU 加速的 cuML 实现"""
    #     print("开始 HDBSCAN 聚类...")
    #     data = np.array(data, dtype=np.float32)
        
    #     if data.shape[0] == 0:
    #         raise ValueError("输入数据为空，无法执行聚类")
        
    #     clusterer = HDBSCAN(
    #         min_cluster_size=min_cluster_size,
    #         min_samples=min_samples,
    #         cluster_selection_method='eom',
    #         allow_single_cluster=True,
    #         verbose=True
    #     )
        
    #     if self.device.type == "cuda":
    #         print(f"聚类前 GPU 内存分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    #         print(f"聚类前 GPU 内存保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    #     with tqdm(total=1, desc="运行 HDBSCAN 聚类") as pbar:
    #         labels = clusterer.fit_predict(data)
    #         pbar.update(1)
        
    #     unique_labels = np.unique(labels[labels != -1])
    #     centers = np.array([data[labels == label].mean(axis=0) for label in unique_labels])
        
    #     if self.device.type == "cuda":
    #         print(f"聚类后 GPU 内存分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    #         print(f"聚类后 GPU 内存保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    #         torch.cuda.empty_cache()
        
    #     print(f"发现的聚类数量: {len(unique_labels)}")
    #     return labels, centers

    def method3(self, base, query):
        """纵拼方法：联合基础向量和查询向量进行表示，A 处理（无G操作）"""
        # 自动生成 base_ids
        base_ids = np.arange(base.shape[0])
        
        # 1. 计算协方差矩阵分解
        all_data = np.vstack((base, query))
        all_data_torch = torch.from_numpy(all_data).float().to(self.device)
        
        cov_mat, scale_mat, rotation_mat = COV.covariance_eigen_decomposition(all_data)
        cov_mat = torch.from_numpy(cov_mat).float().to(self.device) if isinstance(cov_mat, np.ndarray) else cov_mat.to(self.device)
        scale_mat = torch.from_numpy(scale_mat).float().to(self.device) if isinstance(scale_mat, np.ndarray) else scale_mat.to(self.device)
        rotation_mat = torch.from_numpy(rotation_mat).float().to(self.device) if isinstance(rotation_mat, np.ndarray) else rotation_mat.to(self.device)
        
        # 2. 计算矩阵 A
        A = COV.transform_data(cov_mat, scale_mat, rotation_mat)
        A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A
        
        # 检查维度
        if A_np.shape[1] != all_data.shape[1]:
            raise ValueError(f"维度不匹配: A 有 {A_np.shape[1]} 维，预期 {all_data.shape[1]} 维")
        
        # 3. 第一步：矩阵 A 转换
        result = all_data_torch @ torch.from_numpy(A_np).float().to(self.device)
        
        # 4. 第二步：G 操作（已去除）
        # result = LGPF.newdata(result)
        
        print(f"Method 3 结果形状: {result.shape}")
        
        # 5. 执行向量聚类
        labels, centers = self.minibatch_kmeans_clustering(result.cpu().numpy())
        
        # 6. 分离查询部分
        enhanced_base = result[:len(base), :]
        print(f"Method 3 分离后的结果形状: {enhanced_base.shape}")
        
        # 7. 计算 key 值（不排序）
        enhanced_base_np = enhanced_base.cpu().numpy()
        centers_torch = torch.from_numpy(centers).float().to(self.device)
        keys = []
        for i in tqdm(range(len(enhanced_base_np)), desc="计算 Method 3 的 key 值"):
            vec_torch = torch.from_numpy(enhanced_base_np[i]).float().to(self.device)
            cluster_id = labels[i]
            if cluster_id != -1:
                dist_to_center = torch.norm(vec_torch - centers_torch[cluster_id]).item()
                sum_dist_to_all_centers = torch.sum(torch.norm(centers_torch[cluster_id] - centers_torch, dim=1)).item()
                key = dist_to_center + sum_dist_to_all_centers
            else:
                key = float('inf')
            keys.append(key)
            del vec_torch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        del enhanced_base_np, centers_torch
        
        # 返回原顺序的 keys 作为 np.array
        return np.array(keys)

def main():
    """主函数：读取 CSV 数据并应用不同方法进行处理"""
    file_path = "./Q&Data/Dominick.csv"
    
    data = pd.read_csv(file_path, nrows=1000)
    
    print("数据集中的列名:")
    print(list(data.columns))
    
    feature_columns = ['move', 'store', 'price', 'profit']
    base = data[feature_columns].values
    
    if base.shape[0] == 0:
        raise ValueError("预处理后基础数据为空")
    
    query = base.copy()
    
    print(f"基础数据形状: {base.shape}")
    print(base)
    
    processor = DataProcessor()
    
    print("\n运行 Method 3 (纵拼)...")
    keys = processor.method3(base, query)
    print(f"Method 3 返回的 keys: {keys}")

if __name__ == "__main__":
    main()