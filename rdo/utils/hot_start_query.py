import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_large_dataset(input_df):
    """
    生成800万条数据的函数

    参数:
    input_df (pd.DataFrame): 包含7列的原始DataFrame

    返回:
    pd.DataFrame: 包含800万条生成数据的新DataFrame
    """
    # 检查输入DataFrame是否符合要求
    #if len(input_df.columns) != 7:
    #    raise ValueError("输入DataFrame必须包含7列")

    # 计算各列的均值用于填充
    col_means = input_df.mean()

    # 初始化结果DataFrame
    result_df = pd.DataFrame(columns=input_df.columns)

    # 获取列名列表
    columns = input_df.columns.tolist()

    # 生成40组数据，每组20万条
    for group in tqdm(range(1), desc="生成进度"):
        # 随机选择i和j列
        while True:
            i, j = np.random.choice(len(columns), 2, replace=False)
            if abs(i - j) > 1:  # 确保i和j的差大于1
                break

        # 准备当前组的数据容器
        group_data = []

        # 生成20万条数据
        for _ in range(100000):
            # 创建新行
            new_row = {}

            # 对i和j列使用原始数据范围内的随机值
            new_row[columns[i]] = np.random.choice(input_df[columns[i]])
            new_row[columns[j]] = np.random.choice(input_df[columns[j]])

            # 其他列使用均值填充
            for col_idx, col in enumerate(columns):
                if col_idx not in [i, j]:
                    new_row[col] = col_means[col]

            group_data.append(new_row)

        # 将当前组数据添加到结果中
        group_df = pd.DataFrame(group_data)
        result_df = pd.concat([result_df, group_df], ignore_index=True)

    return result_df

def generate_high_dataset(input_df):
    repeats = 200
    random_state = 42

    # 设置随机种子
    np.random.seed(random_state)

    # 计算原始数据量
    original_size = len(input_df)
    print(f"原始数据量: {original_size}")

    # 计算组数
    n_groups = original_size // (5 * repeats)
    print(f"组数: {n_groups}")

    # 计算需要生成的新数据量
    new_data_size = n_groups * repeats
    print(f"需要生成的新数据量: {new_data_size}")

    # 存储生成的新数据
    new_data_list = []

    # 对每一组进行操作
    for group in range(n_groups):
        # 从原始数据中随机选择一个样本
        random_sample = input_df.sample(n=1, random_state=random_state + group)
        # 将该样本重复指定次数
        repeated_sample = pd.concat([random_sample] * repeats, ignore_index=True)
        # 添加到新数据列表
        new_data_list.append(repeated_sample)
        #print(f"第 {group + 1} 组完成: 从原始数据中选择了1个样本，重复了 {repeats} 次")

    # 合并所有新生成的数据
    new_data = pd.concat(new_data_list, ignore_index=True)

    print(f"生成的新数据量: {len(new_data)}")

    # 将新数据与原始数据拼接
    combined_data = pd.concat([input_df, new_data], ignore_index=True)
    #print(f"最终拼接后数据量: {len(combined_data)}")

    return combined_data