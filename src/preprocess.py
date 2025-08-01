import pandas as pd
import numpy as np
import ast


def load_data(file_path):
    """
    加载并解析 CSV 数据文件。

    Args:
        file_path (str): 数据文件路径。

    Returns:
        tuple: 输入特征 (X) 和输出标签 (y)。
    """
    # 加载 CSV 文件
    df = pd.read_csv(file_path)

    # 解析输入特征列
    df['Input Features'] = df['Input Features'].apply(parse_input_vector)

    # 解析并扁平化标签
    df['Flattened Labels'] = df.apply(
        lambda row: flatten_labels(row, ["Action Fields", "Liability Fields", "Extension Fields"]),
        axis=1
    )

    # 准备输入特征和目标标签
    X = np.vstack(df['Input Features'].values)
    y = np.array(df['Flattened Labels'].tolist())

    return X, y


def parse_input_vector(input_vector_str):
    """
    解析输入特征字符串为 numpy 数组。

    Args:
        input_vector_str (str): 字符串形式的输入特征。

    Returns:
        np.array: 数值形式的输入特征。
    """
    return np.array(ast.literal_eval(input_vector_str))


def flatten_labels(row, label_groups):
    """
    将分组的标签字段解压为扁平化的二进制向量。

    Args:
        row (pd.Series): 数据行。
        label_groups (list): 标签字段的分组列表。

    Returns:
        list: 扁平化的二进制标签列表。
    """
    flattened = []
    for group in label_groups:
        values = ast.literal_eval(row[group])  # 转换为列表
        flattened.extend(values)
    return flattened
