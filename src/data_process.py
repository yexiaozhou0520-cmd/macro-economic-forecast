# -*- coding: utf-8 -*-
"""
宏观经济数据预处理模块
Project C: 宏观经济预测模型
Central University of Finance and Economics
"""

import pandas as pd
import numpy as np
import os

def load_raw_data(file_path="data/raw/macro_data.csv"):
    """
    加载原始宏观经济数据
    """
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    print(f"✅ 原始数据加载完成，数据形状：{df.shape}")
    return df

def clean_macro_data(df):
    """
    数据清洗：缺失值、异常值、时间格式统一
    """
    # 复制数据避免修改原始文件
    data = df.copy()
    
    # 转换时间列
    data["date"] = pd.to_datetime(data["date"])
    
    # 缺失值填充（线性插值，宏观数据标准方法）
    data = data.interpolate(method="linear")
    
    # 删除空行
    data = data.dropna()
    
    print(f"✅ 数据清洗完成，清洗后形状：{data.shape}")
    return data

def save_processed_data(data, output_path="data/processed/cleaned_data.csv"):
    """
    保存清洗后的数据
    """
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 处理后数据已保存到：{output_path}")

if __name__ == "__main__":
    # 测试运行
    if not os.path.exists("data/raw/macro_data.csv"):
        print("⚠️  未找到原始数据，将生成模拟宏观经济数据")
    else:
        df = load_raw_data()
        cleaned = clean_macro_data(df)
        save_processed_data(cleaned)