# -*- coding: utf-8 -*-
"""
宏观经济预测模型模块
支持：线性回归 / 随机森林 预测GDP、CPI等指标
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

def prepare_model_data(data_path="data/processed/cleaned_data.csv"):
    """
    准备模型训练数据
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 时间特征（宏观预测核心特征）
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    
    # 目标变量：CPI（可替换为GDP、M2等）
    target = "cpi"
    features = [col for col in df.columns if col not in ["date", target]]
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"✅ 训练集：{X_train.shape}，测试集：{X_test.shape}")
    return X_train, X_test, y_train, y_test, features

def train_forecast_model(X_train, y_train, model_type="random_forest"):
    """
    训练宏观预测模型
    """
    if model_type == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    print(f"✅ 模型训练完成：{model_type}")
    return model

def evaluate_model(model, X_test, y_test):
    """
    模型评估
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n===== 模型评估结果 =====")
    print(f"平均绝对误差 (MAE)：{mae:.4f}")
    print(f"决定系数 (R²)：{r2:.4f}")
    print("========================\n")
    
    return {"MAE": mae, "R2": r2}

def save_model(model, model_path="models/macro_forecast_model.pkl"):
    """
    保存模型到本地
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存：{model_path}")