# -*- coding: utf-8 -*-
"""
宏观经济预测可视化模块
生成时间序列趋势图、预测对比图、特征重要性图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def create_output_dir():
    """创建图表输出文件夹"""
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")

def plot_trend(data_path="data/processed/cleaned_data.csv"):
    """
    绘制宏观指标时间趋势图
    """
    create_output_dir()
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["cpi"], label="CPI", color="#1f77b4", linewidth=2)
    plt.plot(df["date"], df["gdp"], label="GDP增速", color="#ff7f0e", linewidth=2)
    plt.title("宏观经济指标时间趋势", fontsize=14)
    plt.xlabel("时间")
    plt.ylabel("指数/增速")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/macro_trend.png", dpi=300)
    plt.close()
    print("✅ 时间趋势图已保存")

def plot_forecast_compare(y_test, y_pred, model_name="随机森林"):
    """
    绘制真实值 vs 预测值对比图
    """
    create_output_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="真实值", marker="o", markersize=3)
    plt.plot(y_pred, label="预测值", marker="s", markersize=3)
    plt.title(f"{model_name} 预测效果对比", fontsize=14)
    plt.xlabel("样本序号")
    plt.ylabel("CPI指数")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/forecast_compare.png", dpi=300)
    plt.close()
    print("✅ 预测对比图已保存")

def plot_feature_importance(model, features):
    """
    绘制特征重要性图
    """
    create_output_dir()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        df_imp = pd.DataFrame({"feature": features, "importance": imp})
        df_imp = df_imp.sort_values("importance", ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(df_imp["feature"], df_imp["importance"], color="#2ca02c")
        plt.title("特征重要性", fontsize=14)
        plt.tight_layout()
        plt.savefig("results/figures/feature_importance.png", dpi=300)
        plt.close()
        print("✅ 特征重要性图已保存")