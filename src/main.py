# -*- coding: utf-8 -*-
"""
Project C: 宏观经济预测模型
主运行程序 - 一键执行：数据生成 → 清洗 → 建模 → 评估 → 可视化
适合留学申请：可复现、无外部依赖、结果自动输出
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入自定义模块
from data_process import load_raw_data, clean_macro_data, save_processed_data
from model import prepare_model_data, train_forecast_model, evaluate_model, save_model
from visualize import plot_trend, plot_forecast_compare, plot_feature_importance

def generate_simulation_data():
    """
    自动生成模拟宏观经济数据（无外部数据依赖，本地直接运行）
    包含：日期、GDP增速、CPI、M2、汇率、失业率
    """
    print("\n===== 生成模拟宏观经济数据 =====")
    
    # 时间范围：5年月度数据
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=365*5)
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    
    np.random.seed(42)
    
    data = {
        "date": dates,
        "gdp": 5.0 + np.random.normal(0, 0.8, len(dates)).cumsum() / 10,
        "cpi": 100 + np.random.normal(0, 0.5, len(dates)).cumsum(),
        "m2": 8.0 + np.random.normal(0, 0.3, len(dates)).cumsum() / 10,
        "exchange_rate": 7.0 + np.random.normal(0, 0.1, len(dates)).cumsum(),
        "unemployment": 5.0 + np.random.normal(0, 0.2, len(dates))
    }
    
    df = pd.DataFrame(data)
    
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    
    df.to_csv("data/raw/macro_data.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 模拟数据生成完成，共 {len(df)} 条月度数据")
    return df

def save_final_report(metrics):
    """
    保存最终预测报告
    """
    if not os.path.exists("results"):
        os.makedirs("results")
    
    with open("results/forecast_report.txt", "w", encoding="utf-8") as f:
        f.write("===== 宏观经济预测模型报告 =====\n")
        f.write(f"报告生成时间：{pd.Timestamp.now()}\n")
        f.write(f"模型：随机森林回归\n")
        f.write(f"预测目标：CPI\n")
        f.write(f"MAE：{metrics['MAE']:.4f}\n")
        f.write(f"R²：{metrics['R2']:.4f}\n")
        f.write("===============================\n")
    
    print("✅ 预测报告已保存到 results/forecast_report.txt")

# ==================== 主流程 ====================
if __name__ == "__main__":
    print("🚀 启动宏观经济预测全流程")
    
    # 1. 生成模拟数据
    generate_simulation_data()
    
    # 2. 数据清洗
    df = load_raw_data()
    cleaned_data = clean_macro_data(df)
    save_processed_data(cleaned_data)
    
    # 3. 建模训练
    X_train, X_test, y_train, y_test, features = prepare_model_data()
    model = train_forecast_model(X_train, y_train, model_type="random_forest")
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
    
    # 4. 可视化
    plot_trend()
    y_pred = model.predict(X_test)
    plot_forecast_compare(y_test, y_pred)
    plot_feature_importance(model, features)
    
    # 5. 输出报告
    save_final_report(metrics)
    
    print("\n🎉 全部流程执行完成！")
    print("📁 结果查看：")
    print("  - 图表：results/figures/")
    print("  - 报告：results/forecast_report.txt")
    print("  - 模型：models/")
    print("  - 数据：data/processed/")