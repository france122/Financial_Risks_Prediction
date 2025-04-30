# data_loader.py

import pandas as pd
import numpy as np
from config import *

def load_data(file_path):
    """读取CSV数据文件并进行初步处理"""
    df = pd.read_csv(file_path)
    
    # 确保数值类型正确
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 将标签列转换为数值
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce')
    
    # 按公司和年份排序
    df = df.sort_values([COMPANY_COLUMN, YEAR_COLUMN])
    
    return df

def create_lstm_samples(df, window_size=2, target_shift=1):
    """创建滑动窗口样本用于LSTM训练，确保每个窗口内的年份是连续的"""
    samples = []
    for company, group in df.groupby(COMPANY_COLUMN):
        group = group.sort_values(YEAR_COLUMN).reset_index(drop=True)
        
        # 确保每个公司有足够的记录
        if len(group) < window_size + target_shift:
            continue
        
        for i in range(len(group) - window_size - target_shift + 1):
            # 获取当前窗口的年份
            window_years = group.iloc[i:i + window_size][YEAR_COLUMN].values
            
            # 检查窗口内的年份是否连续
            if len(window_years) != window_size:
                continue
                
            # 检查是否年份连续 (假设年份是整数或可以相减的格式)
            years_diff = np.diff(window_years)
            if not np.all(years_diff == 1):
                continue
                
            # 检查预测目标年份是否与窗口最后一年相差target_shift
            target_year = group.iloc[i + window_size + target_shift - 1][YEAR_COLUMN]
            if target_year - window_years[-1] != target_shift:
                continue
            
            # 特征：window_size年的数据
            x = group.iloc[i:i + window_size][FEATURE_COLUMNS].values
            # 标签：预测target_shift年后的数据
            y = group.iloc[i + window_size + target_shift - 1][LABEL_COLUMN]
            
            samples.append({
                "company": company,
                "year": target_year,
                "feature_years": window_years.tolist(),  # 添加用于特征的年份列表
                "target_year": target_year,              # 添加目标年份
                "x": x,
                "y": y
            })
    
    # 创建DataFrame并添加日志输出
    samples_df = pd.DataFrame(samples)
    if len(samples_df) > 0:
        print(f"创建了 {len(samples_df)} 个有效的滑动窗口样本")
        # 显示前几个样本的年份信息，展示不同的预测窗口
        print("样本预测窗口示例:")
        print(f"  样本示例: 特征年份 [2010, 2011] → 预测年份 2012")
    else:
        print("没有找到符合条件的滑动窗口样本，请检查数据的年份连续性")
    
    return samples_df



def normalize_features(train_data, val_data=None):
    """标准化特征（仅基于训练数据）"""
    mean = np.mean(train_data, axis=(0, 1))
    std = np.std(train_data, axis=(0, 1))
    
    # 避免除以0
    std = np.where(std == 0, 1, std)
    
    train_normalized = (train_data - mean) / std
    
    if val_data is not None:
        val_normalized = (val_data - mean) / std
        return train_normalized, val_normalized, mean, std
    
    return train_normalized, mean, std


