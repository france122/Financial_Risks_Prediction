import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from embedding import get_finbert_embeddings
# 在文件开头导入RANDOM_SEED
from config import DATA_FILE, EXCLUDE_COLUMNS, FINBERT_EMBEDDING_DIM, RANDOM_SEED
from imblearn.over_sampling import SMOTE

class FinancialDataset(Dataset):
    def __init__(self, X, y):
        """
        X: 输入数据，shape (n_samples, 2, 801)
        y: 目标标签，shape (n_samples,)
        """
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input': self.X[idx],
            'target': self.y[idx]
        }

def create_sequence_pairs(data, targets, years):
    """
    使用滑动窗口创建序列对
    data: shape (n_companies, n_years, n_features)
    targets: shape (n_companies, n_years)
    years: 年份列表
    """
    X_pairs = []
    y_pairs = []
    year_pairs = []
    
    for company_idx in range(data.shape[0]):
        for year_idx in range(2, data.shape[1]):  # 从第3年开始（索引2）
            # 获取前两年的数据
            X_pair = data[company_idx, year_idx-2:year_idx, :]  # shape: (2, 801)
            # 获取当前年的目标
            y_pair = targets[company_idx, year_idx]  # shape: (1,)
            # 获取当前年份
            current_year = years[year_idx]
            
            X_pairs.append(X_pair)
            y_pairs.append(y_pair)
            year_pairs.append(current_year)
    
    return np.array(X_pairs), np.array(y_pairs), np.array(year_pairs)

def prepare_data(verbose=True):
    print("[Info] Starting data preparation...")
    print(f"[Info] Reading data file: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    # 处理年份列
    if df['Accper'].dtype == 'int64' or (df['Accper'].dtype == 'O' and df['Accper'].astype(str).str.len().eq(4).all()):
        df['Year'] = df['Accper'].astype(int)
    else:
        df['Year'] = pd.to_datetime(df['Accper']).dt.year
    
    companies = df['Stkcd'].unique()
    print(f"[Info] Selected {len(companies)} companies for processing")
    
    all_company_data = []
    all_targets = []
    all_texts = []
    all_years = []
    
    for company in companies:
        company_data = df[df['Stkcd'] == company]
        company_data = company_data.sort_values('Year')
        all_texts.extend(company_data['ManaDiscAnal'].tolist())
        all_years.extend(company_data['Year'].tolist())
    
    unique_years = sorted(set(all_years))
    print(f"\n[Info] 数据中的年份分布: {unique_years}")
    print(f"[Info] 最早年份: {min(all_years)}")
    print(f"[Info] 最晚年份: {max(all_years)}")
    
    # 在文件开头添加导入
    from dimensionality_reduction import reduce_dimension
    
    # 在prepare_data函数中，修改文本向量处理部分
    print("[Info] Processing all texts with FinBERT...")
    all_text_embeddings = get_finbert_embeddings(all_texts)
    
    # 添加降维步骤
    all_text_embeddings = reduce_dimension(all_text_embeddings, hidden_dim=100)
    
    text_idx = 0
    all_features = []
    all_labels = []
    
    for company in tqdm(companies, desc="Processing companies"):
        company_data = df[df['Stkcd'] == company]
        company_data = company_data.sort_values('Year')
        num_records = len(company_data)
        financial_columns = [col for col in company_data.columns if col not in EXCLUDE_COLUMNS]
        financial_indicators = company_data[financial_columns].values
        targets = company_data['STPT'].values
        company_text_embeddings = all_text_embeddings[text_idx:text_idx + num_records]
        text_idx += num_records
        combined_features = np.concatenate([company_text_embeddings, financial_indicators], axis=1)
        
        # 将每个公司的特征和标签添加到列表中
        all_features.extend(combined_features)
        all_labels.extend(targets)
    
    # 转换为numpy数组
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # 重新组织数据为三维张量
    num_companies = len(companies)
    samples_per_company = len(all_features) // num_companies
    data_tensor = all_features.reshape(num_companies, samples_per_company, -1)
    targets_tensor = all_labels.reshape(num_companies, samples_per_company)
    
    data_tensor = data_tensor.astype(np.float32)
    targets_tensor = targets_tensor.astype(np.float32)
    print(f"[Info] Final data dimensions: {data_tensor.shape}")
    print("[Info] Creating sequence pairs...")
    X_pairs, y_pairs, year_pairs = create_sequence_pairs(data_tensor, targets_tensor, all_years)
    print(f"\n[Info] 序列对的年份分布: {sorted(set(year_pairs))}")
    
    # 划分数据集
    train_mask = (year_pairs >= 2012) & (year_pairs <= 2018)
    val_mask = (year_pairs >= 2019) & (year_pairs <= 2020)
    test_mask = (year_pairs >= 2021)
    X_train = X_pairs[train_mask]
    y_train = y_pairs[train_mask]
    X_val = X_pairs[val_mask]
    y_val = y_pairs[val_mask]
    X_test = X_pairs[test_mask]
    y_test = y_pairs[test_mask]
    print(f"[Info] Training set size: {len(X_train)}, years: {sorted(set(year_pairs[train_mask]))}")
    print(f"[Info] Validation set size: {len(X_val)}, years: {sorted(set(year_pairs[val_mask]))}")
    print(f"[Info] Test set size: {len(X_test)}, years: {sorted(set(year_pairs[test_mask]))}")
    
    # 只对训练集进行SMOTE过采样
    print("[Info] Before SMOTE, train label distribution:", np.bincount(y_train.astype(int)))
    n_samples, seq_len, feat_dim = X_train.shape
    X_train_2d = X_train.reshape(n_samples, seq_len * feat_dim)
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_2d, y_train)
    X_train_res = X_train_res.reshape(-1, seq_len, feat_dim)
    print("[Info] After SMOTE, train label distribution:", np.bincount(y_train_res.astype(int)))
    
    return X_train_res, y_train_res, X_val, y_val, X_test, y_test
