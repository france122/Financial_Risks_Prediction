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
from config import (
    DATA_FILE, EXCLUDE_COLUMNS, FINBERT_EMBEDDING_DIM, 
    RANDOM_SEED, USE_SMOTE, USE_TEXT_EMBEDDING, USE_FINANCIAL_INDICATORS
)
from imblearn.over_sampling import SMOTE

class FinancialDataset(Dataset):
    def __init__(self, X, y, company_ids=None, years_1=None, years_2=None):
        """
        X: 输入数据，shape (n_samples, 2, n_features)
        y: 目标标签，shape (n_samples,)
        company_ids: 公司代码列表
        years_1: 第一年年份列表
        years_2: 第二年年份列表
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # 检查数据维度
        print(f"[Debug] Dataset初始化:")
        print(f"- X shape: {self.X.shape}")
        print(f"- y shape: {self.y.shape}")
        
        # 确保company_ids和years的长度与X一致
        if company_ids is None:
            print("[Warning] No company_ids provided, using empty strings")
            self.company_ids = [''] * len(X)
        else:
            if len(company_ids) != len(X):
                raise ValueError(f"company_ids长度({len(company_ids)})与X长度({len(X)})不一致")
            # 确保company_ids都是字符串格式
            self.company_ids = [str(cid) for cid in company_ids]
            
        if years_1 is None or years_2 is None:
            print("[Warning] No years provided, using zeros")
            self.years_1 = [0] * len(X)
            self.years_2 = [0] * len(X)
        else:
            if len(years_1) != len(X) or len(years_2) != len(X):
                raise ValueError(f"years长度与X长度不一致: years_1={len(years_1)}, years_2={len(years_2)}, X={len(X)}")
            self.years_1 = [int(y) for y in years_1]
            self.years_2 = [int(y) for y in years_2]
        
        print(f"[Debug] 数据集初始化完成:")
        print(f"- 样本数量: {len(self.X)}")
        print(f"- 公司ID数量: {len(self.company_ids)}")
        print(f"- 年份对数量: {len(self.years_1)}")
        if len(self.years_1) > 0:
            print(f"- 年份范围: {min(self.years_1)}-{max(self.years_1)}, {min(self.years_2)}-{max(self.years_2)}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input': self.X[idx],
            'target': self.y[idx],
            'company_id': self.company_ids[idx],
            'year_1': self.years_1[idx],
            'year_2': self.years_2[idx]
        }

def create_sequence_pairs(data, targets, years, companies):
    """
    使用滑动窗口创建序列对
    data: shape (n_companies, n_years, n_features)
    targets: shape (n_companies, n_years)
    years: 年份列表
    companies: 公司代码列表
    
    返回:
    X_pairs: shape (n_samples, 2, n_features) - 每个样本包含两年的特征
    y_pairs: shape (n_samples,) - 第三年的STPT值
    company_ids: 公司代码列表
    years_1: 第一年年份列表
    years_2: 第二年年份列表
    """
    print(f"[Debug] 创建序列对:")
    print(f"- 输入数据形状: {data.shape}")
    print(f"- 目标数据形状: {targets.shape}")
    print(f"- 年份数量: {len(years)}")
    print(f"- 公司数量: {len(companies)}")
    
    X_pairs = []
    y_pairs = []
    company_ids = []
    years_1 = []
    years_2 = []
    
    # 计算每个公司的年份数
    years_per_company = len(years) // len(companies)
    print(f"- 每个公司的年份数: {years_per_company}")
    
    if len(years) != len(companies) * years_per_company:
        raise ValueError(f"年份总数({len(years)})不能被公司数量({len(companies)})整除")
    
    # 重塑年份列表为二维数组 (n_companies, n_years)
    try:
        years = np.array(years).reshape(len(companies), years_per_company)
    except ValueError as e:
        print(f"[Error] 年份数组重塑失败: {e}")
        print(f"- 年份总数: {len(years)}")
        print(f"- 目标形状: ({len(companies)}, {years_per_company})")
        raise
    
    for company_idx in range(data.shape[0]):
        company_code = str(companies[company_idx])  # 确保是字符串格式
        company_years = years[company_idx]
        
        # 从第一年开始，到倒数第三年结束，因为我们需要三年的数据
      
        for year_idx in range(0, data.shape[1]-2):
            # 获取连续两年的特征数据作为输入
            X_pair = data[company_idx, year_idx:year_idx+2, :]  # shape: (2, n_features)
            # 获取第三年的目标值
            y_pair = targets[company_idx, year_idx+2]  # 第三年的STPT
            # 获取两年的年份
            year_1 = int(company_years[year_idx])
            year_2 = int(company_years[year_idx+1])
            
            X_pairs.append(X_pair)
            y_pairs.append(y_pair)
            company_ids.append(company_code)
            years_1.append(year_1)
            years_2.append(year_2)
    
    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)
    
    print(f"[Debug] 序列对创建完成:")
    print(f"- X_pairs shape: {X_pairs.shape}")
    print(f"- y_pairs shape: {y_pairs.shape}")
    print(f"- 公司ID数量: {len(company_ids)}")
    print(f"- 年份对数量: {len(years_1)}, {len(years_2)}")
    
    # 验证所有列表长度一致
    lengths = {
        'X_pairs': len(X_pairs),
        'y_pairs': len(y_pairs),
        'company_ids': len(company_ids),
        'years_1': len(years_1),
        'years_2': len(years_2)
    }
    if len(set(lengths.values())) > 1:
        raise ValueError(f"输出列表长度不一致: {lengths}")
    
    return X_pairs, y_pairs, company_ids, years_1, years_2

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
        # 无论是否使用文本嵌入，都收集年份数据
        all_years.extend(company_data['Year'].tolist())
        if USE_TEXT_EMBEDDING:
            all_texts.extend(company_data['ManaDiscAnal'].tolist())
    
    unique_years = sorted(set(all_years))
    print(f"\n[Info] 数据中的年份分布: {unique_years}")
    print(f"[Info] 最早年份: {min(all_years)}")
    print(f"[Info] 最晚年份: {max(all_years)}")
    
    from dimensionality_reduction import reduce_dimension
    
    # 处理文本向量
    if USE_TEXT_EMBEDDING:
        print("[Info] Processing texts with FinBERT...")
        all_text_embeddings = get_finbert_embeddings(all_texts)
        all_text_embeddings = reduce_dimension(all_text_embeddings, hidden_dim=100)
        print(f"[Info] Text embedding shape: {all_text_embeddings.shape}")
    
    text_idx = 0
    all_features = []
    all_labels = []
    
    for company in tqdm(companies, desc="Processing companies"):
        company_data = df[df['Stkcd'] == company]
        company_data = company_data.sort_values('Year')
        num_records = len(company_data)
        
        features_list = []
        
        # 添加文本特征
        if USE_TEXT_EMBEDDING:
            company_text_embeddings = all_text_embeddings[text_idx:text_idx + num_records]
            features_list.append(company_text_embeddings)
            text_idx += num_records
        
        # 添加财务指标
        if USE_FINANCIAL_INDICATORS:
            financial_columns = [col for col in company_data.columns if col not in EXCLUDE_COLUMNS]
            financial_indicators = company_data[financial_columns].values
            features_list.append(financial_indicators)
        
        # 合并所有特征
        if len(features_list) > 1:
            combined_features = np.concatenate(features_list, axis=1)
        else:
            combined_features = features_list[0]
        
        targets = company_data['Iexsdf_dum'].values
        
        all_features.extend(combined_features)
        all_labels.extend(targets)
    
    # 转换为numpy数组
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    print(f"[Info] Feature dimensions: {all_features.shape[1]}")
    if USE_TEXT_EMBEDDING:
        print(f"[Info] - Text embedding dimensions: 100")
    if USE_FINANCIAL_INDICATORS:
        print(f"[Info] - Financial indicator dimensions: {len(financial_columns)}")
    
    # 重新组织数据为三维张量
    num_companies = len(companies)
    samples_per_company = len(all_features) // num_companies
    data_tensor = all_features.reshape(num_companies, samples_per_company, -1)
    targets_tensor = all_labels.reshape(num_companies, samples_per_company)
    
    data_tensor = data_tensor.astype(np.float32)
    targets_tensor = targets_tensor.astype(np.float32)
    print(f"[Info] Final data dimensions: {data_tensor.shape}")
    
    print("[Info] Creating sequence pairs...")
    X_pairs, y_pairs, company_ids, years_1, years_2 = create_sequence_pairs(data_tensor, targets_tensor, all_years, companies)
    
    # 划分数据集
    train_years = years_2  # 使用第二年作为划分依据
    train_mask = np.array([year >= 2011 and year <= 2017 for year in train_years])
    val_mask = np.array([year >= 2018 and year <= 2019 for year in train_years])
    test_mask = np.array([year >= 2020 for year in train_years])
    
    X_train = X_pairs[train_mask]
    y_train = y_pairs[train_mask]
    company_ids_train = [company_ids[i] for i in range(len(train_mask)) if train_mask[i]]
    years_1_train = [years_1[i] for i in range(len(train_mask)) if train_mask[i]]
    years_2_train = [years_2[i] for i in range(len(train_mask)) if train_mask[i]]
    
    X_val = X_pairs[val_mask]
    y_val = y_pairs[val_mask]
    company_ids_val = [company_ids[i] for i in range(len(val_mask)) if val_mask[i]]
    years_1_val = [years_1[i] for i in range(len(val_mask)) if val_mask[i]]
    years_2_val = [years_2[i] for i in range(len(val_mask)) if val_mask[i]]
    
    X_test = X_pairs[test_mask]
    y_test = y_pairs[test_mask]
    company_ids_test = [company_ids[i] for i in range(len(test_mask)) if test_mask[i]]
    years_1_test = [years_1[i] for i in range(len(test_mask)) if test_mask[i]]
    years_2_test = [years_2[i] for i in range(len(test_mask)) if test_mask[i]]
    
    # 如果启用SMOTE过采样
    if USE_SMOTE:
        print("[Info] Applying SMOTE oversampling...")
        print(f"[Debug] Before SMOTE - Training samples: {len(X_train)}")
        print(f"[Debug] Before SMOTE - Label distribution: {np.bincount(y_train.astype(int))}")
        
        smote = SMOTE(k_neighbors=1,random_state=RANDOM_SEED)
        # 将X_train重塑为2D数组以适应SMOTE
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_train_2d_res, y_train_res = smote.fit_resample(X_train_2d, y_train)
        
        # 将X_train重塑回3D
        X_train = X_train_2d_res.reshape(-1, X_train.shape[1], X_train.shape[2])
        y_train = y_train_res
        
        # 计算需要复制的次数
        n_original = len(company_ids_train)
        n_total = len(y_train)
        
        # 为新样本复制原始样本的metadata
        if n_total > n_original:
            # 计算需要多少额外的样本
            n_extra = n_total - n_original
            # 从原始样本中随机选择索引进行复制
            extra_indices = np.random.choice(n_original, size=n_extra, replace=True)
            
            # 扩展company_ids和years
            company_ids_train.extend([company_ids_train[i] for i in extra_indices])
            years_1_train.extend([years_1_train[i] for i in extra_indices])
            years_2_train.extend([years_2_train[i] for i in extra_indices])
        
        print(f"[Debug] After SMOTE - Training samples: {len(X_train)}")
        print(f"[Debug] After SMOTE - Label distribution: {np.bincount(y_train.astype(int))}")
        print(f"[Debug] After SMOTE - Company IDs: {len(company_ids_train)}")
        print(f"[Debug] After SMOTE - Years pairs: {len(years_1_train)}, {len(years_2_train)}")
        
        # 验证数据长度一致性
        lengths = {
            'X_train': len(X_train),
            'y_train': len(y_train),
            'company_ids': len(company_ids_train),
            'years_1': len(years_1_train),
            'years_2': len(years_2_train)
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f"SMOTE后数据长度不一致: {lengths}")
    
    print(f"[Info] Training set size: {len(X_train)}")
    print(f"[Info] Validation set size: {len(X_val)}")
    print(f"[Info] Test set size: {len(X_test)}")
    
    return (X_train, y_train, company_ids_train, years_1_train, years_2_train,
            X_val, y_val, company_ids_val, years_1_val, years_2_val,
            X_test, y_test, company_ids_test, years_1_test, years_2_test)
