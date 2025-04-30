# train.py

import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, recall_score,f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from model_cusboost import train_cusboost
from config import *
from data_loader import load_data,  create_lstm_samples, normalize_features
from model_lstm import LSTMModel, train_lstm
from utils import set_seed, ensure_dir, save_model, log_to_file
import sys


# 设置控制台输出编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')
# 如果在某些环境中上面的方法不可用，可以使用以下替代方法
if not hasattr(sys.stdout, 'reconfigure'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    # 获取脚本所在目录，确保相对路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 修改为绝对路径
    input_path = os.path.join(script_dir, INPUT_PATH)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    model_save_path = os.path.join(script_dir, MODEL_SAVE_PATH)
    log_path = os.path.join(script_dir, LOG_PATH)
    
    is_classification=True
    # 初始化
    print("====== 开始训练 ======")
    set_seed(SEED)
    ensure_dir(output_dir)
    log_to_file(log_path, "====== Training Started ======")
    print(f"随机种子设置为: {SEED}")
    print(f"输出目录: {output_dir}")
    print(f"输入文件路径: {input_path}")

    # 读取数据
    print("正在加载数据...")
    df = load_data(input_path)
    log_to_file(log_path, f"Loaded data with shape: {df.shape}")
    print(f"数据加载完成，形状: {df.shape}")

    # 获取数据中的年份信息
    unique_years = df[YEAR_COLUMN].unique()
    unique_years = np.sort(unique_years)
    log_to_file(log_path, f"Years in dataset: {unique_years}")
    print(f"数据集中的年份: {unique_years}")
    
    # ====== 修改的部分：先创建滑动窗口，再划分数据集 ======
    print("正在创建滑动窗口样本...")
    # 在整个数据集上创建滑动窗口样本
    all_samples_df = create_lstm_samples(df, window_size=WINDOW_SIZE, target_shift=TARGET_SHIFT)
    
    if len(all_samples_df) == 0:
        log_to_file(log_path, "ERROR: Failed to create valid sliding window samples. Check year continuity.")
        print("错误: 无法创建有效的滑动窗口样本。请检查年份的连续性。")
        return
        
    log_to_file(log_path, f"Created {len(all_samples_df)} total samples")
    print(f"总共创建了 {len(all_samples_df)} 个样本")
    
    # 查看样本的年份分布
    sample_years = all_samples_df['target_year'].values
    unique_sample_years = np.sort(np.unique(sample_years))
    print(f"样本中的目标预测年份: {unique_sample_years}")
    log_to_file(log_path, f"Target years in samples: {unique_sample_years}")
    
    # 根据目标年份划分训练集/验证集/测试集
    # 使用最后的年份作为测试集
    test_years = unique_sample_years[-int(len(unique_sample_years) * 0.2):]  # 最后20%的年份作为测试集
    # 使用倒数第二部分年份作为验证集
    val_years = unique_sample_years[-int(len(unique_sample_years) * 0.4):-int(len(unique_sample_years) * 0.2)]  # 中间20%的年份作为验证集
    # 使用剩余年份作为训练集
    train_years = unique_sample_years[:-int(len(unique_sample_years) * 0.4)]  # 前60%的年份作为训练集
    
    log_to_file(log_path, f"Train years: {train_years}")
    log_to_file(log_path, f"Validation years: {val_years}")
    log_to_file(log_path, f"Test years: {test_years}")
    print(f"训练集使用年份: {train_years}")
    print(f"验证集使用年份: {val_years}")
    print(f"测试集使用年份: {test_years}")
    
    # 基于目标年份划分样本
    train_samples_df = all_samples_df[all_samples_df['target_year'].isin(train_years)]
    val_samples_df = all_samples_df[all_samples_df['target_year'].isin(val_years)]
    test_samples_df = all_samples_df[all_samples_df['target_year'].isin(test_years)]
    
    log_to_file(log_path, f"Train samples: {len(train_samples_df)}")
    log_to_file(log_path, f"Validation samples: {len(val_samples_df)}")
    log_to_file(log_path, f"Test samples: {len(test_samples_df)}")
    print(f"训练样本数量: {len(train_samples_df)}")
    print(f"验证样本数量: {len(val_samples_df)}")
    print(f"测试样本数量: {len(test_samples_df)}")
    
    # 如果划分后的样本集为空，中止训练
    if len(train_samples_df) == 0 or len(val_samples_df) == 0 or len(test_samples_df) == 0:
        log_to_file(log_path, "ERROR: One or more sample sets is empty after division. Check year distribution.")
        print("错误: 数据集划分后，有一个或多个样本集为空。请检查年份分布。")
        return
    
    # 训练数据特征和标签
    print("正在准备特征和标签...")
    X_train = np.stack(train_samples_df['x'].values)
    y_train = np.array(train_samples_df['y'].values)
    
    # 验证数据特征和标签
    X_val = np.stack(val_samples_df['x'].values)
    y_val = np.array(val_samples_df['y'].values)
    
    # 测试数据特征和标签
    X_test = np.stack(test_samples_df['x'].values)
    y_test = np.array(test_samples_df['y'].values)
    
    # 输出训练数据和测试数据的形状
    print(f"训练数据: {X_train.shape}, {y_train.shape}")
    print(f"验证数据: {X_val.shape}, {y_val.shape}")
    print(f"测试数据: {X_test.shape}, {y_test.shape}")

    log_to_file(log_path, f"Training data: {X_train.shape}, {y_train.shape}")
    log_to_file(log_path, f"Validation data: {X_val.shape}, {y_val.shape}")
    log_to_file(log_path, f"Testing data: {X_test.shape}, {y_test.shape}")

    if USE_SMOTE:
        label_counts = np.bincount(y_train.astype(int))
        print("标签分布:")
        for label, count in enumerate(label_counts):
            log_to_file(log_path, f"Label {label} count: {count}")
            print(f"  标签 {label} 数量: {count}")
        
        # 处理不平衡数据 - 使用SMOTE过采样
        log_to_file(log_path, "Applying SMOTE oversampling to balance dataset")
        print("正在应用SMOTE过采样平衡数据集...")
        
        # 重塑数据为2D，因为SMOTE需要2D输入
        original_shape = X_train.shape
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
        # 应用SMOTE
        smote = SMOTE(random_state=SEED)
        X_train_2d_resampled, y_train_resampled = smote.fit_resample(X_train_2d, y_train)
        
        # 重塑回原始形状
        X_train = X_train_2d_resampled.reshape(-1, original_shape[1], original_shape[2])
        y_train = y_train_resampled
        
        # 记录SMOTE后的分布
        label_counts_after = np.bincount(y_train.astype(int))
        print("SMOTE后标签分布:")
        for label, count in enumerate(label_counts_after):
            log_to_file(log_path, f"After SMOTE: Label {label} count: {count}")
            print(f"  标签 {label} 数量: {count}")
    
    # 标准化特征
    print("正在标准化特征...")
    X_train_norm, X_val_norm, mean, std = normalize_features(X_train, X_val)
    X_test_norm = (X_test - mean) / std
    print("特征标准化完成")
    
    # 训练模型
    if IS_LSTM:
        print("开始训练LSTM模型...")
        model, val_preds, train_losses, val_metrics = train_lstm(X_train_norm, y_train, X_val_norm, y_val, is_classification)
        print("LSTM模型训练完成")
    else:
        print("开始训练CUSBoost模型...")
        model, val_preds, train_losses, val_metrics = train_cusboost(X_train_norm, y_train, X_val_norm, y_val, is_classification)
        print("CUSBoost模型训练完成")
    
    # 评估验证集
    print("正在评估验证集...")
    val_preds = val_preds.flatten()
    val_preds_binary = (val_preds > 0.5).astype(int)
    val_metric = accuracy_score(y_val, val_preds_binary)
    metric_name = "Accuracy"
    
    # 计算验证集F1和AUC - 使用加权平均以匹配分类报告
    val_f1 = f1_score(y_val, val_preds_binary, average='weighted')
    val_f1_macro = f1_score(y_val, val_preds_binary, average='macro')
    try:
        val_auc = roc_auc_score(y_val, val_preds)
    except:
        val_auc = 0  # 如果计算AUC出错（例如只有一个类别），则设为0
    
    # 计算验证集Sensitivity(敏感度/召回率)
    val_sensitivity = recall_score(y_val, val_preds_binary, pos_label=1)
    
    print(f"验证集 F1 Score (weighted): {val_f1:.4f}")
    print(f"验证集 F1 Score (macro): {val_f1_macro:.4f}")
    print(f"验证集 AUC: {val_auc:.4f}")
    log_to_file(log_path, f"Validation F1 Score: {val_f1:.4f}")
    log_to_file(log_path, f"Validation AUC: {val_auc:.4f}")
    log_to_file(log_path, f"Validation Sensitivity: {val_sensitivity:.4f}")
    
    # 详细分类报告
    val_report = classification_report(y_val, val_preds_binary)
    log_to_file(log_path, f"Validation Classification Report:\n{val_report}")
    print(f"验证集分类报告:\n{val_report}")
  
    
    log_to_file(log_path, f"Validation {metric_name}: {val_metric:.4f}")
    print(f"验证集 {metric_name}: {val_metric:.4f}")
    
    # 测试模型
    print("正在评估测试集...")
    
    # 根据模型类型选择不同的评估方式
    if IS_LSTM:
        # LSTM模型的评估方式
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
            test_preds = model(X_test_tensor).cpu().numpy().flatten()
    else:
        # CUSBoost模型的评估方式
        X_test_reshaped = X_test_norm.reshape(X_test_norm.shape[0], -1)  # 重塑为2D
        test_preds, _ = model.predict(X_test_reshaped)
        test_preds = test_preds.astype(float) * 0.5 + 0.5  # 转换为[0,1]范围的概率
    
    # 评估测试集
    test_preds_binary = (test_preds > 0.5).astype(int)
    test_metric = accuracy_score(y_test, test_preds_binary)
    
    # 计算F1、AUC和Sensitivity
    test_f1 = f1_score(y_test, test_preds_binary, average='weighted')
    test_f1_macro = f1_score(y_test, test_preds_binary, average='macro')
    try:
        test_auc = roc_auc_score(y_test, test_preds)
    except:
        test_auc = 0  # 如果计算AUC出错（例如只有一个类别），则设为0
    
    # 计算Sensitivity(敏感度/召回率) - 正类的召回率
    test_sensitivity = recall_score(y_test, test_preds_binary, pos_label=1)
    cm=confusion_matrix(y_test,test_preds_binary)
    
    print(f"测试集 F1 Score (weighted): {test_f1:.4f}")
    print(f"测试集 F1 Score (macro): {test_f1_macro:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"测试集 Sensitivity: {test_sensitivity:.4f}")
    print(f"混淆矩阵: {cm}")
    
    log_to_file(log_path, f"Test F1 Score (weighted): {test_f1:.4f}")
    log_to_file(log_path, f"Test F1 Score (macro): {test_f1_macro:.4f}")
    log_to_file(log_path, f"Test AUC: {test_auc:.4f}")
    log_to_file(log_path, f"Test Sensitivity: {test_sensitivity:.4f}")
    log_to_file(log_path, f"Confusion Matrix: {cm}")

    
    # 详细分类报告
    test_report = classification_report(y_test, test_preds_binary)
    log_to_file(log_path, f"Test Classification Report:\n{test_report}")
    print(f"测试集分类报告:\n{test_report}")
    
    log_to_file(log_path, f"Test {metric_name}: {test_metric:.4f}")
    print(f"测试集 {metric_name}: {test_metric:.4f}")
    
    
    # save_model(model, model_save_path, IS_LSTM)
    # print(f"模型已保存到 {model_save_path}")
    
    print("====== 训练完成 ======")


if __name__ == "__main__":
    main()

#先选出X和Y再进行划分训练集和测试集，交叉验证，分层？（按股票分层，是否打乱年份）
#用验证集选出的超参数重新对在训练集和验证集合并的数据集上做训练
#加上是否ST特征，描述性分析（连续ST）
#做对比