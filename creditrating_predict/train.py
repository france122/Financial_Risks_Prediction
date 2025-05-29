import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from config import *
from dataloader import load_data, create_lstm_samples, normalize_features
from lstm import train_lstm, map_labels, reverse_map
from utils import set_seed, ensure_dir, save_model, log_to_file
import sys


# 设置控制台输出编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')
if not hasattr(sys.stdout, 'reconfigure'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def main():
    # 获取脚本所在目录，确保相对路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(script_dir, INPUT_PATH)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    model_save_path = os.path.join(script_dir, MODEL_SAVE_PATH)
    log_path = os.path.join(script_dir, LOG_PATH)

    is_classification = True  # 三分类任务
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

    unique_years = df[YEAR_COLUMN].unique()
    unique_years = np.sort(unique_years)
    log_to_file(log_path, f"Years in dataset: {unique_years}")
    print(f"数据集中的年份: {unique_years}")

    print("正在创建滑动窗口样本...")
    all_samples_df = create_lstm_samples(df, window_size=WINDOW_SIZE, target_shift=TARGET_SHIFT)

    if len(all_samples_df) == 0:
        log_to_file(log_path, "ERROR: Failed to create valid sliding window samples. Check year continuity.")
        print("错误: 无法创建有效的滑动窗口样本。请检查年份的连续性。")
        return

    log_to_file(log_path, f"Created {len(all_samples_df)} total samples")
    print(f"总共创建了 {len(all_samples_df)} 个样本")

    sample_years = all_samples_df['target_year'].values
    unique_sample_years = np.sort(np.unique(sample_years))
    print(f"样本中的目标预测年份: {unique_sample_years}")
    log_to_file(log_path, f"Target years in samples: {unique_sample_years}")

    test_years = unique_sample_years[-int(len(unique_sample_years) * 0.2):]
    val_years = unique_sample_years[-int(len(unique_sample_years) * 0.4):-int(len(unique_sample_years) * 0.2)]
    train_years = unique_sample_years[:-int(len(unique_sample_years) * 0.4)]

    log_to_file(log_path, f"Train years: {train_years}")
    log_to_file(log_path, f"Validation years: {val_years}")
    log_to_file(log_path, f"Test years: {test_years}")
    print(f"训练集使用年份: {train_years}")
    print(f"验证集使用年份: {val_years}")
    print(f"测试集使用年份: {test_years}")

    train_samples_df = all_samples_df[all_samples_df['target_year'].isin(train_years)]
    val_samples_df = all_samples_df[all_samples_df['target_year'].isin(val_years)]
    test_samples_df = all_samples_df[all_samples_df['target_year'].isin(test_years)]

    log_to_file(log_path, f"Train samples: {len(train_samples_df)}")
    log_to_file(log_path, f"Validation samples: {len(val_samples_df)}")
    log_to_file(log_path, f"Test samples: {len(test_samples_df)}")
    print(f"训练样本数量: {len(train_samples_df)}")
    print(f"验证样本数量: {len(val_samples_df)}")
    print(f"测试样本数量: {len(test_samples_df)}")

    if len(train_samples_df) == 0 or len(val_samples_df) == 0 or len(test_samples_df) == 0:
        log_to_file(log_path, "ERROR: One or more sample sets is empty after division. Check year distribution.")
        print("错误: 数据集划分后，有一个或多个样本集为空。请检查年份分布。")
        return

    X_train = np.stack(train_samples_df['x'].values)
    y_train = np.array(train_samples_df['y'].values)

    X_val = np.stack(val_samples_df['x'].values)
    y_val = np.array(val_samples_df['y'].values)

    X_test = np.stack(test_samples_df['x'].values)
    y_test = np.array(test_samples_df['y'].values)

    print(f"训练数据: {X_train.shape}, {y_train.shape}")
    print(f"验证数据: {X_val.shape}, {y_val.shape}")
    print(f"测试数据: {X_test.shape}, {y_test.shape}")

    log_to_file(log_path, f"Training data: {X_train.shape}, {y_train.shape}")
    log_to_file(log_path, f"Validation data: {X_val.shape}, {y_val.shape}")
    log_to_file(log_path, f"Testing data: {X_test.shape}, {y_test.shape}")

    if USE_SMOTE:
        label_counts = np.bincount((y_train + 1).astype(int))  # 映射到 [0,1,2]
        print("SMOTE前标签分布:")
        for label, count in enumerate(label_counts):
            print(f"  标签 {label - 1} 数量: {count}")
            log_to_file(log_path, f"Label {label - 1} count (before SMOTE): {count}")

        original_shape = X_train.shape
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        smote = SMOTE(random_state=SEED, sampling_strategy='auto')
        X_train_2d_resampled, y_train_resampled = smote.fit_resample(X_train_2d, y_train)

        X_train = X_train_2d_resampled.reshape(-1, original_shape[1], original_shape[2])
        y_train = y_train_resampled

        label_counts_after = np.bincount((y_train + 1).astype(int))
        print("SMOTE后标签分布:")
        for label, count in enumerate(label_counts_after):
            print(f"  标签 {label - 1} 数量: {count}")
            log_to_file(log_path, f"Label {label - 1} count (after SMOTE): {count}")

    print("正在标准化特征...")
    X_train_norm, X_val_norm, mean, std = normalize_features(X_train, X_val)
    X_test_norm = (X_test - mean) / std
    print("特征标准化完成")

    # 将 [-1, 0, 1] 映射到 [0, 1, 2]
    y_train_mapped = map_labels(y_train)
    y_val_mapped = map_labels(y_val)
    y_test_mapped = map_labels(y_test)

    # 训练模型
    print("开始训练LSTM模型...")
    use_enhanced_model = True  # 是否使用增强版LSTM（带注意力机制）
    use_focal_loss = True       # 是否使用Focal Loss

    model, val_preds, train_losses, val_metrics = train_lstm(
        X_train=X_train_norm,
        y_train=y_train_mapped,
        X_val=X_val_norm,
        y_val=y_val_mapped,
        is_classification=True,
        use_focal_loss=use_focal_loss,
        use_enhanced_model=use_enhanced_model
    )
    print("LSTM模型训练完成")

    # 保存模型
    print(f"正在保存模型到 {model_save_path}...")
    save_model(model, model_save_path)
    print("模型保存完成")

    # 还原回原始标签
    val_preds_original = reverse_map(val_preds)
    val_preds_binary = val_preds_original  # 多分类不需要 >0.5 转换

    # 评估验证集
    print("正在评估验证集...")
    val_acc = accuracy_score(y_val, val_preds_original)
    val_f1_weighted = f1_score(y_val, val_preds_original, average='weighted')
    val_f1_macro = f1_score(y_val, val_preds_original, average='macro')

    print(f"验证集 Accuracy: {val_acc:.4f}")
    print(f"验证集 F1 Score (weighted): {val_f1_weighted:.4f}")
    print(f"验证集 F1 Score (macro): {val_f1_macro:.4f}")
    report = classification_report(y_val, val_preds_original)
    print(f"验证集分类报告:\n{report}")
    log_to_file(log_path, f"Validation Accuracy: {val_acc:.4f}")
    log_to_file(log_path, f"Validation F1 Score (weighted): {val_f1_weighted:.4f}")
    log_to_file(log_path, f"Validation F1 Score (macro): {val_f1_macro:.4f}")
    log_to_file(log_path, f"Validation Classification Report:\n{report}")

    # 测试集评估
    print("正在评估测试集...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
        test_logits = model(X_test_tensor)
        test_probs = torch.softmax(test_logits, dim=1)
        test_preds = torch.argmax(test_probs, dim=1).cpu().numpy()

    test_preds_original = reverse_map(test_preds)

    test_acc = accuracy_score(y_test, test_preds_original)
    test_f1_weighted = f1_score(y_test, test_preds_original, average='weighted')
    test_f1_macro = f1_score(y_test, test_preds_original, average='macro')
    cm = confusion_matrix(y_test, test_preds_original)

    print(f"测试集 Accuracy: {test_acc:.4f}")
    print(f"测试集 F1 Score (weighted): {test_f1_weighted:.4f}")
    print(f"测试集 F1 Score (macro): {test_f1_macro:.4f}")
    print(f"混淆矩阵:\n{cm}")
    log_to_file(log_path, f"Test Accuracy: {test_acc:.4f}")
    log_to_file(log_path, f"Test F1 Score (weighted): {test_f1_weighted:.4f}")
    log_to_file(log_path, f"Test F1 Score (macro): {test_f1_macro:.4f}")
    log_to_file(log_path, f"Confusion Matrix:\n{cm}")



    # 详细分类报告
     # 分类报告
    test_report = classification_report(y_test, test_preds_original)
    cm = confusion_matrix(y_test, test_preds_original)
    log_to_file(log_path, f"Test Classification Report:\n{test_report}")
    log_to_file(log_path, f"Confusion Matrix:\n{cm}")
    print(f"测试集分类报告:\n{test_report}")
    print(f"混淆矩阵:\n{cm}")
    
    
    # save_model(model, model_save_path, IS_LSTM)
    # print(f"模型已保存到 {model_save_path}")
    
    print("====== 训练完成 ======")


if __name__ == "__main__":
    main()

#先选出X和Y再进行划分训练集和测试集，交叉验证，分层？（按股票分层，是否打乱年份）
#用验证集选出的超参数重新对在训练集和验证集合并的数据集上做训练
#加上是否ST特征，描述性分析（连续ST）
#做对比