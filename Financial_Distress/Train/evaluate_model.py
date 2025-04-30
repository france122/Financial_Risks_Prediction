# evaluate_model.py

import pandas as pd
import numpy as np
import torch
import os
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from data_loader import load_data, create_lstm_samples, normalize_features
from model_lstm import LSTMModel
from utils import ensure_dir
import sys

# 设置控制台输出编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')
# 如果在某些环境中上面的方法不可用，可以使用以下替代方法
if not hasattr(sys.stdout, 'reconfigure'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def load_model(model_path, input_size, is_classification=True):
    """加载保存的模型"""
    output_activation = 'sigmoid' if is_classification else None
    model = LSTMModel(
        input_size=input_size,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_activation=output_activation
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def evaluate_classification_metrics(y_true, y_pred_proba):
    """计算分类评估指标"""
    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    # 计算最佳阈值（可选）
    # J = tpr - fpr
    # idx = np.argmax(J)
    # best_threshold = thresholds[idx]
    
    # 使用默认阈值0.5计算其他指标
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算F1分数
    f1 = f1_score(y_true, y_pred)
    
    # 计算召回率/灵敏度（正类的召回率）
    sensitivity = recall_score(y_true, y_pred)
    
    # 计算特异度（负类的召回率）
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # 计算精确率
    precision = precision_score(y_true, y_pred)
    
    # 计算准确率
    accuracy = (y_true == y_pred).mean()
    
    # 输出分类报告
    report = classification_report(y_true, y_pred)
    
    return {
        'auc': auc_score,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'report': report
    }

def visualize_roc_curve(fpr, tpr, auc_score):
    """绘制ROC曲线"""
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 初始化
    ensure_dir(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, "lstm_model_final.pt")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 加载数据
    df = load_data(INPUT_PATH)
    print(f"加载数据: {df.shape}")
    
    # 按年份划分数据集
    unique_years = df[YEAR_COLUMN].unique()
    unique_years = np.sort(unique_years)
    
    # 使用与训练脚本相同的数据划分逻辑
    test_years = unique_years[-WINDOW_SIZE-TARGET_SHIFT:]
    test_df = df[df[YEAR_COLUMN].isin(test_years)]
    
    # 创建测试数据窗口
    test_samples_df = create_lstm_samples(test_df, window_size=WINDOW_SIZE, target_shift=TARGET_SHIFT)
    
    if len(test_samples_df) == 0:
        print("错误: 未能创建有效的测试窗口样本。")
        return
    
    # 提取特征和标签
    X_test = np.stack(test_samples_df['x'].values)
    y_test = np.array(test_samples_df['y'].values)
    
    # 判断是分类还是回归任务
    unique_labels = np.unique(y_test)
    is_classification = len(unique_labels) <= 5
    
    if not is_classification:
        print("注意: 检测到回归任务，但此脚本主要评估分类指标。")
    
    # 加载归一化参数（假设已经在训练时保存）
    # 这里简化处理，直接使用测试数据进行标准化
    # 实际应用中应该使用训练集计算得到的均值和标准差
    mean = np.mean(X_test, axis=(0, 1))
    std = np.std(X_test, axis=(0, 1))
    std = np.where(std == 0, 1, std)
    
    X_test_norm = (X_test - mean) / std
    
    # 加载模型
    model, device = load_model(model_path, input_size=X_test.shape[2], is_classification=is_classification)
    
    # 进行预测
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
        test_preds = model(X_test_tensor).cpu().numpy().flatten()
    
    # 评估分类指标
    metrics = evaluate_classification_metrics(y_test, test_preds)
    
    # 输出结果
    print("\n===== 模型评估指标 =====")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n分类报告:")
    print(metrics['report'])
    
    # 可视化ROC曲线
    visualize_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'])
    print(f"ROC曲线已保存至 {os.path.join(OUTPUT_DIR, 'roc_curve.png')}")
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'company': test_samples_df['company'],
        'feature_years': test_samples_df['feature_years'],
        'target_year': test_samples_df['target_year'],
        'true': y_test,
        'pred_proba': test_preds,
        'pred_binary': (test_preds > 0.5).astype(int)
    })
    
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"详细预测结果已保存至 {results_path}")

if __name__ == "__main__":
    main() 