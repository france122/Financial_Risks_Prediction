import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data_loader import prepare_data, FinancialDataset
from torch.utils.data import ConcatDataset
import numpy as np
from config import (
    RANDOM_SEED,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    LSTM_HIDDEN_SIZE
)
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score

class FinancialModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=LSTM_HIDDEN_SIZE):  # 使用配置的LSTM_HIDDEN_SIZE
        super(FinancialModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)

def train_model():
    print("====== 开始训练 ======")
    print(f"随机种子设置为: {RANDOM_SEED}")
    
    print("[Info] Starting data preparation...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    
    # 打印数据集信息
    print(f"训练数据: {X_train.shape}, {y_train.shape}")
    print(f"验证数据: {X_val.shape}, {y_val.shape}")
    print(f"测试数据: {X_test.shape}, {y_test.shape}")
    print(f"训练样本数量: {len(X_train)}")
    print(f"验证样本数量: {len(X_val)}")
    print(f"测试样本数量: {len(X_test)}")
    
    train_dataset = FinancialDataset(X_train, y_train)
    val_dataset = FinancialDataset(X_val, y_val)
    test_dataset = FinancialDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 使用配置的BATCH_SIZE
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 创建combined_dataset (验证集 + 测试集)
    combined_dataset = ConcatDataset([val_dataset, test_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[2]
    model = FinancialModel(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 使用配置的LEARNING_RATE
    # num_epochs = 10  # 删除这行，使用配置的NUM_EPOCHS
    
    # 初始化最佳验证损失和对应的epoch
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\n开始训练模型...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch['input']
            targets = batch['target']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        
        # 验证
        val_loss = evaluate_model(model, val_loader, criterion, print_metrics=False)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"\n[Info] 最优模型出现在第 {best_epoch} 个epoch，验证集loss: {best_val_loss:.4f}")
    
    # 加载最佳模型进行最终评估
    print("\n正在评估最终模型...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    print("\n验证集评估结果：")
    evaluate_model(model, val_loader, criterion, desc="验证集最终评估")
    
    print("\n测试集评估结果：")
    evaluate_model(model, combined_loader, criterion, desc="测试集最终评估")
    
    print("\n====== 训练完成 ======")

def evaluate_model(model, data_loader, criterion, desc="模型评估结果", print_metrics=True):
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input']
            targets = batch['target']
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    if print_metrics and len(np.unique(all_targets)) > 1:
        from sklearn.metrics import classification_report
        print(f"\n{desc}：")
        print("\n分类报告:")
        print(classification_report(all_targets, all_predictions))
        
        auc = roc_auc_score(all_targets, all_probs)
        print(f"\nAUC Score: {auc:.4f}")
        
        cm = confusion_matrix(all_targets, all_predictions)
        print("\n混淆矩阵:")
        print(cm)
    
    return total_loss / len(data_loader)

if __name__ == "__main__":
    train_model()
