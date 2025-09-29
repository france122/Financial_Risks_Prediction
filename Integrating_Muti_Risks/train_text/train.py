import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data_loader import prepare_data, FinancialDataset
import numpy as np
import pandas as pd
from config import (
    RANDOM_SEED,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    LSTM_HIDDEN_SIZE
)
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score

class FinancialModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=LSTM_HIDDEN_SIZE):
        super(FinancialModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 返回最后一个时间步的hidden状态
        return lstm_out[:, -1, :]

def collect_embeddings():
    print("====== 开始收集LSTM特征表示 ======")
    print(f"随机种子设置为: {RANDOM_SEED}")
    
    print("[Info] Starting data preparation...")
    # 假设prepare_data现在返回公司ID和年份信息
    X_train, y_train, company_ids_train, years_1_train, years_2_train, \
    X_val, y_val, company_ids_val, years_1_val, years_2_val, \
    X_test, y_test, company_ids_test, years_1_test, years_2_test = prepare_data()
    
    print(f"训练数据: {X_train.shape}")
    print(f"验证数据: {X_val.shape}")
    print(f"测试数据: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = FinancialDataset(X_train, y_train, company_ids_train, years_1_train, years_2_train)
    val_dataset = FinancialDataset(X_val, y_val, company_ids_val, years_1_val, years_2_val)
    test_dataset = FinancialDataset(X_test, y_test, company_ids_test, years_1_test, years_2_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 注意：这里设置shuffle=False
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[2]
    model = FinancialModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # 使用MSE损失，因为我们在学习特征表示
    
    print("\n开始训练模型...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch['input']
            targets = batch['target']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1).repeat(1, LSTM_HIDDEN_SIZE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss/len(train_loader):.4f}")
    
    print("\n开始收集特征表示...")
    
    def collect_from_loader(loader, phase):
        model.eval()
        all_embeddings = []
        all_companies = []
        all_years_1 = []
        all_years_2 = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                print(f"\n[Debug] Processing batch {batch_idx}")
                inputs = batch['input']
                companies = batch['company_id']
                years_1 = batch['year_1']
                years_2 = batch['year_2']
                targets = batch['target']
                
                print(f"[Debug] Batch sizes:")
                print(f"- inputs: {inputs.shape}")
                print(f"- companies: {len(companies)}")
                print(f"- targets: {targets.shape}")
                print(f"- years_1: {len(years_1)}")
                print(f"- years_2: {len(years_2)}")
                
                embeddings = model(inputs)
                
                # 处理每个样本
                for i in range(len(inputs)):
                    all_embeddings.append(embeddings[i].cpu().numpy())
                    # 确保company_id是字符串格式
                    company_id = companies[i]
                    if isinstance(company_id, torch.Tensor):
                        company_id = company_id.item()
                    all_companies.append(str(company_id))
                    all_years_1.append(int(years_1[i]))
                    all_years_2.append(int(years_2[i]))
                    all_targets.append(targets[i].cpu().numpy().item())
        
        # 检查所有列表的长度是否一致
        lengths = {
            'embeddings': len(all_embeddings),
            'companies': len(all_companies),
            'years_1': len(all_years_1),
            'years_2': len(all_years_2),
            'targets': len(all_targets)
        }
        print(f"\n[Debug] 各列表长度: {lengths}")
        print(f"[Debug] 年份示例:")
        for i in range(min(5, len(all_years_1))):
            print(f"  - 样本{i}: year_1={all_years_1[i]}, year_2={all_years_2[i]}")
        
        if len(set(lengths.values())) > 1:
            raise ValueError(f"列表长度不一致: {lengths}")
        
        # 创建DataFrame
        data_dict = {
            'Stkcd': all_companies,
            'target': all_targets,
            'year_1': all_years_1,
            'year_2': all_years_2
        }
        
        # 添加所有embedding列到字典中
        embeddings_array = np.array(all_embeddings)
        for i in range(LSTM_HIDDEN_SIZE):
            data_dict[f'embedding_{i}'] = embeddings_array[:, i]
        
        # 一次性创建DataFrame
        embeddings_df = pd.DataFrame(data_dict)
        
        # 保存到CSV
        embeddings_df.to_csv(f'lstm_embeddings_{phase}.csv', index=False)
        print(f"[Info] {phase}集特征形状: {embeddings_df.shape}")
        return embeddings_df
    
    print("收集训练集特征...")
    train_df = collect_from_loader(train_loader, 'train')
    print("收集验证集特征...")
    val_df = collect_from_loader(val_loader, 'val')
    print("收集测试集特征...")
    test_df = collect_from_loader(test_loader, 'test')
    
    print("\n====== 特征收集完成 ======")
    print(f"训练集特征形状: {train_df.shape}")
    print(f"验证集特征形状: {val_df.shape}")
    print(f"测试集特征形状: {test_df.shape}")
    print("\n特征已保存为CSV文件：")
    print("- lstm_embeddings_train.csv")
    print("- lstm_embeddings_val.csv")
    print("- lstm_embeddings_test.csv")

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
    evaluate_model(model, test_loader, criterion, desc="测试集最终评估")
    
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
    collect_embeddings()