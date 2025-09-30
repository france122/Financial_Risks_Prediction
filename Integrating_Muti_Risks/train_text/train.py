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
        
        log_msg=f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss/len(train_loader):.4f}"
        print(log_msg)
        with open("training_log.txt", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    
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
#         print(f"\n[Debug] 各列表长度: {lengths}")
#         print(f"[Debug] 年份示例:")
#         for i in range(min(5, len(all_years_1))):
#             print(f"  - 样本{i}: year_1={all_years_1[i]}, year_2={all_years_2[i]}")
        
#         if len(set(lengths.values())) > 1:
#             raise ValueError(f"列表长度不一致: {lengths}")
        
        # 创建DataFrame
        data_dict = {
            'Stkcd': all_companies,
            'target': all_targets,
            'year_1': all_years_1,
            'year_2': all_years_2
        }
        
        # 添加所有embedding列到字典中
        embeddings_array = np.array(all_embeddings)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(-1, 1)
            
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


if __name__ == "__main__":
    collect_embeddings()
