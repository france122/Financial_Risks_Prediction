# model_lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import *

# 1. 改进LSTM模型
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_activation=None, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True  # 使用双向LSTM
        )
        # 双向LSTM输出维度翻倍
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 1),
            nn.Tanh()
        )
        self.fc = nn.Linear(hidden_size*2, 64)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(64, 1)
        self.output_activation = output_activation
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # 注意力机制
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)  # [batch, seq_len]
        attn_weights = attn_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*2]
        
        out = F.relu(self.fc(context))
        out = self.dropout(out)
        out = self.output(out)
        
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        else:
            return out

# 2. 标准LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_activation=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.output_activation = output_activation
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
        
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        else:
            return out

# 3. Focal Loss - 提高F1分数，更关注难以分类的样本
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 避免计算BCE_loss两次
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

# 4. F1 Score Loss - 直接优化F1分数 
class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # 转换为二进制预测
        y_pred = (inputs > 0.5).float()
        y_true = targets
        
        # 计算TP, FP, FN
        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1 - y_true))
        fn = torch.sum((1 - y_pred) * y_true)
        
        # 计算精确率和召回率
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        # 计算F1
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        
        # 返回1-F1，因为我们是最小化损失
        return 1 - f1

# 5. 计算类别权重以处理数据不平衡问题
def calculate_class_weights(y_train):
    """计算类别权重来处理不平衡数据"""
    class_counts = np.bincount(y_train.astype(int))
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

def train_lstm(X_train, y_train, X_val, y_val, is_classification=False, use_focal_loss=False, use_f1_loss=False, use_class_weights=False, use_enhanced_model=False):
    # 确定是分类还是回归问题
    output_activation = 'sigmoid' if is_classification else None
    
    # 选择损失函数
    if is_classification:
        if use_focal_loss:
            loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        elif use_f1_loss:
            loss_fn = F1Loss()
        else:
            if use_class_weights:
                # 计算类别权重
                weights = calculate_class_weights(y_train)
                loss_fn = lambda x, y: F.binary_cross_entropy(x, y, weight=weights.to(x.device))
            else:
                loss_fn = nn.BCELoss()
    else:
        loss_fn = nn.MSELoss()
    
    # 构建模型
    if use_enhanced_model:
        model = EnhancedLSTMModel(
            input_size=X_train.shape[2], 
            hidden_size=LSTM_HIDDEN_SIZE, 
            num_layers=NUM_LAYERS,
            output_activation=output_activation
        )
    else:
        model = LSTMModel(
            input_size=X_train.shape[2], 
            hidden_size=LSTM_HIDDEN_SIZE, 
            num_layers=NUM_LAYERS,
            output_activation=output_activation
        )
    
    # 使用GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 准备数据加载器
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 训练循环
    train_losses = []
    val_metrics = []  # 存储验证集上的F1分数
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # 评估验证集
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
            val_preds = model(val_x)
            
            # 如果是分类问题，计算F1分数
            if is_classification:
                val_preds_binary = (val_preds.cpu().numpy() > 0.5).astype(int)
                from sklearn.metrics import f1_score
                val_f1 = f1_score(y_val, val_preds_binary.flatten())
                val_metrics.append(val_f1)
                
                # 保存最佳模型
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f}, Val F1: {val_f1:.4f}")
            else:
                val_loss = nn.MSELoss()(val_preds, val_y).item()
                val_metrics.append(val_loss)
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")
    
    # 如果有更好的模型，加载它
    if best_model_state is not None and is_classification:
        model.load_state_dict(best_model_state)
    
    # 最终评估验证集
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = model(val_x).cpu().numpy()
    
    return model, val_preds, train_losses, val_metrics