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
        self.output = nn.Linear(64, 3)  # 输出3个类别
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
        
        if self.output_activation == 'softmax':
            return F.softmax(out, dim=1)
        else:
            return out


# 2. 标准LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_activation=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 输出3个类别
        self.output_activation = output_activation
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
        
        if self.output_activation == 'softmax':
            return F.softmax(out, dim=1)
        else:
            return out


# 3. 多分类Focal Loss - 更关注某些类别（如 -1 和 1）
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """
        :param alpha: list或tensor，每个类别的权重，长度等于类别数
        :param gamma: 控制难易样本的调节程度
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        inputs: shape [batch_size, num_classes]
        targets: shape [batch_size]
        """
        targets = targets.long()
        logpt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)

        # 选择真实类别的概率
        logpt = logpt.gather(1, targets.view(-1, 1)).view(-1)
        pt = pt.gather(1, targets.view(-1, 1)).view(-1)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            at = alpha.gather(0, targets.view(-1))  # 按照目标类别提取权重
            loss = loss * at

        return loss.mean()


# 4. F1 Score Loss - 直接优化F1分数（适用于二分类）
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


# 5. 将 [-1, 0, 1] 映射到 [0, 1, 2]
def map_labels(y):
    y = np.array(y)  # 确保是 numpy 数组
    return (y + 1).astype(int)  # -1 → 0, 0 → 1, 1 → 2

def reverse_map(y):
    y = np.array(y)  # 确保是 numpy 数组
    return (y - 1).astype(int)  # 0→-1, 1→0, 2→1


def train_lstm(X_train, y_train, X_val, y_val, is_classification=False, use_focal_loss=False, use_f1_loss=False, use_enhanced_model=False):
    # 构建模型
    if use_enhanced_model:
        model = EnhancedLSTMModel(
            input_size=X_train.shape[2], 
            hidden_size=LSTM_HIDDEN_SIZE, 
            num_layers=NUM_LAYERS
        )
    else:
        model = LSTMModel(
            input_size=X_train.shape[2], 
            hidden_size=LSTM_HIDDEN_SIZE, 
            num_layers=NUM_LAYERS
        )
    
    # 使用GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 选择损失函数
    if is_classification:
        if use_focal_loss:
            class_weights = [2.0, 0.5, 2.0]  # 对 -1 (0) 和 1 (2) 提高关注度
            loss_fn = MultiClassFocalLoss(alpha=class_weights, gamma=2.0)
        elif use_f1_loss:
            loss_fn = F1Loss()
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 准备数据加载器
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)  # 分类任务用 long
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
            val_y_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
            val_logits = model(val_x)
            val_probs = F.softmax(val_logits, dim=1)
            val_preds_tensor = torch.argmax(val_probs, dim=1)

            # 将张量转为 numpy
            val_preds = val_preds_tensor.cpu().numpy()
            val_y_np = val_y_tensor.cpu().numpy()

            # 如果是分类问题，计算F1分数和其他指标
            if is_classification:
                from sklearn.metrics import classification_report, f1_score

                # 还原原始标签 [-1, 0, 1]
                val_preds_original = reverse_map(val_preds)
                val_y_original = reverse_map(val_y_np)

                report = classification_report(val_y_original, val_preds_original, output_dict=False)
                val_f1 = f1_score(val_y_original, val_preds_original, average='macro')

                print(f"\nEpoch {epoch+1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f}\n{report}")

                # 保存最佳模型
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                val_metrics.append(val_f1)
                    
    # 如果有更好的模型，加载它
    if best_model_state is not None and is_classification:
        model.load_state_dict(best_model_state)
    
    # 最终评估验证集
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_logits = model(val_x)
        val_probs = F.softmax(val_logits, dim=1)
        val_preds_tensor = torch.argmax(val_probs, dim=1)
        val_preds = val_preds_tensor.cpu().numpy()
        val_preds_original = reverse_map(val_preds)
    
    return model, val_preds_original, train_losses, val_metrics