import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import *


# 1. 改进LSTM模型：双向 + Attention + Dropout
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # 双向LSTM
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Tanh()
        )
        self.fc = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(64, 1)  # 输出一个logit（二分类）

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # 注意力机制
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        out = F.relu(self.fc(context))
        out = self.dropout(out)
        out = self.output(out).squeeze()

        return out


# 2. 标准LSTM模型（单向）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 输出一个logit

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 最后一个时间步输出
        return out.squeeze()


# 3. 二分类Focal Loss（更关注类别1）
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.98, gamma=2.0):
        """
        :param alpha: 更高值代表更关注类别1（正类）
        :param gamma: 难易样本调节项
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


# 4. F1 Score Loss（直接优化F1分数）
class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        y_pred = torch.sigmoid(inputs)
        y_pred = (y_pred > 0.5).float()
        y_true = targets.float()

        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1 - y_true))
        fn = torch.sum((1 - y_pred) * y_true)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        return 1 - f1


# 5. 映射标签：将原始数据中的多类标签转换为二分类 [0, 1]
def map_labels(y):
    y = np.array(y)
    return ((y == 1) | (y == -1)).astype(int)  # 所有非0视为1（异常类）


# 6. 训练函数（支持Focal Loss、F1 Loss、Attention LSTM）
def train_lstm_binary(X_train, y_train, X_val, y_val, use_focal_loss=True, use_f1_loss=False, use_enhanced_model=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    model = model.to(device)

    # 损失函数选择
    if use_focal_loss:
        loss_fn = BinaryFocalLoss(alpha=0.98)
    elif use_f1_loss:
        loss_fn = F1Loss()
    else:
        class_weight = torch.tensor([5.0]).to(device)  # 更加关注类别1
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 数据集准备
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_losses = []
    val_metrics = []  # 存储验证集上的F1分数
    best_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证评估
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_y_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

            val_logits = model(val_x)
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).float().cpu().numpy()
            val_true = val_y_tensor.cpu().numpy()

            from sklearn.metrics import classification_report, f1_score
            report = classification_report(val_true, val_preds, target_names=['Class 0', 'Class 1'])
            val_f1 = f1_score(val_true, val_preds)

            print(f"\nEpoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}\n{report}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()

        val_metrics.append(best_f1)

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 最终预测
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_logits = model(val_x)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).float().cpu().numpy()

    return model, val_preds, train_losses, best_f1