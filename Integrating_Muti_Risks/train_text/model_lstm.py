import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import RANDOM_SEED
from config import RANDOM_SEED, AUTOENCODER_HIDDEN_DIM, AUTOENCODER_EPOCHS, BATCH_SIZE
class TextAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=100):
        super(TextAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



# 修改train_autoencoder函数的默认参数
def train_autoencoder(text_embeddings, hidden_dim=AUTOENCODER_HIDDEN_DIM, epochs=AUTOENCODER_EPOCHS, batch_size=BATCH_SIZE):
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 转换数据为tensor
    data = torch.FloatTensor(text_embeddings)
    
    # 创建模型和优化器
    model = TextAutoencoder(input_dim=text_embeddings.shape[1], hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # 训练自编码器
    model.train()
    n_batches = len(data) // batch_size
    
    print("[Info] 开始训练文本自编码器...")
    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = data[start_idx:end_idx]
            
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/n_batches:.4f}")
    
    # 返回训练好的模型
    return model

# 修改reduce_dimension函数的默认参数
def reduce_dimension(text_embeddings, hidden_dim=AUTOENCODER_HIDDEN_DIM):
    """将文本向量降维到指定维度
    
    Args:
        text_embeddings: numpy array, shape (n_samples, 768)
        hidden_dim: 目标维度，默认100
    
    Returns:
        reduced_embeddings: numpy array, shape (n_samples, hidden_dim)
    """
    print(f"[Info] 将文本向量从768维降至{hidden_dim}维...")
    
    # 训练自编码器
    model = train_autoencoder(text_embeddings, hidden_dim=hidden_dim)
    
    # 使用encoder部分进行降维
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(text_embeddings)
        reduced_embeddings, _ = model(data)
    
    return reduced_embeddings.numpy()
