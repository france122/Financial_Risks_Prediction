# 数据相关
DATA_FILE = 'data.csv'
EXCLUDE_COLUMNS = ['Unnamed: 0','STPT', 'Accper', 'Stkcd', 'ManaDiscAnal']

# 随机种子设置
RANDOM_SEED = 42

# 模型相关
FINBERT_MODEL = 'yiyanghkust/finbert-tone'
FINBERT_EMBEDDING_DIM = 768
FINANCIAL_INDICATOR_DIM = 33  # 可根据实际数据自动计算

# 自编码器参数
AUTOENCODER_HIDDEN_DIM = 50  # 文本向量降维后的维度
AUTOENCODER_EPOCHS = 100

# LSTM参数
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_OUTPUT_SIZE = 1

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
