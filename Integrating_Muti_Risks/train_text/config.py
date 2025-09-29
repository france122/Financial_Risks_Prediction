# 数据相关
DATA_FILE = 'input_stpt.csv'
EXCLUDE_COLUMNS = ['STPT', 'Accper', 'Stkcd', 'ManaDiscAnal']

# 随机种子设置
RANDOM_SEED = 42

# 模型相关
FINBERT_MODEL = 'yiyanghkust/finbert-tone'
FINBERT_EMBEDDING_DIM = 768
FINANCIAL_INDICATOR_DIM = 27  # 可根据实际数据自动计算

# 特征选择参数
USE_FINANCIAL_INDICATORS = True  # 是否使用财务指标特征
USE_TEXT_EMBEDDING = False  # 是否使用文本向量特征

# 自编码器参数
AUTOENCODER_HIDDEN_DIM = 100  # 文本向量降维后的维度
AUTOENCODER_EPOCHS = 50

# LSTM参数
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_OUTPUT_SIZE = 1

# 过采样参数
USE_SMOTE = True  # 是否在数据加载时使用SMOTE过采样

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# DeepSeek API配置
USE_TEXT_SUMMARY = True  # 是否使用文本摘要
MAX_SUMMARY_TOKENS = 512  # 摘要最大token数
DEEPSEEK_API_RATE_LIMIT = 0.1  # API调用间隔（秒）

