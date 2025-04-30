# config.py

# ==== 数据配置 ====
INPUT_PATH = "ST_data/filtered_data_with_more.csv"
OUTPUT_DIR = "results/"
MODEL_SAVE_PATH = OUTPUT_DIR + "lstm_model.pt"
LOG_PATH = OUTPUT_DIR + "training_log.txt"

# ==== 特征选择 ====
# FEATURE_COLUMNS = [ 'CR','AudOp', 'EmotionTone1', 'EmotionTone2', 
#        'ROA', 'PS(TTM)', 'ROE(C)',  'QR', 'WC/TA', 'WC/S', 'EBIT/TA',
#        'EBITDA/TA', 'CF/TA', 'CF/S', 'CF/TL', 'EBIT/S', 'EBITDA/S', 'RE/TA',
#        'S/TA', 'QA/S', 'CL/TA', 'LTL/TA', 'TL/TA', 'I/WC', 'LTL/CA', 'lnTA',
#        'lnS', 'FA/TA', 'E/TA', 'CL/TL', 'GDPGrowthRate', 'PerCptGDPGrowthRate',
#        'GDP', 'PerCapitaGDP', 'Population', 'PopGrowthRate', 'UrbanPopRatio',
#        'CPI']
FEATURE_COLUMNS = ['ROA', 'PS(TTM)', 'ROE(C)', 'CR', 'QR',
       'WC/TA', 'WC/S', 'EBIT/TA', 'EBITDA/TA', 'CF/TA', 'CF/S', 'CF/TL',
       'EBIT/S', 'EBITDA/S', 'RE/TA', 'S/TA', 'QA/S', 'CL/TA', 'LTL/TA',
       'TL/TA', 'I/WC', 'LTL/CA', 'lnTA', 'lnS', 'FA/TA', 'E/TA', 'CL/TL']
LABEL_COLUMN = "STPT"
COMPANY_COLUMN = "Stkcd"
YEAR_COLUMN = "Accper"

# ==== 时间窗口参数 ====
# 以下参数定义了滑动窗口的行为：
# - WINDOW_SIZE: 每个样本使用的连续年份数量作为特征
# - TARGET_SHIFT: 目标预测年份与特征窗口末尾的偏移量
#
# 例如:
# WINDOW_SIZE=2, TARGET_SHIFT=1 表示:
# 使用 [2010, 2011] 预测 2012
# 使用 [2011, 2012] 预测 2013
# 依此类推...
#
# 要增加预测的年数，增加 WINDOW_SIZE
# 要预测更远的未来，增加 TARGET_SHIFT
WINDOW_SIZE = 2       # 用过去连续的WINDOW_SIZE年预测未来TARGET_SHIFT年
TARGET_SHIFT = 1       # 预测特征窗口结束后的第TARGET_SHIFT年

# ==== 模型参数 ====

USE_SMOTE = False
IS_LSTM = False

LSTM_HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
SEED = 42
