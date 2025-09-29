# Financial LSTM Feature Extraction Project

## 项目简介
本项目基于财务指标数据，利用LSTM模型进行特征提取。项目通过时序特征学习，生成具有时序特征的公司风险表示。

## 目录结构
```
├── config.py                    # 配置文件，集中管理参数
├── data_loader.py              # 数据加载与预处理
├── train.py                    # 特征提取主流程
├── requirements.txt            # 依赖包列表
├── README.md                   # 项目说明
```

## 依赖安装
建议使用Python 3.8+。

请先安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
请将标准化后的数据文件命名为 `input_stpt.csv`，放在项目根目录下。数据需包含如下字段：
- Stkcd（公司代码）
- Accper（会计期间/年份）
- STPT（目标变量）
- 以及其他财务指标（如ROA、ROE等）

## 运行方法
直接运行特征提取脚本：

```bash
python train.py
```

所有参数均可在 `config.py` 中修改，包括：
- LSTM_HIDDEN_SIZE：LSTM隐藏层维度（默认128）
- BATCH_SIZE：批处理大小
- LEARNING_RATE：学习率
- NUM_EPOCHS：训练轮数
- USE_SMOTE：是否使用SMOTE过采样

## 项目原理与流程

### 1. 数据处理流程
1. **数据加载**：读取标准化后的财务数据，每家公司每年一条记录
2. **序列构建**：构建滑动窗口式的时序数据（每个样本包含两年数据）
3. **LSTM特征提取**：使用LSTM模型学习128维的时序特征表示

### 2. 关键技术模块

#### （1）时序特征学习
- 使用滑动窗口构建连续两年的输入序列
- LSTM模型学习时序特征
- 输出128维的特征向量表示

#### （2）数据组织结构
最终生成的特征表格包含：
- Stkcd：公司代码
- year_1, year_2：输入年份对
- embedding_0 ~ embedding_127：128维LSTM特征
- target：对应的目标变量

### 3. 输出文件
程序会生成三个CSV文件：
- lstm_embeddings_train.csv：训练集特征
- lstm_embeddings_val.csv：验证集特征
- lstm_embeddings_test.csv：测试集特征

每个文件包含完整的公司ID、年份信息和对应的LSTM特征向量。

### 4. 数据集划分
- 训练集：2011-2017年数据
- 验证集：2018-2019年数据
- 测试集：2020年及以后数据

## 注意事项
1. 特征提取过程保持了数据的时序性
2. SMOTE过采样仅应用于训练集
3. 所有超参数都可在config.py中调整
4. 生成的特征向量可用于后续的预测任务
5. 数据预处理时会自动排除非数值型列（如文本字段）
