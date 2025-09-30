# Financial LSTM Feature Extraction Project

## 项目简介
本项目用于基于财务文本（如MDA）和财务指标，利用FinBERT和LSTM模型进行特征提取。项目通过多阶段降维和特征学习，生成具有时序特征的公司风险表示。

## 目录结构
```
├── config.py                    # 配置文件，集中管理参数
├── data_loader.py              # 数据加载与预处理
├── embedding.py                # 文本嵌入生成（FinBERT）
├── dimensionality_reduction.py # 降维模块（自编码器）
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
请将标准化后的数据文件命名为 `data.csv`，放在项目根目录下。数据需包含如下字段：
- Stkcd（公司代码）
- Accper（会计期间/年份）
- ManaDiscAnal（MDA文本）
- STPT（目标变量）
- 以及其他财务指标

## 运行方法
直接运行特征提取脚本：

```bash
python train.py
```

所有参数均可在 `config.py` 中修改，包括：
- AUTOENCODER_HIDDEN_DIM：自编码器降维后的维度（默认100）
- LSTM_HIDDEN_SIZE：LSTM隐藏层维度（默认128）
- 其他训练参数（批量大小、学习率等）

## 项目原理与流程

### 1. 数据处理流程
1. **数据加载**：读取标准化后的财务数据，每家公司每年一条记录
2. **文本嵌入**：使用FinBERT模型提取MDA文本的语义特征（768维向量）
3. **降维处理**：使用自编码器将768维文本向量降至100维
4. **特征拼接**：将降维后的文本特征与财务指标拼接
5. **序列构建**：构建滑动窗口式的时序数据（每个样本包含两年数据）
6. **LSTM特征提取**：使用LSTM模型学习128维的时序特征表示

### 2. 关键技术模块

#### （1）文本处理与降维
- FinBERT提取768维文本向量
- 自编码器降维至100维
- 保持文本语义信息的同时减少维度

#### （2）时序特征学习
- 使用滑动窗口构建连续两年的输入序列
- LSTM模型学习时序特征
- 输出128维的特征向量表示

#### （3）数据组织结构
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
- 训练集：2012-2017年数据
- 验证集：2018-2019年数据
- 测试集：2020年及以后数据

## 注意事项
1. 特征提取过程保持了数据的时序性
2. SMOTE过采样仅应用于训练集
3. 所有超参数都可在config.py中调整
4. 生成的特征向量可用于后续的预测任务
