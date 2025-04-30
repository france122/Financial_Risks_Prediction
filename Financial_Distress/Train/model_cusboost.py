# model_cusboost.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble._weight_boosting import _samme_proba
from imblearn.under_sampling import RandomUnderSampler
from cus_sampling import cus_sampler 

from config import *  # 假设你在 config.py 中定义了 EPOCHS 等全局变量

class CUSBoostClassifier:
    def __init__(self, n_estimators=50, depth=3):
        self.M = n_estimators
        self.depth = depth
        self.undersampler = RandomUnderSampler(replacement=False)

    def fit(self, X, Y):
        self.models = []
        self.alphas = []
        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')

            # 注意 cus_sampler 是你自定义的采样函数
            X_undersampled, y_undersampled, chosen_indices = cus_sampler(X, Y)

            tree.fit(X_undersampled, y_undersampled, sample_weight=W[chosen_indices])
            P = tree.predict(X)

            err = np.sum(W[P != Y])
            if err > 0.5:
                continue
            if err <= 0:
                err = 1e-7

            try:
                alpha = 0.5 * (np.log((1 - err) / err)) if (np.log(1 - err) - np.log(err)) != 0 else 0
            except:
                alpha = 0

            W *= np.exp(-alpha * Y * P)
            W /= W.sum()

            self.models.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha * tree.predict(X)
        return np.sign(FX), FX

    def predict_proba(self, X):
        proba = sum(tree.predict_proba(X) * alpha for tree, alpha in zip(self.models, self.alphas))
        proba = np.array(proba) / sum(self.alphas)
        proba = np.exp(proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        return (proba / normalizer).astype(float)

    def predict_proba_samme(self, X):
        proba = sum(_samme_proba(est, 2, X) for est in self.models)
        proba = np.array(proba) / sum(self.alphas)
        proba = np.exp(proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        return (proba / normalizer).astype(float)


def reshape_for_boost(X):
    """
    将 [samples, timesteps, features] 转换为 [samples, timesteps * features]
    """
    return X.reshape(X.shape[0], -1)


def train_cusboost(X_train, y_train, X_val, y_val, n_estimators=EPOCHS, depth=3):
    """
    使用CUSBoost进行训练
    参数:
        - X_train, y_train: 训练集
        - X_val, y_val: 验证集
        - n_estimators: 弱分类器数量
        - depth: 每棵树的最大深度
    返回:
        - model: 训练好的模型
        - val_preds_proba: 验证集预测概率（正类）
        - train_losses: 占位，CUSBoost 不返回每轮损失
        - val_metrics: 验证F1分数
    """

    # reshape 处理
    X_train_reshaped = reshape_for_boost(X_train)
    X_val_reshaped = reshape_for_boost(X_val)

    # 初始化模型
    model = CUSBoostClassifier(n_estimators=n_estimators, depth=depth)

    # 训练模型
    model.fit(X_train_reshaped, y_train)

    # 验证
    val_preds, _ = model.predict(X_val_reshaped)
    val_preds = (val_preds > 0).astype(int)
    val_preds_proba = model.predict_proba(X_val_reshaped)[:, 1]

    f1 = f1_score(y_val, val_preds)
    print(f"CUSBoost Final F1 on Validation: {f1:.4f}")

    train_losses = []  # CUSBoost无每轮损失记录
    val_metrics = [f1]

    return model, val_preds_proba, train_losses, val_metrics
