# utils.py

import os
import torch
import numpy as np
import random
import pickle

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, path, is_lstm=True):
    """保存模型，支持不同类型的模型
    
    Args:
        model: 要保存的模型
        path: 保存路径
        is_lstm: 是否是LSTM模型，如果为False则假定为CUSBoost模型
    """
    if is_lstm:
        # LSTM模型使用PyTorch的保存方法
        torch.save(model.state_dict(), path)
    else:
        # 其他模型（如CUSBoost）使用pickle保存
        with open(path, 'wb') as f:
            pickle.dump(model, f)

def load_model(model, path, is_lstm=True):
    """加载模型，支持不同类型的模型
    
    Args:
        model: 模型对象
        path: 模型保存路径
        is_lstm: 是否是LSTM模型
    
    Returns:
        加载后的模型
    """
    if is_lstm:
        model.load_state_dict(torch.load(path))
        return model
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_to_file(log_path, text):
    with open(log_path, 'a') as f:
        f.write(text + '\n')

