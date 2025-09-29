import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from config import FINBERT_MODEL, BATCH_SIZE
import os
from huggingface_hub import snapshot_download
import requests
import time
from typing import List
from api_config import APIConfig

def download_model():
    try:
        # 设置下载目录
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_path = os.path.join(cache_dir, "models--yiyanghkust--finbert-tone")
        
        # 只在模型不存在时下载
        if not os.path.exists(model_path):
            print("[Info] Model not found locally, downloading...")
            snapshot_download(
                repo_id=FINBERT_MODEL,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            print("[Info] Model download completed!")
        else:
            print("[Info] Using local model cache...")
            
        return model_path
    except Exception as e:
        print(f"[Error] Failed to download model: {e}")
        print("[Info] Please check your internet connection or try using a proxy")
        raise

# 全局加载模型和分词器
try:
    print("[Info] Loading FinBERT model and tokenizer...")
    model_path = download_model()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("[Info] Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"[Error] Failed to load model: {e}")
    raise

def get_deepseek_summary(text: str, max_tokens: int = 512) -> str:
    """
    使用DeepSeek API对文本进行摘要
    
    Args:
        text: 输入文本
        max_tokens: 摘要最大token数
        
    Returns:
        str: 摘要文本
    """
    # 验证API配置
    if not APIConfig.validate_config():
        print("[Warning] DeepSeek API未配置，将返回原文截断")
        return text[:1024]
    
    # 构建提示词
    prompt = f"""请对以下管理层讨论与分析进行总结，要求：
1. 保留关键的财务信息和业务发展情况
2. 总结控制在{max_tokens}个token以内
3. 使用简洁客观的语言
4. 按照时间顺序组织内容

原文：
{text}
"""
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3  # 使用较低的temperature以获得更确定性的输出
    }
    
    try:
        response = requests.post(
            APIConfig.DEEPSEEK_API_URL,
            headers=APIConfig.get_deepseek_headers(),
            json=data
        )
        response.raise_for_status()
        summary = response.json()['choices'][0]['message']['content']
        return summary
    except Exception as e:
        print(f"调用DeepSeek API时出错: {str(e)}")
        # 如果API调用失败，返回原文的前512个字符
        return text[:1024]  # 使用1024个字符作为备选，因为中文字符在token中通常会被拆分

def get_finbert_embeddings(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """
    使用FinBERT模型对文本进行向量化
    
    Args:
        texts: 文本列表
        batch_size: 批处理大小
        
    Returns:
        np.ndarray: 文本向量矩阵
    """
    # 加载FinBERT模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    all_embeddings = []
    
    print("[Info] 开始处理文本...")
    print("[Info] 1. 使用DeepSeek进行文本摘要...")
    
    # 首先对所有文本进行摘要
    summarized_texts = []
    for text in tqdm(texts, desc="生成文本摘要"):
        if not isinstance(text, str) or len(text.strip()) == 0:
            # 处理空文本或非字符串输入
            summarized_texts.append("")
            continue
        
        # 添加延时以遵守API限制
        time.sleep(0.1)  # 根据API限制调整延时
        summary = get_deepseek_summary(text)
        summarized_texts.append(summary)
    
    print("[Info] 2. 使用FinBERT进行向量化...")
    
    # 分批处理文本
    for i in tqdm(range(0, len(summarized_texts), batch_size), desc="生成文本向量"):
        batch_texts = summarized_texts[i:i + batch_size]
        
        # 处理空文本
        batch_texts = [text if isinstance(text, str) and len(text.strip()) > 0 else "" for text in batch_texts]
        
        # 对批次进行编码
        encoded = tokenizer(batch_texts,
                          padding=True,
                          truncation=True,
                          max_length=512,
                          return_tensors="pt")
        
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        with torch.no_grad():
            model_output = model(**encoded)
            
        # 使用[CLS]标记的输出作为句子表示
        embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    
    # 合并所有批次的结果
    all_embeddings = np.vstack(all_embeddings)
    print(f"[Info] 文本向量化完成，shape: {all_embeddings.shape}")
    
    return all_embeddings 