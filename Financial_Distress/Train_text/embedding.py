import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from config import FINBERT_MODEL, BATCH_SIZE
import os
from huggingface_hub import snapshot_download

import os
# 设置HF镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_model():
    try:
        from huggingface_hub import snapshot_download
        
        # 设置下载目录
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_path = os.path.join(cache_dir, "models--yiyanghkust--finbert-tone")
        
        # 只在模型不存在时下载
        if not os.path.exists(model_path):
            print("[Info] Using mirror: https://hf-mirror.com")
            print("[Info] Model not found locally, downloading...")
            snapshot_download(
                repo_id="yiyanghkust/finbert-tone",
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            print("[Info] Model download completed!")
        else:
            print("[Info] Using local model cache...")
            
        return model_path
    except Exception as e:
        print(f"[Error] Failed to download model: {e}")
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

def get_finbert_embeddings(texts, batch_size=BATCH_SIZE):
    embeddings = []
    
    # 分批处理文本
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        # 对文本进行编码
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            # 使用[CLS]标记的输出作为整个文本的表示
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings) 