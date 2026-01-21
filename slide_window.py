import torch
import numpy as np
from tqdm import tqdm
import pickle

def sliding_window_encode(texts, model, window_size=512, stride=256, inference_batch_size=32):
    """
    针对 all-mpnet-base-v2 优化的长文本滑动窗口嵌入
    """
    # 确保模型最大长度设置正确
    model.max_seq_length = window_size
    all_embeddings = []
    
    # 预先定义分词参数，避免循环内重复调用
    tokenizer = model.tokenizer
    
    # 1. 准备阶段：将文本转化为 Token IDs (不截断)
    # 对于 12.5 万条数据，建议分块处理防止内存溢出
    for i in tqdm(range(0, len(texts), 128), desc="Encoding Patents"):
        batch_texts = texts[i : i + 128]
        
        # 编码原始文本为 IDs
        inputs = tokenizer(batch_texts, truncation=False, padding=False)
        all_batch_chunks = []
        doc_to_chunk_map = [] # 记录哪些 chunks 属于哪个文档

        for doc_idx, input_ids in enumerate(inputs["input_ids"]):
            if len(input_ids) <= window_size:
                all_batch_chunks.append(tokenizer.decode(input_ids, skip_special_tokens=True))
                doc_to_chunk_map.append(doc_idx)
            else:
                # 滑动窗口切片
                for j in range(0, len(input_ids), stride):
                    chunk = input_ids[j : j + window_size]
                    all_batch_chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
                    doc_to_chunk_map.append(doc_idx)
                    if j + window_size >= len(input_ids):
                        break

        # 2. 批量计算 Embedding (使用固定的 inference_batch_size 保护显存)
        chunk_embs = model.encode(
            all_batch_chunks, 
            batch_size=inference_batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )

        # 3. 聚合：使用逻辑索引聚合均值向量
        doc_to_chunk_map = np.array(doc_to_chunk_map)
        for doc_idx in range(len(batch_texts)):
            # 提取属于该文档的所有 chunk 向量
            mask = (doc_to_chunk_map == doc_idx)
            mean_emb = np.mean(chunk_embs[mask], axis=0)
            all_embeddings.append(mean_emb)

    return np.array(all_embeddings)

# --- 运行建议 ---
# final_embeddings = optimized_sw_encode(docs, embedding_model, inference_batch_size=128)