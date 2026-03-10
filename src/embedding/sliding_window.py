import torch
import numpy as np
from tqdm import tqdm
import pickle
import os
from sentence_transformers import SentenceTransformer


def sliding_window_encode(texts, model, window_size=512, stride=256, inference_batch_size=32):
    model.max_seq_length = window_size
    all_embeddings = []
    tokenizer = model.tokenizer
    
    for i in tqdm(range(0, len(texts), 128), desc="Encoding documents"):
        batch_texts = texts[i : i + 128]
        inputs = tokenizer(batch_texts, truncation=False, padding=False)
        all_batch_chunks = []
        doc_to_chunk_map = []

        for doc_idx, input_ids in enumerate(inputs["input_ids"]):
            if len(input_ids) <= window_size:
                all_batch_chunks.append(tokenizer.decode(input_ids, skip_special_tokens=True))
                doc_to_chunk_map.append(doc_idx)
            else:
                for j in range(0, len(input_ids), stride):
                    chunk = input_ids[j : j + window_size]
                    all_batch_chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
                    doc_to_chunk_map.append(doc_idx)
                    if j + window_size >= len(input_ids):
                        break

        chunk_embs = model.encode(
            all_batch_chunks, 
            batch_size=inference_batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )

        doc_to_chunk_map = np.array(doc_to_chunk_map)
        for doc_idx in range(len(batch_texts)):
            mask = (doc_to_chunk_map == doc_idx)
            mean_emb = np.mean(chunk_embs[mask], axis=0)
            all_embeddings.append(mean_emb)

    return np.array(all_embeddings)


class EmbeddingManager:
    def __init__(self, model_name: str, local_model_dir: str = "./my_models"):
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.local_model_path = os.path.join(local_model_dir, model_name)
        self.model = None
    
    def load_model(self):
        if not os.path.exists(self.local_model_path):
            print(f"模型不存在，正在下载并保存到本地: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.save(self.local_model_path)
            print(f"模型已保存至：{self.local_model_path}")
        else:
            print(f"模型已存在，从本地加载: {self.local_model_path}")
            self.model = SentenceTransformer(self.local_model_path)
        return self.model
    
    def compute_embeddings(self, texts, save_path: str = None, window_size: int = 384, stride: int = 192, inference_batch_size: int = 32):
        if save_path and os.path.exists(save_path):
            print(f"加载已保存的嵌入: {save_path}")
            with open(save_path, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            print("计算新的嵌入...")
            embeddings = sliding_window_encode(
                texts, 
                self.model, 
                window_size=window_size,
                stride=stride,
                inference_batch_size=inference_batch_size
            )
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                print(f"嵌入已保存到: {save_path}")
        return embeddings
