from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Optional, Dict, Any
import json
import os


def load_stopwords(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = json.load(f)
        print(f"已加载 {len(stopwords['common'])} 个停用词")
        return stopwords['common']
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return set()
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        return set()


class BERTopicPipeline:
    def __init__(self, config):
        self.config = config
        self.topic_model = None
        self.hierarchical_topics = None
    
    def train(self, docs: List[str], embeddings, sentence_model, umap_model, best_cluster_size: int):#, best_labels=None
        stop_words = load_stopwords(self.config.stopwords_file)
        
        vectorizer_model = CountVectorizer(
            **self.config.vectorizer_params,
            stop_words=stop_words
        )
        
        best_clusterer = HDBSCAN(
            **self.config.hdbscan_params,
            min_cluster_size=best_cluster_size
        )
        
        # 使用 manual topic modeling 方式，直接传入 best_labels
        # if best_labels is not None:
        #     # 绕过 UMAP 和 HDBSCAN，直接使用 best_labels
        #     self.topic_model = BERTopic(
        #         embedding_model=sentence_model,
        #         vectorizer_model=vectorizer_model,
        #         umap_model=None,
        #         hdbscan_model=None,
        #         top_n_words=15,
        #         calculate_probabilities=False
        #     )
        #     topics, _ = self.topic_model.fit_transform(docs, embeddings=embeddings, y=best_labels)
        # else:
        self.topic_model = BERTopic(
            embedding_model=sentence_model,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=best_clusterer,
            calculate_probabilities=False
        )
        topics, _ = self.topic_model.fit_transform(docs, embeddings=embeddings)
        
        save_path = self.config.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.topic_model.save(save_path)
        print(f"主题模型已保存到 {save_path}")
        
        return self.topic_model
    
    def load(self, path: str):
        self.topic_model = BERTopic.load(path)
        print("初始主题模型加载成功！")
        return self.topic_model
    
    def compute_hierarchical_topics(self, docs: List[str]):
        if self.topic_model:
            self.hierarchical_topics = self.topic_model.hierarchical_topics(docs)
        return self.hierarchical_topics
