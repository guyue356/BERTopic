from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import os


@dataclass
class BERTopicConfig:
    data_file: str = r"data\bertopic_data_new.jsonl"
    text_field: str = "text"
    timestamp_field: str = "year"
    
    version: str = "V_pp3"
    start_year: int = 2000
    end_year: int = 2025
    data_source: str = "abpa"
    
    embedding_model: str = "all-mpnet-base-v2"
    local_model_dir: str = "./my_models"
    
    window_size: int = 384
    stride: int = 192
    inference_batch_size: int = 32
    
    umap_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_neighbors": 100, # 50-100,越大结构越稳定
        "n_components": 10,
        "min_dist": 0.0, #
        "metric": "cosine",
        "random_state": 42,
        "low_memory": True
    })
    
    hdbscan_params: Dict[str, Any] = field(default_factory=lambda: {
        "min_samples": 10,
        "metric": "euclidean",
        "prediction_data": True
    })
    
    vectorizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "ngram_range": (1, 3),
        "min_df": 40,
        "max_features": 50000,
        "max_df": 0.8
    })
    
    search_sizes: List[int] = field(default_factory=lambda: [800,850,900,910,920,930,940,950,960,970,980,990,1000])
    
    stopwords_file: str = "data/stopwords.json"
    n_groups: int = 50
    
    @property
    def year_range(self) -> str:
        return f"{self.start_year}-{self.end_year}"
    
    @property
    def embeddings_path(self) -> str:

        #return f"results/embedding_results/embeddings_{self.data_source}_amb_slide_window3.pkl"
        return f"results/embedding_results/embeddings_PP_{self.data_source}_amb_slide_window3.pkl"
    
    @property
    def umap_cache_path(self) -> str:
        return f"umap_model_{self.data_source}_{self.start_year}-{self.end_year}_{self.version}.joblib"
    
    @property
    def model_save_path(self) -> str:
        # D:\WOS2025\bertopic_{data_source}_{year_range}_{version}
        # return f"results/topic_models/bertopic_{self.data_source}_{self.year_range}_{self.version}"
        return rf"D:\WOS2025\bertopic_{self.data_source}_{self.year_range}_{self.version}"
    
    @property
    def output_dir(self) -> str:
        return f"BERTopic_Results_{self.data_source}{self.year_range}_Allset_{self.version}"
    
    @property
    def local_model_path(self) -> str:
        return os.path.join(self.local_model_dir, self.embedding_model )
    