---
name: patents-bertopic
description: Autonomous BERTopic development agent for patent data analysis. Uses patents_qy_sample.jsonl data, syncs DEV_SPEC.md into chapter-based reference files, identifies next pending task, implements code following spec patterns, runs tests with up to 3 auto-fix rounds. Use when user says "patent auto", "自动测试", "专利分析自动", or wants automated patent topic modeling workflow.
---
# 数据分析 BERTopic Skill

## 任务

使用 sample.jsonl 数据进行主题建模分析，实现基于BERTopic的专利主题分析pipeline。

## 环境

- Python环境: dl_env
- 数据文件: data/patents_qy_sample.jsonl

## 数据规格

### 数据格式

```json
{"id": 1, "year": 2001, "text": "文本内容...", "type": "patent"}
```

### 字段说明

- `id`: 专利唯一标识
- `year`: 专利申请年份
- `text`: 文本内容
- `type`: 类型 (patent)

## 配置规范

### 数据配置

```python
# src/config/bertopic_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class PatentBERTopicConfig:
    # 数据源
    data_file: str = "data/patents_qy_sample.jsonl"
    text_field: str = "text"
    timestamp_field: str = "year"
  
    # 版本控制
    version: str = "V1"
    start_year: int = 2000
    end_year: int = 2025
    data_source: str = "测试样本"
  
    # 嵌入模型
    embedding_model: str = "all-mpnet-base-v2"
    local_model_dir: str = "./my_models"
  
    # 滑动窗口参数
    window_size: int = 384
    stride: int = 192
    inference_batch_size: int = 32
  
    # UMAP参数
    umap_params: dict = None
  
    # HDBSCAN参数
    hdbscan_params: dict = None
  
    # Vectorizer参数
    vectorizer_params: dict = None
  
    # 搜索参数
    search_sizes: list = None
  
    def __post_init__(self):
        if self.umap_params is None:
            self.umap_params = {
                "n_neighbors": 15,
                "n_components": 5,
                "min_dist": 0.0,
                "metric": "cosine",
                "random_state": 5,
                "low_memory": True
            }
        if self.hdbscan_params is None:
            self.hdbscan_params = {
                "min_samples": 10,
                "metric": "euclidean",
                "prediction_data": True
            }
        if self.vectorizer_params is None:
            self.vectorizer_params = {
                "ngram_range": (1, 3),
                "min_df": 20,
                "max_features": 100000,
                "max_df": 0.7
            }
        if self.search_sizes is None:
            self.search_sizes = [100, 120, 150, 170, 200, 210, 220, 230, 240, 250, 300]
```

## 模块设计规范

### 1. 目录结构

```
BERTopic/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── bertopic_config.py
│   │   └── model_config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── sliding_window.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── umap_utils.py
│   │   ├── hdbscan_utils.py
│   │   └── bertopic_model.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   └── hierarchical.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── report_generator.py
│   │   └── charts.py
│   └── utils/
│       ├── __init__.py
│       ├── device.py
│       └── io_utils.py
├── notebooks/
├── main.py
└── requirements.txt
```

### 2. 数据模块 (src/data/)

#### loader.py

```python
import json
from typing import Tuple, List

def load_jsonl_data(file_path: str) -> Tuple[List[str], List[int]]:
    """加载JSONL格式的专利数据
  
    Returns:
        Tuple[文本列表, 年份列表]
    """
    docs = []
    timestamps = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            docs.append(item['text'])
            timestamps.append(item['year'])
    return docs, timestamps

def validate_patent_data(docs: List[str], timestamps: List[int]) -> bool:
    """验证专利数据有效性"""
    if not docs or not timestamps:
        return False
    if len(docs) != len(timestamps):
        return False
    return True
```

#### preprocessing.py

```python
import re
from typing import List

def preprocess_patent_texts(texts: List[str]) -> List[str]:
    """预处理专利文本"""
    return [preprocess_single_text(text) for text in texts]

def preprocess_single_text(text: str) -> str:
    """单条专利文本预处理"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
```

### 3. 主入口脚本

```python
# main.py
from src.config.bertopic_config import PatentBERTopicConfig
from src.data.loader import load_jsonl_data
from src.data.preprocessing import preprocess_patent_texts
from src.embedding import EmbeddingManager
from src.model import BERTopicPipeline
from src.visualization import ChartGenerator
import os

def main():
    config = PatentBERTopicConfig()
  
    docs_raw, timestamps = load_jsonl_data(config.data_file)
    docs = preprocess_patent_texts(docs_raw)
  
    embedding_mgr = EmbeddingManager(config.embedding_model)
    embeddings = embedding_mgr.get_or_compute(docs, config)
  
    pipeline = BERTopicPipeline(config)
    topic_model = pipeline.train(docs, embeddings, timestamps)
  
    charts = ChartGenerator(topic_model, config)
    charts.generate_all(output_dir=f"Patent_Results_{config.data_source}")

if __name__ == "__main__":
    main()
```

## 激活环境

```bash
conda activate dl_env
```

## 注意事项

- 数据为50000条专利样本
- 文本字段较长，需注意内存管理
- 年份范围: 2000-2025
- 使用滑动窗口嵌入处理长文本
