---
name: refactor-bertopic
description: Autonomous spec-driven development agent. Syncs DEV_SPEC.md into chapter-based reference files, identifies the next pending task from the schedule, implements code following spec architecture and patterns, runs tests with up to 3 auto-fix rounds, and persists progress with atomic commits. Use when user says "auto code", "自动开发", "自动写代码", "auto dev", "一键开发", "autopilot", or wants fully automated spec-to-code workflow.
---
# BERTopic 代码重构 Skill

## 任务

将 main_bert_topic.ipynb 中的功能重构到 src/ 文件夹，实现长期可维护的模块化结构。

## 重构原则

### 1. 目录结构设计

```
BERTopic/
├── src/                          # 核心源代码
│   ├── __init__.py
│   ├── config/                   # 配置管理
│   │   ├── __init__.py
│   │   ├── bertopic_config.py   # BERTopic相关配置
│   │   └── model_config.py       # 模型路径配置
│   ├── data/                     # 数据处理
│   │   ├── __init__.py
│   │   ├── loader.py             # 数据加载器
│   │   └── preprocessing.py      # 文本预处理
│   ├── embedding/                # 向量化模块
│   │   ├── __init__.py
│   │   ├── base.py               # 基础嵌入类
│   │   └── sliding_window.py     # 滑动窗口嵌入 (从slide_window.py迁移)
│   ├── model/                    # 主题模型
│   │   ├── __init__.py
│   │   ├── umap_utils.py         # UMAP降维工具
│   │   ├── hdbscan_utils.py      # HDBSCAN聚类工具
│   │   └── bertopic_model.py     # BERTopic模型封装
│   ├── clustering/               # 聚类分析
│   │   ├── __init__.py
│   │   └── hierarchical.py       # 层次聚类
│   ├── visualization/             # 可视化模块
│   │   ├── __init__.py
│   │   ├── report_generator.py   # 报告生成 (从report_generator.py迁移)
│   │   └── charts.py             # 图表生成
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── device.py             # GPU设备检测
│       └── io_utils.py           # 文件IO工具
├── notebooks/                    # Jupyter notebooks (原main_bert_topic_test.ipynb迁移)
├── config.yaml                   # 统一配置文件 (可选)
├── main.py                       # 入口脚本
└── requirements.txt
```

### 2. 模块设计规范

#### 配置模块 (src/config/)

- 集中管理所有配置参数
- 支持YAML/JSON配置文件
- 包含默认参数和可覆盖机制

#### 数据模块 (src/data/)

- **loader.py**:
  - `load_json_data(file_path: str) -> Tuple[List[str], List[int]]`
  - `validate_data(docs: List[str], timestamps: List[int]) -> bool`
- **preprocessing.py**:
  - `preprocess_texts(texts: List[str]) -> List[str]`
  - `clean_special_chars(text: str) -> str`

#### 嵌入模块 (src/embedding/)

- **base.py**: 定义嵌入器基类
- **sliding_window.py**:
  - 保留现有 `sliding_window_encode()` 函数
  - 添加 `EmbeddingManager` 类管理模型加载和缓存

#### 模型模块 (src/model/)

- **umap_utils.py**:
  - `create_umap_model(params: dict) -> UMAP`
  - `load_or_fit_umap(embeddings, params, cache_path) -> np.ndarray`
- **hdbscan_utils.py**:
  - `grid_search_clustering(umap_embeddings, search_sizes, base_params) -> dict`
  - `find_best_clustering(search_results) -> int`
- **bertopic_model.py**:
  - `BERTopicPipeline` 类封装完整流程

#### 聚类模块 (src/clustering/)

- **hierarchical.py**:
  - `merge_topics(topic_model, docs, n_groups) -> DataFrame`
  - `get_linkage_matrix(hierarchical_topics) -> np.ndarray`

#### 可视化模块 (src/visualization/)

- **report_generator.py**: 保留现有函数，添加类型注解
- **charts.py**:
  - `generate_barchart(topic_model, docs, ...) -> go.Figure`
  - `generate_documents_plot(...) -> go.Figure`
  - `generate_all_charts(topic_model, docs, output_dir) -> None`

### 3. 配置文件示例

```python
# src/config/bertopic_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class BERTopicConfig:
    # 数据源
    data_file: str = "data/patents_qy.json"
    text_field: str = "combined_text"
    timestamp_field: str = "year"
  
    # 版本控制
    version: str = "V0"
    start_year: int = 2000
    end_year: int = 2025
    data_source: str = "QY专利"
  
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

### 4. 主入口脚本示例

```python
# main.py
from src.config.bertopic_config import BERTopicConfig
from src.data.loader import DataLoader
from src.data.preprocessing import TextPreprocessor
from src.embedding import EmbeddingManager
from src.model import BERTopicPipeline
from src.visualization import ChartGenerator
import os

def main():
    # 1. 加载配置
    config = BERTopicConfig()
  
    # 2. 加载数据
    loader = DataLoader(config.data_file)
    docs_raw, timestamps = loader.load()
  
    # 3. 预处理
    preprocessor = TextPreprocessor()
    docs = preprocessor.process(docs_raw)
  
    # 4. 嵌入
    embedding_mgr = EmbeddingManager(config.embedding_model)
    embeddings = embedding_mgr.get_or_compute(docs, config)
  
    # 5. 训练模型
    pipeline = BERTopicPipeline(config)
    topic_model = pipeline.train(docs, embeddings, timestamps)
  
    # 6. 可视化
    charts = ChartGenerator(topic_model, config)
    charts.generate_all(output_dir=f"BERTopic_Results_{config.data_source}{config.start_year}-{config.end_year}")

if __name__ == "__main__":
    main()
```

### 5. 迁移步骤

1. **创建目录结构**

   - 创建 `src/` 及所有子目录
   - 添加 `__init__.py` 文件
2. **迁移配置文件**

   - 从notebook提取配置参数到 `src/config/bertopic_config.py`
3. **迁移数据模块**

   - 创建 `src/data/loader.py`
   - 创建 `src/data/preprocessing.py`
4. **迁移嵌入模块**

   - 移动 `slide_window.py` 到 `src/embedding/`
   - 创建 `EmbeddingManager` 类
5. **迁移模型模块**

   - 提取UMAP/HDBSCAN相关代码
   - 创建 `BERTopicPipeline` 类
6. **迁移可视化模块**

   - 移动 `report_generator.py` 到 `src/visualization/`
   - 创建 `ChartGenerator` 类
7. **创建入口脚本**

   - 编写 `main.py` 整合所有模块
8. **测试验证**

   - 确保迁移后功能与原notebook一致

### 6. 注意事项

- 保持与现有模块 (如 `report_generator.py`, `slide_window.py`) 的兼容性
- 添加详细的类型注解和文档字符串
- 使用绝对导入避免相对导入问题
- 保持配置与代码分离便于调整参数
