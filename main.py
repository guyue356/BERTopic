import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from src.config.bertopic_config import BERTopicConfig
from src.data.loader import DataLoader, TextPreprocessor
from src.embedding.sliding_window import EmbeddingManager
from src.model.umap_utils import create_umap_model, load_or_fit_umap
from src.model.hdbscan_utils import grid_search_clustering
from src.model.bertopic_model import BERTopicPipeline
from src.visualization import generate_bertopic_report, ChartGenerator
from src.utils.device import get_device


def main():
    config = BERTopicConfig()
    
    device = get_device()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    
    print("=" * 80)
    print("BERTopic Topic Modeling Pipeline")
    print("=" * 80)
    
    print("\n[Step 1] Loading data...")
    loader = DataLoader(config.data_file, config.text_field, config.timestamp_field)
    docs_raw, timestamps = loader.load()
    print(f"[OK] Data loaded, total {len(docs_raw)} records")
    
    print("\n[Step 2] Preprocessing data...")
    preprocessor = TextPreprocessor()
    docs = preprocessor.preprocess_texts(docs_raw)
    
    doc_count = len(docs)
    time_count = len(timestamps)
    assert doc_count == time_count, f"Data mismatch! Docs: {doc_count}, Timestamps: {time_count}."
    print(f"[OK] Data validation passed, {doc_count} records")
    
    print("\n[Step 3] Loading SentenceTransformer model...")
    embedding_mgr = EmbeddingManager(config.embedding_model, config.local_model_dir)
    sentence_model = embedding_mgr.load_model()
    print(f"[OK] Model loaded: {config.embedding_model}")
    
    print("\n[Step 4] Computing embeddings...")
    embeddings = embedding_mgr.compute_embeddings(
        docs,
        save_path=config.embeddings_path,
        window_size=config.window_size,
        stride=config.stride,
        inference_batch_size=config.inference_batch_size
    )
    print(f"[OK] Embeddings computed, shape: {embeddings.shape}")
    
    print("\n[Step 5] UMAP dimensionality reduction...")
    umap_model = create_umap_model(config.umap_params)
    umap_embeddings = load_or_fit_umap(embeddings, config.umap_params, config.umap_cache_path)
    print(f"[OK] UMAP completed, shape: {umap_embeddings.shape}")
    
    print("\n[Step 6] HDBSCAN grid search...")
    search_history, best_m_size, best_topics = grid_search_clustering( #, best_labels
        umap_embeddings,
        config.search_sizes,
        config.hdbscan_params
    )
    print(f"[OK] Best min_cluster_size: {best_m_size}, Topics: {best_topics}")
    
    print("\n[Step 7] Training BERTopic model...")
    pipeline = BERTopicPipeline(config)
    topic_model = pipeline.train(
        docs,
        embeddings,
        sentence_model,
        umap_model,
        best_m_size,
        #best_labels
    )
    topic_info = topic_model.get_topic_info()
    final_topic_count = len(topic_info[topic_info['Topic'] != -1])
    print(f"[OK] BERTopic model trained, 最终主题数: {final_topic_count}, 网格搜索主题数: {best_topics}")
    
    print("\n[Step 8] Computing hierarchical topics...")
    hierarchical_topics = pipeline.compute_hierarchical_topics(docs)
    print(f"[OK] Hierarchical topics computed")
    
    print("\n[Step 9] Generating visualizations...")
    os.makedirs(config.output_dir, exist_ok=True)
    charts = ChartGenerator(topic_model, config)
    chart_data = charts.generate_all_charts(docs, hierarchical_topics, embeddings, timestamps, config.output_dir, config.n_groups)
    print(f"[OK] Visualizations generated")
    
    print("\n[Step 10] Generating report...")
    # 使用英文数据源名称避免编码问题
    safe_data_source = config.data_source.replace("专利", "Patent").replace("样本", "Sample").replace("测试", "Test")
    report_name = os.path.join(config.output_dir, f"report_{config.year_range}_{safe_data_source}_{config.version}.html")
    
    # 准备层次聚类数据
    h_data = {
        "n_groups": config.n_groups,
        "all_topics": chart_data["all_topics"]
    }
    
    generate_bertopic_report(
        umap_cfg=config.umap_params,
        HDBSCAN_cfg=config.hdbscan_params,
        vectorizer_cfg=config.vectorizer_params,
        history=search_history,
        best_size=best_m_size,
        model_name=config.embedding_model,
        output_path=report_name,
        hierarchical_data=h_data
    )
    print(f"[OK] Report generated: {report_name}")
    
    print("\n" + "=" * 80)
    print("BERTopic Topic Modeling Pipeline Completed!")
    print("=" * 80)
    
    return {
        'topic_model': topic_model,
        'docs': docs,
        'embeddings': embeddings,
        'hierarchical_topics': hierarchical_topics,
        'timestamps': timestamps
    }


if __name__ == "__main__":
    results = main()
