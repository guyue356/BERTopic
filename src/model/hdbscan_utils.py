from hdbscan import HDBSCAN
import time
from typing import Dict, Any, List, Tuple, Union
import numpy as np


def grid_search_clustering(umap_embeddings: np.ndarray, search_sizes: List[int], HDBSCAN_cfg: Dict[str, Any]) -> Any:
    best_topics = None
    best_m_size = None
    #best_labels = None
    best_score = float('inf')
    search_history = []  # 新增：用于记录网格搜索过程

    print(f"{'Size':<10} | {'主题数':<10} | {'负样本数':<12} | {'噪声比例':<12} | {'耗时':<8}")
    print("-" * 65)

    for m_size in search_sizes:
        start_t = time.time()
      
        clusterer = HDBSCAN(**HDBSCAN_cfg, min_cluster_size=m_size)
        labels = clusterer.fit_predict(umap_embeddings)
        
        # 计算指标
        n_outliers = (labels == -1).sum()  # 负样本数
        n_topics = len(set(labels)) - (1 if -1 in labels else 0)
        outlier_perc = n_outliers / len(labels)
        duration = time.time() - start_t

        # 保存历史记录
        res = {
            "min_cluster_size": m_size,
            "n_topics": n_topics,
            "n_outliers": n_outliers,
            "outlier_perc": f"{outlier_perc:.1%}",
            "duration": f"{duration:.1f}s",
            "is_best": False
        }
        search_history.append(res)

        # 筛选逻辑：选噪声最小的
        if outlier_perc < best_score:
            best_topics = n_topics
            best_score = outlier_perc
            best_m_size = m_size
            #best_labels = labels

        print(f"{m_size:<10} | {n_topics:<10} | {n_outliers:<12} | {outlier_perc:<12.1%} | {duration:<8.1f}s")
    
    print("-" * 65)
    if best_m_size is None:
        print("未在预设主题数范围内找到参数，建议调低 min_dist 或检查 UMAP 效果")
    else:
        print(f"🏆 符合条件的负样本数最少的参数值: min_cluster_size = {best_m_size} (负样本比例: {best_score:.1%}, 主题数：{best_topics})")

    # 标记最优参数
    for item in search_history:
        if item["min_cluster_size"] == best_m_size:
            item["is_best"] = True

    return search_history, best_m_size, best_topics #, best_labels

def find_best_clustering(search_results: List[Dict]) -> int:
    for item in search_results:
        if item["is_best"]:
            return item["min_cluster_size"]
    return search_results[0]["min_cluster_size"] if search_results else 50
