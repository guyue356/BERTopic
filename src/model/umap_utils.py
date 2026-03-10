from umap import UMAP
import joblib
import os
from typing import Dict, Any, Optional
import numpy as np


def create_umap_model(params: Dict[str, Any]) -> UMAP:
    return UMAP(**params)


def load_or_fit_umap(embeddings: np.ndarray, params: Dict[str, Any], cache_path: str) -> Any:
    if os.path.exists(cache_path):
        print(f"🚀 检测到已存在的降维模型，正在直接加载: {cache_path}")
        umap_embeddings = joblib.load(cache_path)
    else:
        print("⏳ 未发现降维模型，开始执行 UMAP 降维计算...")
        umap_model = UMAP(**params)
        umap_embeddings = umap_model.fit_transform(embeddings)
        joblib.dump(umap_embeddings, cache_path)
        print(f"✔ UMAP 降维完成并已保存至: {cache_path}")
    return umap_embeddings
