import pandas as pd
import os
import time

def generate_bertopic_report(umap_cfg, HDBSCAN_cfg,vectorizer_cfg,history, best_size, model_name, output_path,hierarchical_data=None):
    """
    生成 BERTopic 聚类实验报告的 HTML 文件。
    
    参数:
    - umap_cfg: dict, UMAP 的配置参数
    - history: list of dict, 包含网格搜索历史数据
    - best_size: int, 最优的 min_cluster_size
    - model_name: str, 嵌入模型的名称
    - output_path: str, 输出文件的完整路径
    - hierarchical_data: dict, 包含 {'n_groups': int, 'all_topics': list}，不传则不显示该单元
    """
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理表格数据
    history_df = pd.DataFrame(history)
    html_table = history_df.to_html(classes='table table-hover table-bordered', index=False)
    
    # 模型描述逻辑
    model_desc = (
        f"本分析选用 {model_name} 模型。其核心任务是将文本转化为高质量数字向量（Embedding）。"
        "该模型通过对输入文本上下文的深度理解，确保语义相近的文档在空间中具有更高的相似度。"
    )
    vectorizer_desc = "该模块负责将清洗后的文本转化为词频矩阵，是提取主题标签（Topic Keywords）的核心步骤。"

    # --- 新增：主题层次合并单元的条件渲染逻辑 ---
    hierarchical_html = "" # 默认不显示
    
    if hierarchical_data:
        n_groups = hierarchical_data.get('n_groups')
        all_topics = hierarchical_data.get('all_topics', [])
        
        # 只有当 n_groups 存在时才生成 HTML
        hierarchical_desc = f"""
        本分析采用了基于 <b>Scipy 链接矩阵</b> 的二叉树切分算法。首先提取 BERTopic 生成的 <code>Child_Left_ID</code> 
        与 <code>Child_Right_ID</code> 建立层次关系，随后通过 <code>cut_tree</code> 算法在全局范围内进行动态切分。
        该方法确保了合并过程不仅基于语义相似度，更遵循技术演进的逻辑层次，最终将原始细分主题聚类为 <b>{n_groups}</b> 个核心技术群。
        """
        
        hierarchical_html = f"""
        <div class="card mb-4 border-warning shadow-sm">
            <div class="card-header bg-warning text-dark">6. 主题层次合并算法 (Hierarchical Merging)</div>
            <div class="card-body">
                <p><b>算法逻辑：</b> {hierarchical_desc}</p>
                <div class="row mt-2">
                    <div class="col-md-4">
                        <div class="p-2 border rounded bg-light text-center">
                            <small class="text-muted d-block">合并前主题数</small>
                            <span class="h5">{len(all_topics)}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-2 border rounded bg-light text-center">
                            <small class="text-muted d-block">目标群组数 (n_groups)</small>
                            <span class="h5">{n_groups}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-2 border rounded bg-light text-center">
                            <small class="text-muted d-block">切分工具</small>
                            <span class="h5">Scipy cut_tree</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    # 构造 HTML 模板
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <title>BERTopic 实验报告</title>
        <style> 
            body {{ background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .container {{ max-width: 900px; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 0 20px rgba(0,0,0,0.05); margin-top: 30px; margin-bottom: 30px; }}
            .table th, .table td {{ text-align: center !important; vertical-align: middle !important; }}
            .best-row {{ background-color: #d4edda !important; font-weight: bold; border: 2px solid #28a745 !important; }} 
            .card-header {{ font-weight: bold; font-size: 1.1rem; }}
            .config-list {{ list-style: none; padding-left: 0; }}
            .config-list li {{ margin-bottom: 8px; padding: 5px; border-bottom: 1px inset #eee; }}
            code {{ color: #d63384; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="mb-5 text-center text-primary">BERTopic 主题聚类参数分析报告</h2>
            
            <div class="card mb-4 border-info shadow-sm">
                <div class="card-header bg-info text-white">1. 语义向量化配置 (Embedding)</div>
                <div class="card-body">
                    <p class="card-text"><b>预训练模型:</b> <code class="fs-5">{model_name}</code></p>
                    <p class="text-secondary small">{model_desc}</p>
                </div>
            </div>

            <div class="card mb-4 border-primary shadow-sm">
                <div class="card-header bg-primary text-white">2. UMAP 降维参数配置</div>
                <div class="card-body">
                    <ul class="row config-list">
                        {"".join([f"<li class='col-md-6'><b>{k}:</b> {v}</li>" for k, v in umap_cfg.items()])}
                    </ul>
                </div>
            </div>

            <div class="card mb-4 border-dark shadow-sm">
                <div class="card-header bg-dark text-white">3. HDBSCAN 聚类参数配置</div>
                <div class="card-body">
                    <ul class="row config-list">
                        {"".join([f"<li class='col-md-6'><b>{k}:</b> {v}</li>" for k, v in HDBSCAN_cfg.items()])}
                    </ul>
                </div>
            </div>

            <div class="card mb-4 border-success shadow-sm">
                <div class="card-header bg-success text-white">4. HDBSCAN 网格搜索结果</div>
                <div class="card-body">
                    <div class="alert alert-success">
                        🏆 <b>最优策略:</b> 当 <code>min_cluster_size</code> 为 <b>{best_size}</b> 时，模型在主题数量与噪声控制之间达到了最佳平衡。
                    </div>
                    <div class="table-responsive">
                        {html_table.replace('<tr><td>' + str(best_size), '<tr class="best-row"><td>' + str(best_size))}
                    </div>
                </div>
            </div>

            
            <div class="card mb-4 border-secondary shadow-sm h-100">
                    <div class="card-header bg-secondary text-white">5. CountVectorizer 配置 (特征提取)</div>
                    <div class="card-body">
                        <ul class="config-list">
                            {"".join([f"<li><b>{k}:</b> {str(v)}</li>" for k, v in vectorizer_cfg.items()])}
                        </ul>
                        <p class="text-muted small mt-2">{vectorizer_desc}</p>
                    </div>
                </div>

            {hierarchical_html}

            <hr>
            <p class="text-muted small text-end">报告自动生成于: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ 报告已成功生成至: {output_path}")