import pandas as pd
import os
import time
from typing import Dict, Any, Optional, List


def generate_bertopic_report(umap_cfg: Dict, HDBSCAN_cfg: Dict, vectorizer_cfg: Dict, history: List[Dict], best_size: int, model_name: str, output_path: str, hierarchical_data: Optional[Dict] = None):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if history:
        history_df = pd.DataFrame(history)
        html_table = history_df.to_html(classes='table table-hover table-bordered', index=False)
    else:
        html_table = "<p>无网格搜索历史记录</p>"
    
    model_desc = (
        f"本分析选用 {model_name} 模型。其核心任务是将文本转化为高质量数字向量（Embedding）。"
        "该模型通过对输入文本上下文的深度理解，确保语义相近的文档在空间中具有更高的相似度。"
    )
    vectorizer_desc = "该模块负责将清洗后的文本转化为词频矩阵，是提取主题标签（Topic Keywords）的核心步骤。"

    hierarchical_html = ""
    
    if hierarchical_data:
        n_groups = hierarchical_data.get('n_groups')
        all_topics = hierarchical_data.get('all_topics', [])
        
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
                        {html_table.replace(f'<tr><td>{best_size}', f'<tr class="best-row"><td>{best_size}') if best_size else html_table}
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
    print(f"报告已成功生成至: {output_path}")
    
    # 确保文件确实存在
    if os.path.exists(output_path):
        print(f"确认报告文件存在: {output_path}, 大小: {os.path.getsize(output_path)} 字节")
    else:
        print(f"警告: 报告文件未创建: {output_path}")
    
    # 生成导航索引页
    generate_navigation_index(output_dir, hierarchical_data, output_path)


def generate_navigation_index(output_dir: str, hierarchical_data: Optional[Dict] = None, report_filename: str = None):
    """生成导航索引页面"""
    # 从目录路径中提取信息
    dir_name = os.path.basename(output_dir)
    parts = dir_name.split('_')
    
    # 尝试提取年份范围和数据源
    year_range = "2000-2025"
    data_source = "TestSample"
    display_data_source = "测试样本"  # 用于显示的中文名称
    version = "V1"
    
    # 如果提供了报告文件名，直接从文件名提取信息
    if report_filename:
        report_basename = os.path.basename(report_filename)
        if report_basename.startswith("report_") and report_basename.endswith(".html"):
            report_parts = report_basename.replace('report_', '').replace('.html', '').split('_')
            if len(report_parts) >= 3:
                year_range = report_parts[0]
                data_source = report_parts[1]
                version = report_parts[2]
                # 将英文数据源名称转换回中文显示
                if data_source == "TestSample":
                    display_data_source = "测试样本"
                elif data_source == "QYPatentSample":
                    display_data_source = "QY专利样本"
                elif data_source == "abpa":
                    display_data_source = "ABPA"
                else:
                    display_data_source = data_source
    
    # 如果没有提供报告文件名，尝试查找目录中已有的报告文件
    if not report_filename:
        import glob
        report_files = glob.glob(os.path.join(output_dir, "report_*.html"))
        if report_files:
            report_filename = report_files[0]
            report_basename = os.path.basename(report_filename)
            report_parts = report_basename.replace('report_', '').replace('.html', '').split('_')
            if len(report_parts) >= 3:
                year_range = report_parts[0]
                data_source = report_parts[1]
                version = report_parts[2]
                if data_source == "TestSample":
                    display_data_source = "测试样本"
                elif data_source == "QYPatentSample":
                    display_data_source = "QY专利样本"
                elif data_source == "abpa":
                    display_data_source = "ABPA"
                else:
                    display_data_source = data_source
    
    # 如果仍未找到，使用目录名作为后备
    if not report_filename or not os.path.exists(os.path.join(output_dir, os.path.basename(report_filename))):
        if len(parts) >= 4:
            try:
                data_source_part = parts[1]
                year_range_part = parts[2] if len(parts) > 2 else "2000-2025"
                
                import re
                match = re.search(r'([^\d]+)(\d+-\d+)', data_source_part + year_range_part)
                if match:
                    data_source = match.group(1)
                    year_range = match.group(2)
                
                if len(parts) > 3:
                    version_part = parts[3]
                    version_match = re.search(r'V\d+', version_part)
                    if version_match:
                        version = version_match.group()
            except:
                pass
        
        # 使用提取的信息构建报告文件名
        report_filename = f"report_{year_range}_{data_source}_{version}.html"
    
    # 如果有层次聚类数据，更新标题
    if hierarchical_data:
        n_groups = hierarchical_data.get('n_groups', 50)
        title_base = f'{year_range}年{display_data_source}进一步层次聚类结果一览表-{version}'
    else:
        title_base = f'{year_range}年{display_data_source}主题聚类结果一览表-{version}'
    
    # 生成导航页面内容（删除了数据下载模块）
    index_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{title_base}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background-color: #f4f7f6; }}
        header {{ background: #2c3e50; color: white; padding: 15px 25px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }}
        h1 {{ margin: 0; font-size: 20px; }}
        nav {{ background: #ecf0f1; padding: 10px; display: flex; gap: 10px; border-bottom: 1px solid #ddd; flex-wrap: wrap; }}
        .nav-btn {{ 
            padding: 8px 15px; background: white; border: 1px solid #bdc3c7; border-radius: 4px; 
            cursor: pointer; text-decoration: none; color: #34495e; font-size: 14px; transition: all 0.3s;
            white-space: nowrap;
        }}
        .nav-btn:hover {{ background: #3498db; color: white; border-color: #2980b9; }}
        .nav-btn.active {{ background: #3498db; color: white; }}
        #content-frame {{ flex-grow: 1; border: none; width: 100%; }}
        .param-section {{ background: #f8f9fa; padding: 20px; margin: 20px; border-radius: 8px; border: 1px solid #dee2e6; }}
        .section-title {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <header>
        <h1>{title_base}</h1>
        <span style="font-size: 12px; opacity: 0.8;">版本: {version} | 数据源: {display_data_source} | 年份范围: {year_range}</span>
    </header>
    
    <nav>
        <a class="nav-btn" href="{os.path.basename(report_filename)}" target="chart_frame">✨ 参数说明文档</a>
        <a class="nav-btn" href="barchart.html" target="chart_frame">📈 主题关键词权重图</a>
        <a class="nav-btn" href="hierarchy.html" target="chart_frame">📊 主题层次聚类图</a>
        <a class="nav-btn" href="documents.html" target="chart_frame">📍 主题分布散点图</a>
        <a class="nav-btn" href="topic_overtime_merged.html" target="chart_frame">⌚ 主题时序图</a>
        <a class="nav-btn" href="topics_over_time_value.html" target="chart_frame">⌚ 主题年份矩阵</a>
    </nav>

    <div style="display: flex; height: calc(100vh - 150px);">
        <div style="width: 300px; background: #f8f9fa; padding: 20px; overflow-y: auto; border-right: 1px solid #ddd;">
            <div class="param-section">
                <h3 class="section-title">⚙️ 分析参数</h3>
                <ul style="list-style: none; padding-left: 0;">
                    <li>📅 年份范围: {year_range}</li>
                    <li>📊 数据源: {display_data_source}</li>
                    <li>🏷️ 版本: {version}</li>
                    <li>📈 主题数: 自动优化</li>
                    <li>🔧 嵌入模型: all-mpnet-base-v2</li>
                </ul>
            </div>
        </div>
        
        <iframe name="chart_frame" id="content-frame" src="{os.path.basename(report_filename)}" style="flex-grow: 1;"></iframe>
    </div>

    <script>
        const buttons = document.querySelectorAll('.nav-btn');
        buttons.forEach(btn => {{
            btn.addEventListener('click', function() {{
                buttons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
            }});
        }});
        // 默认高亮第一个按钮（即参数说明页面）
        buttons[0].classList.add('active');
    </script>
</body>
</html>
"""
    
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print(f"导航索引页已生成至: {index_path}")
