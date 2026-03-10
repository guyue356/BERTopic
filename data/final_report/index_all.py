import os
import re

def generate_experiment_portal(filename="Experiment_Portal.html"):
    # 查找所有以 BERTopic_Results_ 开头的文件夹
    base_dir = "."
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(f) and f.startswith("BERTopic_Results_")]
    
    portal_data = []

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        # 查找文件夹下所有以 index 开头的 .html 文件
        indices = [f for f in os.listdir(folder_path) if f.startswith("index") and f.endswith(".html")]
        
        if indices:
            portal_data.append({
                "folder": folder,
                "files": sorted(indices)
            })

    if not portal_data:
        print("❌ 未找到匹配的文件夹或 index 文件，请检查路径。")
        return

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BERTopic 主题聚类实验导航</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <style>
            body {{ background-color: #f8f9fa; padding: 40px 0; }}
            .experiment-card {{ transition: transform 0.2s; margin-bottom: 20px; border-left: 5px solid #0d6efd; }}
            .experiment-card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
            .folder-name {{ color: #2c3e50; font-weight: bold; }}
            .btn-link-custom {{ text-decoration: none; display: block; padding: 5px 0; color: #495057; }}
            .btn-link-custom:hover {{ color: #0d6efd; text-decoration: underline; }}
            .badge-count {{ font-size: 0.8rem; vertical-align: middle; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row mb-5">
                <div class="col-12 text-center">
                    <h1 class="display-5 fw-bold">🧪 BERTopic 主题聚类实验总控台</h1>
                    <p class="lead text-muted">自动检索当前目录下的所有实验结果索引</p>
                </div>
            </div>
            <div class="row">
    """

    for item in portal_data:
        file_links = ""
        for file in item['files']:
            # 构建相对路径
            rel_path = f"{item['folder']}/{file}"
            file_links += f'<a href="{rel_path}" target="_blank" class="btn-link-custom">📄 {file}</a>'

        html_template += f"""
                <div class="col-md-6 col-lg-4">
                    <div class="card experiment-card h-100">
                        <div class="card-body">
                            <h5 class="card-title folder-name text-truncate" title="{item['folder']}">
                                📁 {item['folder'].replace('BERTopic_Results_', '')}
                            </h5>
                            <hr>
                            <div class="file-list">
                                {file_links}
                            </div>
                        </div>
                        <div class="card-footer bg-transparent border-0">
                            <span class="badge bg-secondary badge-count">{len(item['files'])} 个索引文件</span>
                        </div>
                    </div>
                </div>
        """

    html_template += """
            </div>
            
            <footer class="mt-5 text-center text-muted">
                <small>生成的入口文件由脚本自动维护 • 生成时间: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """</small>
            </footer>
        </div>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"✅ 门户网页已更新: {filename}")

import time
if __name__ == "__main__":
    generate_experiment_portal()