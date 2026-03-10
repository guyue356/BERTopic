import os
import pandas as pd
from umap import UMAP
from typing import Optional, List
from src.visualization.topicvalue import export_topics_over_time_html


class ChartGenerator:
    def __init__(self, topic_model, config):
        self.topic_model = topic_model
        self.config = config
    
    def generate_all_charts(self, docs: List[str], hierarchical_topics, embeddings, timestamps: List[int], output_dir: str, n_groups: int = 50):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取主题标签
        topic_info = self.topic_model.get_topic_info()
        formatted_labels = []
        for idx, row in topic_info.iterrows():
            if row['Topic'] != -1:
                keywords = [word for word, _ in self.topic_model.get_topic(row['Topic'])]
                label = f"Topic {row['Topic']}: {', '.join(keywords[:3])}"
                formatted_labels.append(label)
        
        title_base = f'{self.config.year_range}年{self.config.data_source}数据Bertopic主题聚类结果一览表-{self.config.version}'
        
        # --- 1. 生成图表 (按照你想要的顺序) ---
        #-------------------------------------------------------------------------
        # [图 A] 主题关键词条形图 - 存为 barchart.html
        fig_barchart = self.topic_model.visualize_barchart(
            top_n_topics=len(formatted_labels),
            custom_labels=True,
            n_words=10,
            height=400,
            title=f"{self.config.year_range}年{self.config.data_source}各主题关键词的c-TF-IDF权重得分条形图",
        )
        fig_barchart.update_layout(
            width=1500,
            font=dict(family="KaiTi", size=16),
            title_font=dict(family="KaiTi", size=36, color="black", weight="bold")
        )
        path_barchart = os.path.join(output_dir, "barchart.html")
        fig_barchart.write_html(path_barchart)
        #-------------------------------------------------------------------------
        # [图 B] 层次聚类图 - 存为 hierarchy.html
        fig_hierarchy = self.topic_model.visualize_hierarchy(
            hierarchical_topics=hierarchical_topics,
            custom_labels=True,
            title=f"{self.config.year_range}年{self.config.data_source}主题层次聚类图",
            height=800
        )
        fig_hierarchy.update_layout(
            title_x=0.5,
            width=1500,
            font=dict(family="KaiTi", size=16),
            title_font=dict(family="KaiTi", size=30, color="black", weight="bold")
        )
        path_hierarchy = os.path.join(output_dir, "hierarchy.html")
        fig_hierarchy.write_html(path_hierarchy)
        #-------------------------------------------------------------------------
        # [图 C] 文档分布散点图 - 存为 documents.html
        reduced_embeddings = UMAP(
            n_neighbors=self.config.umap_params["n_neighbors"],
            n_components=2,
            min_dist=0.0,
            metric='cosine'
        ).fit_transform(embeddings)
        
        fig_documents = self.topic_model.visualize_documents(
            docs=[doc[:150] + "..." for doc in docs],
            reduced_embeddings=reduced_embeddings,
            custom_labels=True,
            hide_document_hover=False
        )
        # 更新散点图样式
        fig_documents.update_layout(
            title=f"{self.config.year_range}年{self.config.data_source}主题分布图",
            title_x=0.5,
            width=1500,
            height=1200,
            margin=dict(l=50, r=250, t=100, b=50),
            font=dict(family="KaiTi", size=16, color="black"),
            title_font=dict(family="KaiTi", size=30, color="black", weight="bold")
        )
        path_documents = os.path.join(output_dir, "documents.html")
        fig_documents.write_html(path_documents)
        #-------------------------------------------------------------------------
        # [图 D] 主题时序图 - 存为 topic_overtime_merged.html
        topics_over_time = self.topic_model.topics_over_time(docs, timestamps, global_tuning=False, evolution_tuning=False)
        fig_topic_time = self.topic_model.visualize_topics_over_time(
            topics_over_time,
            custom_labels=True)
        # 更新散点图样式
        fig_topic_time.update_layout(
            title=f"{self.config.year_range}年{self.config.data_source}主题时序图",
            title_x=0.5,
            width=1500,
            height=800,
            font=dict(family="KaiTi", size=16, color="black"),
            title_font=dict(family="KaiTi", size=30, color="black", weight="bold")
        )
        path_topic_time = os.path.join(output_dir, "topic_overtime_merged.html")
        fig_topic_time.write_html(path_topic_time)

        #-------------------------------------------------------------------------
        # [图E] 主题年份矩阵图 - 存为 topics_over_time_value.html
        path_topic_time_value = os.path.join(output_dir, "topics_over_time_value.html")
        # 将主题标签列表转换为字典格式
        topic_labels_dict = {i: label for i, label in enumerate(formatted_labels)}
        try:
            from topicvalue import export_topics_over_time_html
            export_topics_over_time_html(
                topics_over_time=topics_over_time,
                topic_labels=topic_labels_dict,
                output_html=path_topic_time_value,
                remove_outliers=False
            )
        except ImportError:
            print("注意: topicvalue模块未安装，跳过主题年份矩阵图生成")
        except Exception as e:
            print(f"生成主题年份矩阵图时出错: {e}")
        
        print(f"🎉 结果可视化已完成！请打开文件夹: {output_dir}查看")
        
        # 返回主题信息用于报告生成
        return {
            "formatted_labels": formatted_labels,
            "topics_over_time": topics_over_time,
            "n_groups": n_groups,
            "all_topics": list(range(len(formatted_labels)))
        }
    
    def generate_barchart(self, top_n_topics: int = 50, n_words: int = 10):
        return self.topic_model.visualize_barchart(
            top_n_topics=top_n_topics,
            custom_labels=True,
            n_words=n_words,
            height=400
        )
    
    def generate_documents_plot(self, docs, embeddings):
        reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        return self.topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, custom_labels=True)
