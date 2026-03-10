import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from bertopic import BERTopic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.generate_labels import set_chinese_labels
from src.visualization.report_generator import generate_bertopic_report
from src.config.bertopic_config import BERTopicConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_topic_model(model_path: str) -> BERTopic:
    """加载保存的BERTopic模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    topic_model = BERTopic.load(model_path)
    logger.info(f"模型加载成功: {model_path}")
    print(f"✓ 模型加载成功: {model_path}")
    return topic_model


def save_chinese_labels(topic_model: BERTopic, output_path: Optional[str] = None) -> str:
    """保存模型的中文标签到文件"""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "labels.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labels = topic_model.topic_labels_
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    logger.info(f"中文标签已保存至: {output_path}")
    print(f"✓ 中文标签已保存至: {output_path}")
    return output_path


def load_chinese_labels(labels_path: str) -> dict:
    """从文件加载中文标签"""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    logger.info(f"从 {labels_path} 加载了 {len(labels)} 个标签")
    return labels


def apply_chinese_labels(topic_model: BERTopic, labels_path: Optional[str] = None, use_ai: bool = True) -> BERTopic:
    """应用中文标签到模型"""
    if labels_path and os.path.exists(labels_path):
        labels = load_chinese_labels(labels_path)
        topic_model.set_topic_labels(labels)
        print(f"✓ 已从文件加载 {len(labels)} 个中文标签")
    elif use_ai:
        print("\n[Step 1] 使用AI生成中文标签...")
        topic_model = set_chinese_labels(topic_model)
        print("✓ 中文标签设置完成")
    else:
        logger.warning("未提供标签文件且 use_ai=False，将使用默认英文标签")
    
    return topic_model


def extract_model_config(topic_model: BERTopic, model_path: str) -> dict:
    """从模型中提取配置参数"""
    config = BERTopicConfig()
    
    umap_cfg = {}
    if topic_model.umap_model:
        umap_cfg = {
            "n_neighbors": topic_model.umap_model.n_neighbors,
            "n_components": topic_model.umap_model.n_components,
            "min_dist": topic_model.umap_model.min_dist,
            "metric": topic_model.umap_model.metric,
            "random_state": topic_model.umap_model.random_state
        }
    else:
        umap_cfg = config.umap_params
    
    hdbscan_cfg = {}
    if topic_model.hdbscan_model:
        hdbscan_model = topic_model.hdbscan_model
        hdbscan_cfg = {
            "min_samples": getattr(hdbscan_model, 'min_samples', None),
            "metric": getattr(hdbscan_model, 'metric', 'euclidean'),
            "prediction_data": True
        }
    else:
        hdbscan_cfg = config.hdbscan_params
    
    vectorizer_cfg = {}
    if topic_model.vectorizer_model:
        vectorizer_cfg = topic_model.vectorizer_model.get_params()
        vectorizer_cfg.pop('stop_words', None)
    else:
        vectorizer_cfg = config.vectorizer_params
    
    embedding_model = config.embedding_model
    
    return {
        "umap_cfg": umap_cfg,
        "hdbscan_cfg": hdbscan_cfg,
        "vectorizer_cfg": vectorizer_cfg,
        "embedding_model": embedding_model,
        "config": config
    }


def generate_report_from_model(
    model_path: str,
    output_dir: str = None,
    use_ai_labels: bool = True,
    n_groups: int = 50,
    labels_path: str = None,
    save_labels: bool = False,
    verbose: bool = False,
    preview: bool = False
):
    """
    从已保存的BERTopic模型生成可视化报告
    
    Args:
        model_path: 模型保存路径
        output_dir: 报告输出目录（默认与模型同目录）
        use_ai_labels: 是否使用AI生成中文标签
        n_groups: 层次聚类分组数
        labels_path: 中文标签文件路径（可选）
        save_labels: 是否保存中文标签到文件
        verbose: 是否输出详细日志
        preview: 是否在生成后预览报告
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("BERTopic 报告生成工具")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    topic_model = load_topic_model(model_path)
    
    topic_model = apply_chinese_labels(topic_model, labels_path, use_ai_labels)
    
    if save_labels:
        labels_output = os.path.join(
            os.path.dirname(model_path) if output_dir is None else output_dir,
            "chinese_labels.json"
        )
        save_chinese_labels(topic_model, labels_output)
    
    print("\n[Step 2] 提取模型配置参数...")
    config_dict = extract_model_config(topic_model, model_path)
    config = config_dict["config"]
    print("[OK] 配置参数提取完成")
    
    print("\n[Step 3] 生成报告...")
    
    if output_dir is None:
        output_dir = os.path.join(
            config.output_dir
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    safe_data_source = config.data_source.replace("专利", "Patent").replace("样本", "Sample").replace("测试", "Test")
    report_name = os.path.join(output_dir, f"report_{config.year_range}_{safe_data_source}_{config.version}.html")
    
    topic_info = topic_model.get_topic_info()
    topic_count = len(topic_info[topic_info['Topic'] != -1])
    
    logger.info(f"模型共包含 {topic_count} 个主题")
    
    hierarchical_data = {
        "n_groups": n_groups,
        "all_topics": list(range(topic_count))
    }
    
    generate_bertopic_report(
        umap_cfg=config_dict["umap_cfg"],
        HDBSCAN_cfg=config_dict["hdbscan_cfg"],
        vectorizer_cfg=config_dict["vectorizer_cfg"],
        history=None,
        best_size=topic_count,
        model_name=config_dict["embedding_model"],
        output_path=report_name,
        hierarchical_data=hierarchical_data
    )
    
    print(f"\n[OK] 报告已生成: {report_name}")
    print("=" * 60)
    print(f"报告生成完成! 总用时: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if preview:
        open_report(report_name)
    
    return topic_model, report_name


def open_report(report_path: str):
    """在浏览器中打开报告"""
    import webbrowser
    report_path = os.path.abspath(report_path)
    url = f"file:///{report_path.replace(os.sep, '/')}"
    print(f"\n[预览] 正在打开报告: {url}")
    webbrowser.open(url)


def generate_batch_reports(
    model_dir: str,
    output_base_dir: str = None,
    use_ai_labels: bool = False,
    n_groups: int = 50,
    preview: bool = False
):
    """
    批量生成多个模型的报告
    
    Args:
        model_dir: 包含多个BERTopic模型的目录
        output_base_dir: 报告输出基础目录
        use_ai_labels: 是否使用AI生成中文标签
        n_groups: 层次聚类分组数
        preview: 是否在生成后预览每个报告
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    model_files = []
    for f in os.listdir(model_dir):
        if f.startswith('bertopic_') and (f.endswith('.pt') or f.endswith('.pkl')):
            model_files.append(os.path.join(model_dir, f))
    
    if not model_files:
        logger.warning(f"在目录 {model_dir} 中未找到BERTopic模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型文件，开始批量生成报告...")
    
    results = []
    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] 处理模型: {os.path.basename(model_path)}")
        try:
            _, report_path = generate_report_from_model(
                model_path=model_path,
                output_dir=output_base_dir,
                use_ai_labels=use_ai_labels,
                n_groups=n_groups,
                preview=preview
            )
            results.append({"model": model_path, "report": report_path, "status": "success"})
        except Exception as e:
            logger.error(f"处理模型 {model_path} 时出错: {e}")
            results.append({"model": model_path, "error": str(e), "status": "failed"})
    
    print("\n" + "=" * 60)
    print("批量生成完成!")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="从BERTopic模型生成可视化报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python -m src.report_main -i results/topic_models/bertopic_abpa_2000-2025_V_pp3
  
  # 不使用AI标签，直接生成报告
  python -m src.report_main -i results/topic_models/bertopic_abpa_2000-2025_V_pp3 --no-ai-labels
  
  # 指定输出目录
  python -m src.report_main -i results/topic_models/bertopic_abpa_2000-2025_V_pp3 -o results/reports
  
  # 加载已有的中文标签文件
  python -m src.report_main -i results/topic_models/bertopic_abpa_2000-2025_V_pp3 --labels-path results/labels.json
  
  # 生成报告后自动预览
  python -m src.report_main -i results/topic_models/bertopic_abpa_2000-2025_V_pp3 --preview
  
  # 批量生成报告
  python -m src.report_main --batch -i results/topic_models/
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=r"D:\WOS2025\bertopic_abpa_2000-2025_V_pp3",
        #default="results/topic_models/bertopic_abpa_2000-2025_V_pp3",
        help="BERTopic模型路径或包含模型的目录(批量模式)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="报告输出目录"
    )
    parser.add_argument(
        "--no-ai-labels",
        action="store_true",
        help="不使用AI生成中文标签"
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=50,
        help="层次聚类分组数"
    )
    parser.add_argument(
        "--labels-path", "-l",
        type=str,
        default=None,
        help="中文标签文件路径（可选，用于加载已有的标签）"
    )
    parser.add_argument(
        "--save-labels", "-s",
        action="store_true",
        help="生成报告后保存中文标签到文件"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="生成报告后在浏览器中预览"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细日志信息"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批量模式：处理输入目录下的所有模型"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        generate_batch_reports(
            model_dir=args.input,
            output_base_dir=args.output,
            use_ai_labels=not args.no_ai_labels,
            n_groups=args.n_groups,
            preview=args.preview
        )
    else:
        generate_report_from_model(
            model_path=args.input,
            output_dir=args.output,
            use_ai_labels=not args.no_ai_labels,
            n_groups=args.n_groups,
            labels_path=args.labels_path,
            save_labels=args.save_labels,
            verbose=args.verbose,
            preview=args.preview
        )


if __name__ == "__main__":
    main()
