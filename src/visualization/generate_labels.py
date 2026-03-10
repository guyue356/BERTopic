import os
import json
from typing import Dict
from bertopic import BERTopic


SYSTEM_PROMPT = """
你是一名专业的国防专利与技术分析专家，擅长从技术关键词中提炼核心主题方向。

【输入数据格式】
用户将提供一个Python字典：
- 键（key）：主题索引编号（整数，如0, 1, 2...）
- 值（value）：该主题的10个英文关键词字符串，以英文逗号分隔
- 关键词已按重要性降序排列

【核心任务】
为每个主题生成一个中文主题名称，要求：
1. **保守概括**：基于关键词的共同技术内涵，提取"上位技术方向"
2. **权重优先**：主要依据前3-5个高权重关键词，后部关键词仅参考
3. **避免扩展**：不引入新概念，不扩展应用场景，不做过度推断
4. **处理模糊**：若关键词分散，使用更抽象的名称（如"综合技术"）

【主题命名规则】
- 名称应体现国防/军事技术特点
- 使用标准技术术语，避免口语化
- 长度控制在6-15个汉字为宜
- 格式："{序号}. {名称}"，序号从1开始连续编号

【输出要求】
- 仅输出合法的JSON对象，无任何额外文本
- JSON结构：{"主题索引": "序号.主题名称"}
- 主题索引与输入保持一致
- 示例输出：{"0": "1.雷达探测", "1": "2.复合材料制备","2": "3.音频处理与降噪"}

【注意事项】
- 关键词可能存在词形变化（单复数等），理解其核心语义
- 国防领域特有术语应保留专业性
"""


def generate_chinese_labels(topic_model: BERTopic) -> Dict[int, str]:
    """
    使用AI生成中文主题标签
    
    Args:
        topic_model: 已加载的BERTopic模型
        
    Returns:
        格式: {topic_id: "序号.中文名称"}
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(".env")
    API_KEY = os.environ.get('DEEPSEEK_API_KEY')
    
    if not API_KEY:
        raise ValueError("未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置")
    
    all_topics_dict = topic_model.get_topics()
    
    clean_topics_dict = {
        topic_id: ",".join([word for word, score in words_list])
        for topic_id, words_list in all_topics_dict.items()
        if topic_id != -1
    }
    
    print(f"已获取 {len(clean_topics_dict)} 个主题，正在通过AI生成中文标签...")
    
    user_prompt = f"""
请分析以下主題关键词字典，
为每个主题生成对应的中文主题名称，
并返回 JSON 字典：\n{clean_topics_dict}
"""
    
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={'type': 'json_object'},
        stream=False
    )
    
    content = json.loads(response.choices[0].message.content)
    formatted_labels = {int(k): v for k, v in content.items()}
    
    print(f"AI标签生成完成，共 {len(formatted_labels)} 个主题")
    return formatted_labels

def set_chinese_labels(topic_model: BERTopic) -> BERTopic:
    labels = generate_chinese_labels(topic_model)

    # 获取真实 topic 顺序（包含 -1）
    topic_ids = topic_model.get_topic_info()["Topic"].tolist()

    label_list = []

    for topic_id in topic_ids:
        if topic_id == -1:
            label_list.append("噪声主题")
        else:
            label_list.append(labels.get(topic_id, f"Topic {topic_id}"))

    print("BERTopic主题数:", len(topic_ids))
    print("生成标签数:", len(label_list))

    topic_model.set_topic_labels(label_list)

    return topic_model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="为BERTopic模型生成中文标签")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=r"D:\WOS2025\bertopic_abpa_2000-2025_V_pp3",
        help="BERTopic模型路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="标签输出文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    print(f"加载模型: {args.model}")
    topic_model = BERTopic.load(args.model)
    
    topic_model = set_chinese_labels(topic_model)
    #topic_model = set_chinese_labels(topic_model)
    topic_model.save(args.model)
    
    if args.output:
        labels = topic_model.custom_labels_
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"标签已保存至: {args.output}")
    else:
        print("\n生成的中文标签:")
        for topic_id, label in enumerate(topic_model.custom_labels_):
            if topic_id != -1:
                print(f"  {topic_id}: {label}")
