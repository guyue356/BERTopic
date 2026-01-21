import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import json
import os
# 从JSON文件加载数据
with open('data\patents_zf.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# # 打印最后两条数据进行检查
# for item in data[-2:]:
#     print(item['year'])
#     #print(item['title'])
#     #print(item['abstract'])
#     print(item['combined_text'])


# 文本
docs = [(item['title']+item['abstract']) for item in data]

# 1. 加载模型（会自动获取对应的 Tokenizer）
model_name = 'all-mpnet-base-v2'

# 1. 统一定义路径
local_model_path = f"./my_models/{model_name}"  # 固定路径


model = SentenceTransformer(local_model_path)

tokenizer = model.tokenizer

# 假设你的专利拼接数据存储在 texts 列表中
# texts = [title*3 + " " + abstract for title, abstract in zip(titles, abstracts)]

def analyze_token_lengths(texts):
    # 2. 计算每个文本的 token 数量
    # 使用 tokenizer.encode 不进行截断，以便统计真实长度
    token_lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    
    # 3. 计算基本统计指标
    max_len = np.max(token_lengths)
    mean_len = np.mean(token_lengths)
    p95 = np.percentile(token_lengths, 95)
    over_limit = sum(1 for l in token_lengths if l > 512)
    percentage_over = (over_limit / len(texts)) * 100

    print(f"--- 统计结果 ---")
    print(f"平均长度: {mean_len:.2f}")
    print(f"最大长度: {max_len}")
    print(f"95% 的数据长度在 {p95:.2f} 以内")
    print(f"超过 512 限制的文本数量: {over_limit} ({percentage_over:.2f}%)")

    # 4. 可视化分布
    plt.figure(figsize=(10, 6))
    sns.histplot(token_lengths, bins=50, kde=True, color='skyblue')
    plt.axvline(512, color='red', linestyle='--', label='512 Token Limit')
    plt.title(f'Token Length Distribution ({model_name})')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# 调用函数
analyze_token_lengths(docs)