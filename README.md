# BERTopic 使用说明

本文档为仓库中 [main_bert_topic.ipynb](main_bert_topic.ipynb) 的配套说明，包含环境依赖、数据要求、关键函数说明、运行流程、结果保存与常见故障排查。适合以文献/专利文本为输入执行 BERTopic 主题建模流程的用户参考。

**目录**

- **概述**: 项目目的与输入输出
- **文件说明**: 关键文件与路径
- **环境与依赖**: Python 包与安装命令
- **数据格式**: JSON 字段说明与示例
- **主要流程**: 从数据到主题的步骤与关键函数
- **参数与模型配置**: Embedding / UMAP / HDBSCAN / Vectorizer 设置说明
- **可视化与保存**: 生成的文件与示例命令
- **常见问题与故障排查**
- **后续建议**

**概述**

该仓库通过 `BERTopic` 对文献/专利类文本进行主题建模：

- 输入：包含 `combined_text` 字段的 JSON 文件（每条记录包含 `year`、`title`、`abstract`、`keywords`、`combined_text`）。
- 输出：主题模型（可保存）、每文档的主题归属 CSV、若干交互式可视化 HTML（主题词柱状图、层次聚类树、时序主题图等）。

**文件说明**

- **主 Notebook**: [main_bert_topic.ipynb](main_bert_topic.ipynb) —— 包含完整的预处理、模型构建、训练、可视化及导出流程。
- **停用词**: [stopwords.json](stopwords.json) —— Notebook 中通过 `load_stopwords()` 加载并传入 `CountVectorizer(stop_words=...)`。
- **输出示例**: `patents_zf_Clustering_detailed_result_V3.csv`、`topic_words_merged_scoresV3.html`、`hierarchical_topics_V3.html`、`topic_overtime_mergedV3.html` 等（文件名会根据 Notebook 中的 `version` 变量变动）。

**环境与依赖**

建议使用 Python 3.11 的虚拟环境。主要依赖：

```
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas plotly torch torchvision
```

可选（若离线使用自备 embedding 模型）：将 `SentenceTransformer` 模型下载到 `./my_models/<model-name>` 并在 Notebook 中用本地路径加载。

**数据格式**

Notebook 假设输入文件为 JSON 列表，每条记录至少有：

- `year` (int / str)：年份或时间戳，用于时序主题分析（`topics_over_time`）。
- `title` (str)：文献/专利标题。
- `abstract` (str)：摘要。
- `keywords` (list 或 str)：作者关键词（可选）。
- `combined_text` (str)：将 `title`、`abstract`、`keywords` 按规则拼接后的文本，作为 `docs` 列表传入模型。

示例（伪 JSON）:

```
{
  "year": 2020,
  "title": "示例标题",
  "abstract": "摘要内容...",
  "keywords": ["A","B"],
  "combined_text": "示例标题。摘要内容... 关键词：A; B"
}
```

**主要流程（Notebook 关键步骤说明）**

1. 加载数据并构造 `docs = [item['combined_text'] for item in data]`。
2. 配置并加载嵌入模型（`SentenceTransformer`）：Notebook 中提供了多种推荐模型名称（如 `all-MiniLM-L6-v2`, `all-mpnet-base-v2` 等），也支持离线本地加载。
3. 计算/加载 embeddings：使用 `compute_and_save_embeddings(docs, save_path)`，函数会检查已有的 pickle 文件并复用，避免重复计算。
4. 配置降维与聚类模型：使用 `UMAP(**params)` 与 `HDBSCAN(**params)`，参数在 Notebook 中有 `model_config` 集中定义。
5. 创建 `BERTopic`：传入 `embedding_model`、`vectorizer_model`（内置停用词）、`umap_model`、`hdbscan_model`。
6. 训练模型：

```
topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
```

7. 导出文档级别主题信息并合并原始字段：通过 `topic_model.get_document_info(docs)`，然后将 `title`、`year`、`abstract` 等插入 DataFrame 并保存为 CSV。
8. 可视化：使用 `topic_model.visualize_barchart()`、`visualize_topics()`、`visualize_documents()`、`visualize_hierarchy()`、`visualize_topics_over_time()` 等函数并保存 HTML。
9. 层次聚类与合并主题：Notebook 中演示如何基于 `hierarchical_topics` 生成 linkage 矩阵，使用 `topic_model.merge_topics(docs, merge_list)` 进行合并。

**关键函数/变量说明**

- `load_stopwords(file_path)`：从 `stopwords.json` 中加载停用词并传给 `CountVectorizer`。
- `compute_and_save_embeddings(docs, save_path)`：计算或加载已保存的 embeddings（pickle）。
- `model_config`：集中管理 `UMAP` 与 `HDBSCAN` 的参数（便于调参）。
- `topic_model.save(path)` / `BERTopic.load(path)`：保存与加载模型实例。

**Embedding 模型建议**

- 轻量平衡：`all-MiniLM-L6-v2`（速度快、良好效果）
- 更高质量：`all-mpnet-base-v2`（维度 768，效果更好但占用更多显存）
- 多语言场景：`paraphrase-multilingual-MiniLM-L12-v2`

如果你需要离线加载模型，请把模型目录放在 `./my_models/<model-name>` 并在 Notebook 中以该路径初始化 `SentenceTransformer`。

**UMAP / HDBSCAN 参数说明（Notebook 中默认/推荐）**

- UMAP 常用：`n_neighbors`（局部/全局平衡，推荐 15-50），`n_components`（BERTopic 推荐 5 以保留信息，展示时用 2）、`min_dist`（控制嵌入紧密度），`metric`（文本常用 `cosine`）。
- HDBSCAN 常用：`min_cluster_size`（最小簇大小，决定聚类粒度），`min_samples`（对离群点敏感度），`prediction_data=True`（便于后续预测与转换）。

**可视化与输出文件**

- 文档主题 CSV：`patents_zf_Clustering_detailed_result_V3.csv`（示例）
- 主题词柱状图 HTML：`topic_words_merged_scores{version}.html`
- 层次主题交互 HTML：`hierarchical_topics_{version}.html`
- 时序主题 HTML：`topic_overtime_merged{version}.html`

示例保存代码：

```
fig = topic_model.visualize_barchart(...)
fig.write_html('topic_words.html')

topic_docs = topic_model.get_document_info(docs)
topic_docs.to_csv('patents_zf_Clustering_detailed_result_V3.csv')
```

**常见问题与故障排查**

- 若无法下载模型或访问 Hugging Face：Notebook 中设置了 `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` 可作为国内镜像；更稳妥的做法是离线下载模型到本地并用本地路径加载。
- GPU 未被识别：检查 `torch.cuda.is_available()`，并打印 `torch.__version__` 与 `torch.cuda.is_available()` 来确认。若无 GPU，模型会在 CPU 上运行但较慢。
- embeddings 文件过大或重复计算：请使用 `compute_and_save_embeddings()` 的 `save_path` 参数保存/加载 pickle 以避免重复计算。
- HDBSCAN 报错或聚类过少/过多：调节 `min_cluster_size` 和 `min_samples`，并检查 UMAP 的 `n_components` 与 `n_neighbors` 是否合理。

**后续建议**

- 数据预处理：对 `combined_text` 做基本清洗（去除冗余符号、处理英文大小写、中文分词如需）可提升主题质量。
- 超参搜索：对 `UMAP.n_neighbors`、`UMAP.min_dist`、`HDBSCAN.min_cluster_size` 做网格搜索以获得业务上稳定且可解释的主题数。
- 批量化与重现性：将配置集中到 `model_config` 并把 `random_state` 固定，便于复现实验。

如果你希望，我可以：

- 将该 README 放到其他目录或补充更详尽的运行示例（含命令行）；
- 帮你一键生成 `requirements.txt` 或创建一个小的运行脚本来自动化 Notebook 的关键步骤。

---

文件位置： [README.md](README.md)

# BERTopic程序使用示例

## 需要运行的主程序：

1.main_bert_topic.ipynb（用于获取全量的主题聚类模型）

2.Subset_clustering.ipynb（子集主题聚类程序）

3.index_all.py（实验总控台更新程序）

## 使用步骤详解

### 1.main_bert_topic.ipynb（用于获取全量的主题聚类模型）

#### 1.1更新配置文件

![1769587670487](image/README/1769587670487.png)

💡词嵌入结果的存放地址，第一次跑可以设置在 `results\embedding_results\`文件夹下，命名一定要方便区分哪一次数据得出来的结果。这是最耗时跑出来的结果，第一次跑完后，只要不换原始数据这里就不用变，后续会自动加载嵌入结果，节省时间；

换数据源跑新的嵌入时一定要记得换路径！！！

其他参数如UMAP、HDBSCAN和vectorizer_model可以不变，聚类效果非常不好再尝试动他们。

##### 1.2配置好后可以直接全部全部运行，环境配好后的理想情况下全能跑通

该程序最终会生成两个html网页文件记录结果

![1769588853242](image/README/1769588853242.png)

如上图所示，一个是2000-2025年层次聚类之前的实验结果，一个是层次合并之后的实验结果

每个文件夹中有如下七个网页文件，分别对应实验参数报告、聚类结果可视化等文件；

![1769589804265](image/README/1769589804265.png)

进入index索引文件，可以在一个网页里查看本次实验跑的所有结果，以及最终的主题-频次矩阵

![1769590021202](image/README/1769590021202.png)

![1769590039697](image/README/1769590039697.png)

### 2.Subset_clustering.ipynb（子集主题聚类程序）

该脚本主要承担每五年再跑一次结果的任务，运行前提：成功跑完上面的main_bert_topic.ipynb

#### 更新配置文件

![1769590825899](image/README/1769590825899.png)

如上图所示

①每一份数据源需要分别更改年份为（2000-2005、2006-2010、2011-2015、2015-2020、2021-2025）各跑一次；

②同一数据源的data_source和词向量嵌入结果的路径以及原始数据文件要保持一致

设置好后直接运行，顺利的话每次运行都和前面一样生成一个结果文件夹。

![1769592097828](image/README/1769592097828.png)

### 3.index_all.py

![1769592299749](image/README/1769592299749.png)

实验总控台更新程序，跑完后可以运行该脚本，自动更新实验总控台，方便查看当前路径下的每一次的实验结果。
