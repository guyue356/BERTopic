# topic_time_table.py

import pandas as pd


def export_topics_over_time_html(
    topics_over_time: pd.DataFrame,
    topic_labels: dict,
    output_html: str = "topics_over_time_value.html",
    remove_outliers: bool = True,
    title: str = "BERTopic 各主题历年文档数变化表"
):
    """
    生成 HTML：各主题（自定义标签）历年文档数变化表
    """

    df = topics_over_time.copy()

    # 1. 移除噪声主题
    if remove_outliers:
        df = df[df["Topic"] != -1]

    # 2. Year × Topic 矩阵
    table = df.pivot_table(
        index="Timestamp",
        columns="Topic",
        values="Frequency",
        fill_value=0
    ).sort_index()
    table = table.astype(int)

    table.index.name = "Year"

    # 3. Topic → 自定义标签
    table.columns = [
        topic_labels.get(topic, f"Topic_{topic}")
        for topic in table.columns
    ]

    # 4. HTML
    html_table = table.to_html(
        border=1,
        classes="topic-table",
        justify="center"
    )

    html_page = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: Arial; padding: 20px; }}
h1 {{ text-align: center; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #999; padding: 6px; text-align: center; }}
th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/xlsx/dist/xlsx.full.min.js"></script>
</head>
<body>

<h1>{title}</h1>
<button onclick="downloadExcel()">📥 下载为 Excel</button>

{html_table}

<script>
function downloadExcel() {{
    const table = document.querySelector(".topic-table");
    const wb = XLSX.utils.table_to_book(table, {{ sheet: "Topics Over Time" }});
    XLSX.writeFile(wb, "topics_over_time.xlsx");
}}
</script>

</body>
</html>
"""


    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_page)

    print(f"[OK] HTML 已生成：{output_html}")

    return table
