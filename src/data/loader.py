from typing import Tuple, List
import json
import re


class DataLoader:
    def __init__(self, file_path: str, text_field: str = "combined_text", timestamp_field: str = "year"):
        self.file_path = file_path
        self.text_field = text_field
        self.timestamp_field = timestamp_field
    
    def load(self) -> Tuple[List[str], List[int]]:
        if self.file_path.endswith('.jsonl'):
            data = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if 'text' in item and 'year' in item:
                            data.append(item)
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        docs = [item[self.text_field] for item in data]
        timestamps = [item[self.timestamp_field] for item in data]
        
        return docs, timestamps
    
    def validate(self, docs: List[str], timestamps: List[int]) -> bool:
        if len(docs) != len(timestamps):
            return False
        return True


# class TextPreprocessor:
#     @staticmethod
#     def preprocess_texts(texts: List[str]) -> List[str]:
#         cleaned_list = []
#         for t in texts:
#             t = re.sub(r"[\x00-\x1F\x7F]", " ", t)
#             t = re.sub(r"[‐-–—−]", "-", t)
#             t = re.sub(r"\s+", " ", t).strip()
#             cleaned_list.append(t)
#         return cleaned_list
    
#     @staticmethod
#     def clean_special_chars(text: str) -> str:
#         text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
#         text = re.sub(r"[‐-–—−]", "-", text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text
    
class TextPreprocessor:
    @staticmethod
    def clean_special_chars(text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # 1. 移除真正的控制字符，但保留 \n (换行, \x0A) 和 \t (制表符, \x09)
        # 范围调整：\x00-\x08 (0-8), \x0B-\x0C (11-12), \x0E-\x1F (14-31), \x7F (127)
        # 显式排除了 \x09 (\t) 和 \x0A (\n)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
        
        # 2. 统一破折号 (这一步可以保留)
        text = re.sub(r"[‐-–—−]", "-", text)
        
        # 3. 规范化空白字符
        # 将连续的普通空格/制表符压缩为一个空格，但保留换行符作为段落分隔
        # 注意：这里不把 \n 替换为空格，而是保留它们，或者视情况将 \n+ 替换为单个 \n
        text = re.sub(r"[^\S\n]+", " ", text) # 匹配除换行外的所有空白字符并压缩
        
        # 可选：如果希望将多个连续换行压缩为一个，防止空行过多
        text = re.sub(r"\n\s*\n", "\n", text)
        
        return text.strip()

    @staticmethod
    def preprocess_texts(texts: List[str]) -> List[str]:
        return [TextPreprocessor.clean_special_chars(t) for t in texts]
