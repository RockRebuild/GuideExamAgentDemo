import os
import sys

from tools import get_embeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        persist_directory="../chroma_db",
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )# 获取摘要库

summary_store = get_vectorstore("guide_summary")

# 获取所有数据
all_data = summary_store.get()

print(f"摘要库文档总数: {len(all_data.get('ids', []))}")

if all_data.get('ids'):
    import random
    indices = random.sample(range(len(all_data['ids'])), min(5, len(all_data['ids'])))
    for i in indices:
        print("=" * 50)
        print(f"章节: {all_data['metadatas'][i].get('full_key', '未知')}")
        print(f"摘要: {all_data['documents'][i]}")
        print("=" * 50)
else:
    print("⚠️ 摘要库为空！请检查 generate_summaries_all.py 是否成功运行。")