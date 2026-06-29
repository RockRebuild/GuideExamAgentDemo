# generate_parent_child_chunks.py
import re

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os, uuid
from dotenv import load_dotenv
load_dotenv()
PDF_PATH = "政策与法律法规统编教材.pdf"
CHROMA_PERSIST_DIR = "./chroma_db"
PARENT_COLLECTION = "guide_parent"    # 父切片库（大粒度，供 LLM 阅读）
CHILD_COLLECTION = "guide_child"      # 子切片库（小粒度，供检索匹配）

embeddings = DashScopeEmbeddings(model="text-embedding-v3")
loader = PyPDFLoader(PDF_PATH)
# loader = PyMuPDFLoader(PDF_PATH)

docs = loader.load()
def clean_ad_lines(text: str) -> str:
    # 要匹配的广告模式
    ad_patterns = [
        r"微信公众号：daoyoukaoshizhongxin",
        r"微信客服：daoyoukaoshipeixun",
        r"QQ:17059435",
        r"微信公众号和客服不要添加！是机构（口碑不好）！！！不是我！！！",
        # 也可以用更通用的规则：去掉含“微信客服”、“微信公众号”且长度较短的行
        r"^.*(?:微信公众号|微信客服|QQ:|扫码|加好友).*$"
    ]
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # 如果该行匹配任何一个广告模式，就跳过
        if any(re.search(p, line) for p in ad_patterns):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# total_chars = sum(len(doc.page_content) for doc in docs)
# print(f"总字符数：{total_chars}")
# print("前 200 字符示例：", docs[0].page_content[:200] if docs else "无内容")


parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)

docs = loader.load()
for doc in docs:
    doc.page_content = clean_ad_lines(doc.page_content)

parent_chunks = parent_splitter.split_documents(docs)
for chunk in parent_chunks:
    chunk.metadata["parent_id"] = str(uuid.uuid4())

# 2. 存父切片到 Chroma
parent_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=PARENT_COLLECTION,
)
parent_ids = [chunk.metadata["parent_id"] for chunk in parent_chunks]
parent_store.add_documents(parent_chunks, ids=parent_ids)

# 3. 从每个父切片生成子切片（小块），并记录父切片 ID
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
child_chunks = []
child_ids = []
child_metadatas = []
for parent in parent_chunks:
    sub_texts = child_splitter.split_text(parent.page_content)
    for sub in sub_texts:
        child_chunks.append(sub)
        child_ids.append(str(uuid.uuid4()))
        child_metadatas.append({"parent_id": parent.metadata["parent_id"]})

# 4. 存子切片到 Chroma
child_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=CHILD_COLLECTION,
)
child_store.add_texts(texts=child_chunks, ids=child_ids, metadatas=child_metadatas)
print(f"父切片：{len(parent_chunks)}，子切片：{len(child_chunks)}")