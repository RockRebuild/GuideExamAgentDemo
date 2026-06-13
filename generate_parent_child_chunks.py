# generate_parent_child_chunks.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os, uuid
from dotenv import load_dotenv
load_dotenv()
PDF_PATH = "全国导游人员资格统一考试模拟试题汇编.pdf"
CHROMA_PERSIST_DIR = "./chroma_db"
PARENT_COLLECTION = "guide_parent"    # 父切片库（大粒度，供 LLM 阅读）
CHILD_COLLECTION = "guide_child"      # 子切片库（小粒度，供检索匹配）

embeddings = DashScopeEmbeddings(model="text-embedding-v3")

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
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