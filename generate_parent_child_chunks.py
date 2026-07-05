# generate_parent_child_chunks.py
import os, uuid, re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from generate_summaries import clean_ad_lines

embeddings = DashScopeEmbeddings(model="text-embedding-v4")


def detect_chapter(text: str) -> str:
    """从文本中检测章节标题"""
    match = re.search(r'第[一二三四五六七八九十\d]+章\s*[^\n]*', text)
    if match:
        return match.group().strip()
    return "未知章节"

PDF_FILES = {
    "导游业务统编教材.pdf": "导游业务统编教材",
    "地方导游基础知识统编教材.pdf": "地方导游基础知识统编教材",
    "全国导游基础知识统编教材.pdf": "全国导游基础知识统编教材",
    "政策与法律法规统编教材.pdf": "政策与法律法规统编教材",
}

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120, chunk_overlap=30,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)

parent_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="guide_parent")
child_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="guide_child")

for pdf_file, book_name in PDF_FILES.items():
    print(f"正在处理：{book_name}")
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # 清洗广告
    for doc in docs:
        doc.page_content = clean_ad_lines(doc.page_content)
        doc.metadata["source"] = book_name  # 🔥 关键：标记教材来源

    # 生成并存入父切片
    parent_chunks = parent_splitter.split_documents(docs)
    for chunk in parent_chunks:
        chunk.metadata["parent_id"] = str(uuid.uuid4())
        chunk.metadata["source"] = book_name
        chunk.metadata["chapter"] = detect_chapter(chunk.page_content)
    parent_ids = [chunk.metadata["parent_id"] for chunk in parent_chunks]
    parent_store.add_documents(parent_chunks, ids=parent_ids)

    # 生成并存入子切片（带 parent_id 关联 + 继承 chapter/page）
    for parent in parent_chunks:
        sub_texts = child_splitter.split_text(parent.page_content)
        child_ids = [str(uuid.uuid4()) for _ in sub_texts]
        child_metadatas = [
            {
                "parent_id": parent.metadata["parent_id"],
                "source": book_name,
                "chapter": parent.metadata.get("chapter", "未知章节"),
                "page": parent.metadata.get("page", "未知页"),
            }
            for _ in sub_texts
        ]
        child_store.add_texts(texts=sub_texts, ids=child_ids, metadatas=child_metadatas)

print("四本教材全部处理完成。")