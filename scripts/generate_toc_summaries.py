# generate_toc_summaries.py
import os
import re
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ================= 四本教材配置 =================
BOOKS = [
    {
        "pdf_path": "导游业务统编教材.pdf",
        "book_name": "导游业务统编教材",
    },
    {
        "pdf_path": "地方导游基础知识统编教材.pdf",
        "book_name": "地方导游基础知识统编教材",
    },
    {
        "pdf_path": "全国导游基础知识统编教材.pdf",
        "book_name": "全国导游基础知识统编教材",
    },
    {
        "pdf_path": "政策与法律法规统编教材.pdf",
        "book_name": "政策与法律法规统编教材",
    },
]

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "guide_summary"
embeddings = DashScopeEmbeddings(model="text-embedding-v4")
summary_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)


def extract_toc_from_pdf(pdf_path: str, max_pages: int = 5) -> dict:
    """
    从 PDF 的前几页提取章节目录。
    返回格式：{"第一章 xxx": "第一节 xxx\n第二节 xxx\n...", ...}
    """
    reader = PdfReader(pdf_path)
    toc_text = ""

    # 只读取前 max_pages 页（目录通常在前几页）
    for page in reader.pages[:max_pages]:
        text = page.extract_text()
        if text:
            toc_text += text + "\n"

    # 按"第X章"拆分
    chapter_pattern = re.compile(r'(第[一二三四五六七八九十\d]+章[^\n]*)')
    parts = chapter_pattern.split(toc_text)

    toc_dict = {}
    current_chapter = None
    current_content = []

    for part in parts:
        if chapter_pattern.match(part):
            # 保存上一章
            if current_chapter and current_content:
                toc_dict[current_chapter.strip()] = "\n".join(current_content).strip()
            # 开始新章
            current_chapter = part
            current_content = []
        else:
            if current_chapter:
                # 只保留看起来像目录的行（包含"第X节"、"一、"等）
                for line in part.split('\n'):
                    line = line.strip()
                    if re.search(
                            r'第[一二三四五六七八九十\d]+节|^[一二三四五六七八九十]、|^\d+\.|^（[一二三四五六七八九十]）',
                            line):
                        current_content.append(line)

    # 保存最后一章
    if current_chapter and current_content:
        toc_dict[current_chapter.strip()] = "\n".join(current_content).strip()

    return toc_dict


def save_toc_to_chroma(toc_dict: dict, book_name: str, store: Chroma):
    """将目录存入 Chroma"""
    texts, metadatas, ids = [], [], []
    for i, (chapter_title, toc_content) in enumerate(toc_dict.items()):
        full_text = f"【{book_name}】{chapter_title}\n目录结构：\n{toc_content}"
        texts.append(full_text)
        metadatas.append({
            "book": book_name,
            "chapter": chapter_title,
            "type": "toc",
            "full_key": f"{book_name}_{chapter_title}_目录"
        })
        ids.append(f"{book_name}_toc_{i}")

    if texts:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"✅ 已存入 {len(texts)} 个章节目录到集合 '{COLLECTION_NAME}'。")
    else:
        print("⚠️ 未提取到任何目录，请检查 PDF 前几页的格式。")


# ================= 主流程 =================
if __name__ == "__main__":
    for book in BOOKS:
        pdf_path = book["pdf_path"]
        book_name = book["book_name"]
        print(f"\n📖 正在提取《{book_name}》的目录...")
        try:
            toc = extract_toc_from_pdf(pdf_path, max_pages=5)
            print(f"📚 检测到 {len(toc)} 个章节目录。")
            save_toc_to_chroma(toc, book_name, summary_store)
            print(f"🎉 《{book_name}》目录提取完成！")
        except Exception as e:
            print(f"❌ 《{book_name}》目录提取失败：{str(e)}")