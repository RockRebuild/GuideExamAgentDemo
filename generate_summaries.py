# generate_summaries.py

import os
import re
from pypdf import PdfReader
from langchain_openai import ChatOpenAI  # 或 ChatDeepSeek，取决于你用的模型
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings  # 或你的嵌入模型

# ================= 配置区 =================
PDF_PATH = "全国导游人员资格统一考试模拟试题汇编.pdf"
COLLECTION_NAME = "guide_summary"          # 摘要库
CHROMA_PERSIST_DIR = "./chroma_db"
from dotenv import load_dotenv
load_dotenv()
# 用于生成摘要的 LLM（建议用便宜的模型，如 DeepSeek Flash 或阿里云 Qwen）
SUMMARY_MODEL = ChatOpenAI(
    model="deepseek-chat",           # 或 "gpt-4o-mini" 等
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0
)

# ================= 1. 提取章节文本 =================
def extract_chapter_texts(pdf_path: str) -> dict:
    """从 PDF 中按章节拆分文本，返回 {章节标题: 章节全文}"""
    reader = PdfReader(pdf_path)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

    # 按章节标题拆分（根据你的实际格式调整正则）
    # 假设章节标题形式为 "第一章 xxx" 或 "第1章 xxx"
    chapter_pattern = r'(第[一二三四五六七八九十\d]+章.*?)(?=第[一二三四五六七八九十\d]+章|$)'
    chapters = {}
    for match in re.finditer(chapter_pattern, full_text, re.DOTALL):
        title = match.group(1).split('\n')[0].strip()  # 取第一行作为标题
        content = match.group(1)  # 该章全部内容
        chapters[title] = content
    return chapters

# ================= 2. 调用 LLM 生成摘要 =================
def generate_summary(chapter_title: str, chapter_text: str) -> str:
    """让 LLM 为一章内容生成简短摘要。"""
    prompt = f"""请用一两句话概括以下章节的核心内容，只返回摘要文本，不要多余解释。

章节标题：{chapter_title}

章节内容：
{chapter_text[:3000]}  # 限制长度，避免 token 超限

摘要："""
    response = SUMMARY_MODEL.invoke(prompt)
    return response.content.strip()

# ================= 3. 存入 Chroma =================
import re
from pypdf import PdfReader

import re
from pypdf import PdfReader

def extract_chapter_texts(pdf_path: str, default_book: str = "模拟试题汇编",
                          start_page: int = 1, end_page: int = None) -> dict:
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if start_page < 1:
        start_page = 1
    if end_page is None or end_page > total_pages:
        end_page = total_pages
    selected_pages = reader.pages[start_page - 1 : end_page]
    full_text = "\n".join([page.extract_text() or "" for page in selected_pages])

    chapters = {}
    current_book = default_book
    current_chapter = ""
    current_section = ""
    content_buffer = []
    in_chapter = False

    # 书名行标识：只要包含“模拟试题汇编”
    book_pattern = re.compile(r'模拟试题汇编')
    # 章/节模式
    chapter_pattern = re.compile(r'第[一二三四五六七八九十\d]+章')
    section_pattern = re.compile(r'第[一二三四五六七八九十\d]+节')

    lines = full_text.split('\n')

    def save_chapter():
        nonlocal content_buffer
        if current_book and current_chapter and content_buffer:
            key = f"{current_book}_{current_chapter}"
            if current_section:
                key += f"_{current_section}"
            chapters[key] = '\n'.join(content_buffer).strip()
        content_buffer = []

    def set_chapter(title: str):
        nonlocal current_chapter, current_section, in_chapter
        save_chapter()
        current_chapter = title
        current_section = ""
        in_chapter = True

    def set_section(title: str):
        nonlocal current_section, content_buffer
        save_chapter()
        current_section = title
        content_buffer = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # 1. 检查是否包含书名
        book_match = book_pattern.search(stripped)
        has_book = book_match is not None

        if has_book:
            # 更新书名：从行中提取纯书名（去掉后面的章/节信息）
            pure_book = re.sub(r'第[一二三四五六七八九十\d]+[章节].*', '', stripped).strip()
            # 如果纯书名为空（比如整行就是“第一章 xxx 模拟试题汇编”），就用 stripped 作书名
            if not pure_book:
                pure_book = stripped.strip()
            current_book = pure_book

        # 2. 检查该行是否包含章标题
        chap_match = chapter_pattern.search(stripped)
        sec_match = section_pattern.search(stripped) if not chap_match else None

        # 3. 如果同时有书名和章标题，先更新书名，再处理章节
        if chap_match:
            # 跨行合并
            title = stripped
            if len(stripped) < 15 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not chapter_pattern.search(next_line) and not section_pattern.search(next_line) and not book_pattern.search(next_line):
                    title = stripped + ' ' + next_line
            set_chapter(title)
            continue

        # 4. 仅节标题（没有章标题）
        if sec_match and in_chapter:
            title = stripped
            if len(stripped) < 15 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not chapter_pattern.search(next_line) and not section_pattern.search(next_line) and not book_pattern.search(next_line):
                    title = stripped + ' ' + next_line
            set_section(title)
            continue

        # 5. 普通内容
        if in_chapter:
            content_buffer.append(stripped)

    save_chapter()
    return chapters

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

def save_summaries_to_chroma(summaries: dict, persist_dir: str = "./chroma_db", collection_name: str = "guide_summary"):
    """将生成的摘要存入 Chroma 摘要库"""
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    texts = []
    metadatas = []
    ids = []
    for i, (chapter_title, summary) in enumerate(summaries.items()):
        texts.append(summary)
        metadatas.append({"chapter": chapter_title, "source": "章节摘要"})
        ids.append(f"summary_{i}")

    if texts:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"已存入 {len(texts)} 条摘要到集合 '{collection_name}'。")
    else:
        print("没有生成任何摘要，请检查章节拆分结果。")

# ================= 主流程 =================
if __name__ == "__main__":
    print("正在提取章节文本...")
    chapter_texts = extract_chapter_texts(PDF_PATH, start_page=15)
    print(f"检测到 {len(chapter_texts)} 章。")

    summaries = {}
    for title, text in chapter_texts.items():
        print(f"正在生成摘要：{title}")
        summary = generate_summary(title, text)
        summaries[title] = summary

    print("正在存入向量库...")
    save_summaries_to_chroma(summaries)
    print("完成！")