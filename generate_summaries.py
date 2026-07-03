# generate_summaries_single.py
import os
import re
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ================= 📌 每次修改这里 =================
PDF_PATH = "政策与法律法规统编教材.pdf"   # 当前这本书的 PDF 路径
BOOK_NAME = "政策与法律法规统编教材"       # 当前这本书的书名
START_PAGE = 3                            # 跳过前言等页面
# ==================================================

# 摘要库配置
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "guide_summary"

# 模型配置
SUMMARY_MODEL = ChatOpenAI(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
    # 显式关闭思考模式，等价于旧 deepseek-chat 的非思考行为
    extra_body={"thinking": {"type": "disabled"}},
)
embeddings = DashScopeEmbeddings(model="text-embedding-v3")


# ================= 广告清洗 =================
def clean_ad_lines(text: str) -> str:
    ad_patterns = [
        r"微信公众号\s*：\s*daoyoukaoshizhongxin",
        r"微信客服\s*：\s*daoyoukaoshipeixun",
        r"QQ\s*:\s*17059435",
        r"下边的\s*微信\s*公众\s*号\s*和\s*客服\s*不要添加\s*[！!]\s*是机构\s*[（(]口碑不好[）)]\s*[！!]{1,2}\s*不是我\s*[！!]{1,2}",
    ]
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if any(re.search(p, line, re.IGNORECASE) for p in ad_patterns):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ================= 章节提取 =================
def extract_chapter_texts(pdf_path: str,
                          book_name: str,
                          start_page: int = 1,
                          end_page: int = None) -> dict:
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    if start_page < 1:
        start_page = 1
    if end_page is None or end_page > total_pages:
        end_page = total_pages
    selected_pages = reader.pages[start_page - 1: end_page]
    full_text = "\n".join([page.extract_text() or "" for page in selected_pages])
    full_text = clean_ad_lines(full_text)
    lines = full_text.split('\n')

    # 正则：严格匹配篇/章/节
    # 篇：整行以“第X篇”开头，后面最多10个字的副标题，且不包含常见非标题词汇
    part_pattern = re.compile(
        r'^第[一二三四五六七八九十\d]+篇(?:\s+[^\n]{0,10})?\s*$'
    )
    # 章：匹配行中出现的“第X章”，允许之前有正文（会被分离）
    chapter_split_pattern = re.compile(
        r'(.*?)(第[一二三四五六七八九十\d]+章(?:[^\n]{0,30})?)(.*)'
    )
    # 节：匹配“第X节”，仅当已进入章时有效
    section_pattern = re.compile(
        r'^第[一二三四五六七八九十\d]+节(?:[^\n]{0,30})?\s*$'
    )

    current_part = ""
    current_chapter = ""
    current_section = ""
    content_buffer = []
    in_chapter = False
    chapters = {}

    def save_chapter():
        nonlocal content_buffer
        if current_chapter and content_buffer:
            key = book_name
            if current_part:
                key += f"_{current_part}"
            key += f"_{current_chapter}"
            if current_section:
                key += f"_{current_section}"
            chapters[key] = '\n'.join(content_buffer).strip()
        content_buffer = []

    def set_part(title: str):
        nonlocal current_part, current_chapter, current_section, in_chapter
        save_chapter()
        current_part = title
        current_chapter = ""
        current_section = ""
        in_chapter = False

    def set_chapter(title: str):
        nonlocal current_chapter, current_section, in_chapter
        save_chapter()
        current_chapter = title
        current_section = ""
        in_chapter = True

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # 跳过纯页码或分隔线
        if re.fullmatch(r'[-—\d\s]{2,}', stripped):
            continue

        # 跳过明显是页眉/页脚的短行（如单独的数字、缩略名）
        if len(stripped) < 5 and not re.search(r'第[一二三四五六七八九十\d]+[章节篇]', stripped):
            continue

        # 1. 先检查是否为篇标题（严格行首匹配）
        if part_pattern.match(stripped):
            title = stripped
            # 跨行合并副标题（如果下一行很短且不是章节标题）
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) < 30 and not chapter_split_pattern.search(next_line) and not section_pattern.match(next_line):
                    title = stripped + ' ' + next_line
            set_part(title)
            continue

        # 2. 检查是否包含章标题（允许行中任意位置）
        chap_match = chapter_split_pattern.search(stripped)
        if chap_match:
            # 章标题之前的文字当作上一章的正文
            before = chap_match.group(1).strip()
            title = chap_match.group(2).strip()
            after = chap_match.group(3).strip()

            if before and in_chapter:
                content_buffer.append(before)

            # 跨行合并章标题（如果标题很短且下一行明显是副标题）
            if len(title) < 15 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not chapter_split_pattern.search(next_line) and not part_pattern.match(next_line) and len(next_line) < 30:
                    title = title + ' ' + next_line

            set_chapter(title)

            # 如果章标题后面还有文字，作为本章内容第一行
            if after:
                content_buffer.append(after)
            continue

        # 3. 检查节标题（必须已进入章，且行首匹配）
        if in_chapter and section_pattern.match(stripped):
            title = stripped
            if len(title) < 15 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not section_pattern.match(next_line) and len(next_line) < 30:
                    title = title + ' ' + next_line
            save_chapter()
            current_section = title
            content_buffer = []
            continue

        # 4. 普通正文
        if in_chapter:
            content_buffer.append(stripped)

    save_chapter()  # 保存最后一章
    return chapters


# ================= 生成摘要 =================
def generate_summary(chapter_title: str, chapter_text: str) -> str:
    prompt = f"""请用一两句话概括以下章节的核心内容，只返回摘要文本，不要多余解释。

章节标题：{chapter_title}

章节内容：
{chapter_text[:3000]}

摘要："""
    response = SUMMARY_MODEL.invoke(prompt)
    return response.content.strip()


# ================= 存入 Chroma =================
def save_summaries_to_chroma(summaries: dict, book_name: str):
    store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    texts, metadatas, ids = [], [], []
    for i, (full_key, summary) in enumerate(summaries.items()):
        # 从 full_key 中解析篇和章（格式：书名_篇_章 或 书名_章）
        parts = full_key.replace(book_name + "_", "", 1).split('_')
        part = ""
        chapter = ""
        if len(parts) > 0 and '篇' in parts[0]:
            part = parts[0]
            chapter = '_'.join(parts[1:]) if len(parts) > 1 else ""
        else:
            chapter = '_'.join(parts)

        texts.append(summary)
        metadatas.append({
            "book": book_name,
            "part": part,
            "chapter": chapter,
            "full_key": full_key
        })
        ids.append(f"{book_name}_summary_{i}")

    if texts:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"✅ 已存入 {len(texts)} 条摘要到集合 '{COLLECTION_NAME}'。")
    else:
        print("⚠️ 没有生成任何摘要，请检查章节拆分结果。")


# ================= 主流程 =================
if __name__ == "__main__":
    print(f"📖 正在处理：《{BOOK_NAME}》")
    chapter_texts = extract_chapter_texts(
        PDF_PATH,
        book_name=BOOK_NAME,
        start_page=START_PAGE
    )
    print(f"📚 检测到 {len(chapter_texts)} 个章节（含篇）。")

    summaries = {}
    for key, text in chapter_texts.items():
        print(f"⏳ 正在生成摘要：{key}")
        summary = generate_summary(key, text)
        summaries[key] = summary

    print("💾 正在存入向量库...")
    save_summaries_to_chroma(summaries, BOOK_NAME)
    print("🎉 完成！")