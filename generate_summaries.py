# generate_summaries_all.py
import os
import re
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ================= 📌 四本教材配置 =================
BOOKS = [
    {
        "pdf_path": "导游业务统编教材.pdf",
        "book_name": "导游业务统编教材",
        "start_page": 4,
    },
    {
        "pdf_path": "地方导游基础知识统编教材.pdf",
        "book_name": "地方导游基础知识统编教材",
        "start_page": 2,
    },
    {
        "pdf_path": "全国导游基础知识统编教材.pdf",
        "book_name": "全国导游基础知识统编教材",
        "start_page": 3,
    },
    {
        "pdf_path": "政策与法律法规统编教材.pdf",
        "book_name": "政策与法律法规统编教材",
        "start_page": 4,
    },
]

# ================= 公共配置 =================
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "guide_summary"

# 摘要生成模型
SUMMARY_MODEL = ChatOpenAI(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
    # 显式关闭思考模式，等价于旧 deepseek-chat 的非思考行为
    extra_body={"thinking": {"type": "disabled"}},
)

# 嵌入模型
embeddings = DashScopeEmbeddings(model="text-embedding-v4")

# 全局 Chroma 存储（只创建一次）
summary_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

# ================= 广告清洗 =================
def clean_ad_lines(text: str) -> str:
    ad_patterns = [
        # 原有规则
        r"微信公众号\s*：\s*daoyoukaoshizhongxin",
        r"微信客服\s*：\s*daoyoukaoshipeixun",
        r"QQ\s*:\s*17059435",

        # 增强规则：匹配包含“微信公众号”、“微信客服”、“QQ”等短行的完整行
        r"^.{0,10}(微信公众号|微信客服|QQ\s*:|扫码加好友|添加微信).{0,30}$",
        r"下边的\s*微信\s*公众\s*号\s*和\s*客服\s*不要添加\s*[！!].*",
        r"是机构.{0,10}口碑不好.{0,10}不是我.{0,5}",

        # 超强规则：如果一行同时包含数字、联系方式和机构名，直接删除
        r"^\s*\d{5,}\s*$",  # 纯数字行（可能是QQ号）
        r"微信号\s*[：:]\s*\w+",
        r"扫\s*码\s*添\s*加",
    ]
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        is_ad = False
        for p in ad_patterns:
            if re.search(p, line, re.IGNORECASE):
                is_ad = True
                break
        if not is_ad:
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
    part_pattern = re.compile(
        r'^第[一二三四五六七八九十\d]+篇(?:\s+[^\n]{0,10})?\s*$'
    )
    chapter_split_pattern = re.compile(
        r'(.*?)(第[一二三四五六七八九十\d]+章(?:[^\n]{0,30})?)(.*)'
    )
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

        # 跳过明显是页眉/页脚的短行
        if len(stripped) < 5 and not re.search(r'第[一二三四五六七八九十\d]+[章节篇]', stripped):
            continue

        # 1. 先检查是否为篇标题
        if part_pattern.match(stripped):
            title = stripped
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) < 30 and not chapter_split_pattern.search(next_line) and not section_pattern.match(next_line):
                    title = stripped + ' ' + next_line
            set_part(title)
            continue

        # 2. 检查是否包含章标题
        chap_match = chapter_split_pattern.search(stripped)
        if chap_match:
            before = chap_match.group(1).strip()
            title = chap_match.group(2).strip()
            after = chap_match.group(3).strip()

            if before and in_chapter:
                content_buffer.append(before)

            if len(title) < 15 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not chapter_split_pattern.search(next_line) and not part_pattern.match(next_line) and len(next_line) < 30:
                    title = title + ' ' + next_line

            set_chapter(title)

            if after:
                content_buffer.append(after)
            continue

        # 3. 检查节标题
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

    save_chapter()
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
def save_summaries_to_chroma(summaries: dict, book_name: str, store: Chroma):
    texts, metadatas, ids = [], [], []
    for i, (full_key, summary) in enumerate(summaries.items()):
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
    overall_success = True
    for book in BOOKS:
        pdf_path = book["pdf_path"]
        book_name = book["book_name"]
        start_page = book.get("start_page", 1)

        print(f"\n📖 正在处理：《{book_name}》")
        try:
            chapter_texts = extract_chapter_texts(
                pdf_path,
                book_name=book_name,
                start_page=start_page
            )
            print(f"📚 检测到 {len(chapter_texts)} 个章节（含篇）。")

            summaries = {}
            for key, text in chapter_texts.items():
                print(f"⏳ 正在生成摘要：{key}")
                summary = generate_summary(key, text)
                summaries[key] = summary

            print("💾 正在存入向量库...")
            save_summaries_to_chroma(summaries, book_name, summary_store)
            print(f"🎉 《{book_name}》处理完成！")

        except Exception as e:
            print(f"❌ 《{book_name}》处理失败：{str(e)}")
            overall_success = False

    if overall_success:
        print("\n🎊 所有教材摘要生成完毕，数据已全部存入向量库。")
    else:
        print("\n⚠️ 部分教材处理失败，请检查错误信息并重试。")