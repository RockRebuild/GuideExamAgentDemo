import asyncio
import json
import os
import re
import sys
from difflib import SequenceMatcher
from typing import Optional, List, Tuple

import httpx
import numpy as np
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import jieba

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool, StructuredTool
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

print("=== tools.py 开始执行 ===", file=sys.stderr, flush=True)

# ============================================================
# 全局配置
# ============================================================
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_PARAGRAPH = "guide_child"   # 段落级索引（细粒度，用于语义/混合检索）
COLLECTION_SUMMARY = "guide_summary"   # 摘要级索引（粗粒度，用于快速定位章节）
COLLECTION_SENTENCE = "guide_sentence" # 命题/句子级索引（预留，需另行生成）

# 用于在工具返回文本中分隔单个上下文的标识符，便于 app.py 拆分出 contexts 列表
CONTEXT_SEPARATOR = "\n---\n"

# 嵌入模型（全局复用）
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    return _embeddings

# ============================================================
# 统一检索入口（精排前移）
# ============================================================
def retrieve_with_rerank(query: str, raw_contexts: List[str]) -> str:
    """
    对原始上下文列表执行精排，并以分隔符拼接为字符串返回。
    同时将精排后的列表存入 st.session_state，方便 app.py 直接获取。
    """
    from retrieval_utils import refine_and_rerank, determine_top_k

    if not raw_contexts:
        return ""

    top_k = determine_top_k(query)
    refined = refine_and_rerank(raw_contexts, query, top_k=top_k)

    if not refined:
        return ""

    # 保存到 session_state 供评估使用（可选）
    st.session_state["last_refined_contexts"] = refined

    # 用分隔符拼接，便于 app.py 拆分
    return CONTEXT_SEPARATOR.join(refined)


# ============================================================
# 向量库获取工具
# ============================================================
def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )


# ============================================================
# BM25 关键词检索索引（段落库）
# ============================================================
_bm25_index = None
_bm25_docs = None

def _init_bm25():
    """用 guide_parent（800字符父切片）构建 BM25 索引，信息量更大，召回更准"""
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return
    parent_store = get_vectorstore("guide_parent")
    all_data = parent_store.get()
    if not all_data['documents']:
        # 兜底：如果 guide_parent 为空，退回到 guide_child
        child_store = get_vectorstore(COLLECTION_PARAGRAPH)
        all_data = child_store.get()
    if not all_data['documents']:
        return
    _bm25_docs = all_data['documents']
    tokenized_corpus = [list(jieba.cut(doc)) for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized_corpus)


# ============================================================
# 文本去重
# ============================================================
def deduplicate_docs(docs: List[Document], threshold: float = 0.8) -> List[Document]:
    unique = []
    for doc in docs:
        text = doc.page_content if hasattr(doc, 'page_content') else doc
        is_dup = False
        for existing in unique:
            existing_text = existing.page_content if hasattr(existing, 'page_content') else existing
            if SequenceMatcher(None, text, existing_text).ratio() > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(doc)
    return unique


# ============================================================
# 数据初始化（首次运行时创建索引）
# ============================================================
def detect_chapter(text: str) -> str:
    match = re.search(r'第[一二三四五六七八九十\d]+章\s*[^\n]*', text)
    if match:
        return match.group().strip()
    return "未知章节"


def initialize_vectorstores(pdf_path: str = "全国导游人员资格统一考试模拟试题汇编.pdf"):
    para_store = get_vectorstore(COLLECTION_PARAGRAPH)
    if para_store._collection.count() == 0:
        print("段落库为空，开始导入 PDF ...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = os.path.basename(pdf_path)
            doc.metadata["chapter"] = detect_chapter(doc.page_content)
            doc.metadata["page"] = i + 1
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        ids = [f"para_{i}" for i in range(len(chunks))]
        para_store.add_documents(chunks, ids=ids)
        print(f"段落库导入完成，共 {len(chunks)} 条。")
    else:
        print(f"段落库已存在，当前文档数：{para_store._collection.count()}")

    summary_store = get_vectorstore(COLLECTION_SUMMARY)
    if summary_store._collection.count() == 0:
        print("摘要库为空。请手动生成章节摘要并调用 add_texts 存入 guide_summary。")

    sent_store = get_vectorstore(COLLECTION_SENTENCE)
    if sent_store._collection.count() == 0:
        print("句子库为空。可按需生成命题/句子切片。")


# ============================================================
# 工具函数（均已接入统一精排入口）
# ============================================================

@tool
def hybrid_search(query: str, k: int = 12) -> str:
    """
    混合检索教材内容，结合语义搜索和关键词搜索。
    当用户询问任何与教材相关的问题时，优先使用本工具。
    """
    para_store = get_vectorstore(COLLECTION_PARAGRAPH)
    semantic_docs = para_store.similarity_search(query, k=k)

    _init_bm25()
    keyword_docs = []
    if _bm25_index is not None and _bm25_docs:
        tokenized_query = list(jieba.cut(query))
        scores = _bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]
        for idx in top_indices:
            keyword_docs.append(Document(page_content=_bm25_docs[idx]))

    all_docs = semantic_docs + keyword_docs
    unique_docs = deduplicate_docs(all_docs, threshold=0.8)[:k]

    if not unique_docs:
        return "未找到相关教材内容。"

    # 构造带元数据的原始上下文（作为精排的输入）
    raw_contexts = []
    for doc in unique_docs:
        chapter = doc.metadata.get("chapter", "未知章节")
        page = doc.metadata.get("page", "未知页")
        raw_contexts.append(f"【来源：{chapter}，第{page}页】\n{doc.page_content}")

    return retrieve_with_rerank(query, raw_contexts)


@tool
def search_textbook(query: str) -> str:
    """
    从教材中检索相关内容（语义搜索）。
    """
    para_store = get_vectorstore(COLLECTION_PARAGRAPH)
    docs_with_scores = para_store.similarity_search_with_relevance_scores(query, k=12)
    relevant_docs = [doc for doc, score in docs_with_scores if score > 0.5]

    if not relevant_docs:
        return "未在教材中找到与您问题相关的内容。"

    raw_contexts = []
    for doc in relevant_docs:
        chapter = doc.metadata.get("chapter", "未知章节")
        page = doc.metadata.get("page", "未知页")
        raw_contexts.append(f"【来源：{chapter}，第{page}页】\n{doc.page_content}")

    return retrieve_with_rerank(query, raw_contexts)


@tool
def multi_search(query: str, k: int = 12) -> str:
    """
    多粒度并行检索：同时查摘要、段落、句子，合并去重。
    """
    summary_store = get_vectorstore(COLLECTION_SUMMARY)
    para_store = get_vectorstore(COLLECTION_PARAGRAPH)
    sent_store = get_vectorstore(COLLECTION_SENTENCE)

    summary_docs = summary_store.similarity_search(query, k=k)
    paragraph_docs = para_store.similarity_search(query, k=k)
    sentence_docs = sent_store.similarity_search(query, k=k)

    all_docs = summary_docs + paragraph_docs + sentence_docs
    unique_docs = deduplicate_docs(all_docs, threshold=0.8)[:k]

    if not unique_docs:
        return "未找到相关内容。"

    raw_contexts = []
    for doc in unique_docs:
        source = doc.metadata.get("source", "教材")
        raw_contexts.append(f"【片段】({source})\n{doc.page_content}")

    return retrieve_with_rerank(query, raw_contexts)


@tool
def parent_child_search(query: str, k: int = 12) -> str:
    """
    父子切片检索：用细粒度子切片匹配，返回大粒度父切片供 LLM 阅读。
    """
    try:
        child_store = get_vectorstore("guide_child")
        parent_store = get_vectorstore("guide_parent")

        child_docs = child_store.similarity_search(query, k=k * 2)
        parent_ids = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid:
                parent_ids.add(pid)
        parent_ids = list(parent_ids)[:k]

        if not parent_ids:
            return "未找到相关教材内容。请尝试更换关键词。"

        parent_data = parent_store.get(ids=parent_ids)
        docs_texts = parent_data.get("documents", [])
        metadatas = parent_data.get("metadatas", [])
        if not docs_texts:
            return "未找到对应的教材段落，请联系管理员。"

        print(f"🔥 parent_child_search 返回的父切片数量: {len(docs_texts)}", flush=True)

        # 构造带来源标注的上下文（与 hybrid_search / search_textbook 保持一致）
        raw_contexts = []
        for i, doc_text in enumerate(docs_texts):
            meta = metadatas[i] if i < len(metadatas) else {}
            chapter = meta.get("chapter", "未知章节")
            page = meta.get("page", "未知页")
            raw_contexts.append(f"【来源：{chapter}，第{page}页】\n{doc_text}")

        return retrieve_with_rerank(query, raw_contexts)

    except Exception as e:
        return f"parent_child_search 工具执行时发生错误：{str(e)}"


@tool
def rewritten_search(original_query: str, k: int = 12) -> str:
    """
    问题改写检索：将用户问题改写为多个变体后并行检索，提高召回率。
    """
    llm = ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0.3,  # 非零温度产生多样化的改写变体，提高召回率
        extra_body={"thinking": {"type": "disabled"}},
    )
    rewrite_prompt = f"""将以下用户问题改写成 3 个不同角度的检索查询，每个查询占一行，不要编号和其他文字。
每个查询应有不同的侧重点，尽可能涵盖问题涉及的不同关键词组合。
原始问题：{original_query}
改写查询："""

    response = llm.invoke(rewrite_prompt)
    rewritten_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    rewritten_queries.append(original_query)

    para_store = get_vectorstore(COLLECTION_PARAGRAPH)
    all_docs = []
    for q in rewritten_queries:
        docs = para_store.similarity_search(q, k=k)
        all_docs.extend(docs)

    unique_docs = deduplicate_docs(all_docs, threshold=0.8)[:k]
    if not unique_docs:
        return "未找到相关内容。"

    raw_contexts = []
    for doc in unique_docs:
        chapter = doc.metadata.get("chapter", "未知章节")
        page = doc.metadata.get("page", "未知页")
        raw_contexts.append(f"【来源：{chapter}，第{page}页】\n{doc.page_content}")
    return retrieve_with_rerank(original_query, raw_contexts)


@tool
def search_questions(chapter: str, qtype: Optional[str] = "全部", count: int = 5) -> str:
    """从题库中按章节和题型检索题目。"""
    try:
        with open("question_bank.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        return "题库文件未找到，请联系管理员。"

    filter_result = [q for q in questions if chapter in q["chapter"]]
    if qtype != "全部":
        filter_result = [q for q in filter_result if q["type"] == qtype]

    selected = filter_result[:count]
    if not selected:
        return f"未找到章节'{chapter}'中题型为'{qtype}'的题目。"

    result = []
    for q in selected:
        result.append(f"ID:{q['id']} [{q['type']}] {q['question']}\n选项：{' / '.join(q['options'])}")
    return "\n\n".join(result) + "\n\n💡 批改时请告诉我题号，例如“科目二 第三章 单选题 第1题”。"


@tool
def grade_answer(question_id: Optional[str] = None,
                 subject: Optional[str] = None,
                 chapter: Optional[str] = None,
                 qtype: Optional[str] = None,
                 index: int = 0,
                 student_answer: str = "") -> str:
    """批改学员的答案。"""
    try:
        with open("question_bank.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        return "题库文件未找到。"

    if question_id:
        question = next((q for q in questions if q["id"] == question_id), None)
    elif chapter and qtype and index > 0:
        filtered = [q for q in questions if q["subject"] == subject and chapter in q["chapter"] and q["type"] == qtype]
        question = filtered[index - 1] if 1 <= index <= len(filtered) else None
        if question:
            question_id = question["id"]
    else:
        return "请提供题目ID，或者同时提供章节、题型和题号。"

    if not question:
        return "未找到对应题目，请检查信息。"

    correct = question["answer"].strip().upper()
    student = student_answer.strip().upper()

    if correct == student:
        result = f"✅ 回答正确！\n题目：{question['question']}\n你的答案：{student}\n解析：{question.get('explanation', '暂无解析')}"
    else:
        result = (f"❌ 回答错误。\n题目：{question['question']}\n你的答案：{student}\n"
                  f"正确答案：{correct}\n解析：{question.get('explanation', '暂无解析')}\n"
                  f"💡 建议：复习「{question['chapter']}」的相关知识点。")
    return result


# ============================================================
# MCP 工具加载（HTTP 模式）
# ============================================================
async def _load_mcp_tools_http_async(url: str) -> List[StructuredTool]:
    tools = []
    client = httpx.AsyncClient(base_url=url)
    resp = await client.post("/mcp", json={"method": "tools/list"})
    data = resp.json()

    for tool_def in data.get("tools", []):
        def make_sync_call(name=tool_def["name"], desc=tool_def.get("description", ""),
                           input_schema=tool_def.get("inputSchema", {})):
            def sync_call(**kwargs):
                async def _call():
                    async with httpx.AsyncClient(base_url=url) as c:
                        resp = await c.post("/mcp", json={
                            "method": "tools/call",
                            "params": {"name": name, "arguments": kwargs}
                        })
                        result = resp.json()
                        content = result.get("content", [])
                        if content:
                            return content[0].get("text", str(result))
                        return str(result)
                return asyncio.run(_call())
            return sync_call

        tools.append(
            StructuredTool.from_function(
                func=make_sync_call(),
                name=tool_def["name"],
                description=tool_def.get("description", ""),
            )
        )
    return tools


def load_mcp_tools_http(url: str) -> List[StructuredTool]:
    return asyncio.run(_load_mcp_tools_http_async(url))


# ============================================================
# 辅助：从工具返回的字符串中提取 contexts（供 app.py → RAGAS 使用）
# ============================================================
def extract_contexts_from_response(response: str) -> List[str]:
    """按分隔符拆分，返回与 LLM 所见到完全一致的 contexts 列表"""
    if not response:
        return []
    return [c.strip() for c in response.split(CONTEXT_SEPARATOR) if c.strip()]