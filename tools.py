import json
import os
import re
# from rank_bm25 import BM25Okapi
# import jieba
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.embeddings import DashScopeEmbeddings

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader



print("当前工作目录:", os.getcwd())
print("预期文件完整路径:", os.path.join(os.getcwd(), "question_bank.json"))


@tool
def search_questions(chapter: str, qtype: Optional[str] = "全部", count: int = 5) -> str:
    """从题库中按章节和题型检索题目。参数：chapter(必须), qtype(可选，单选/多选/判断), count(返回数量)，从本地题库中精确检索真实的考试题目。
    当用户请求出题、找题目、做练习时必须使用本工具，严禁自行编造题目。"""
    try:
        with open("question_bank.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        return "题库文件未找到，请联系管理员。"
    filter_result = [q for q in questions if chapter in q["chapter"]]
    if qtype != "全部":
        filter_result = [q for q in filter_result if q["type"] == qtype]
    # 取前 count 道题
    selected = filter_result[:count]
    if not selected:
        return f"未找到章节'{chapter}'中题型为'{qtype}'的题目。"

    # 格式化输出
    result = []
    for q in selected:
        result.append(f"ID:{q['id']} [{q['type']}] {q['question']}\n选项：{' / '.join(q['options'])}")
    return "\n\n".join(result)

# 创建或加载向量库（只需执行一次）
# 设置你的 DeepSeek API Key（推荐用环境变量，别硬编码）
os.environ["DEEPSEEK_API_KEY"] = "sk-"
os.environ["DASHSCOPE_API_KEY"] = "sk-"


def load_embedding_model():
    """加载并缓存 BGE 嵌入模型"""
    print("step 1")
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")

    return embeddings

def load_vectorstore():
    """加载或创建 Chroma 向量库"""
    embeddings = load_embedding_model()
    print("step 2")

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,  # 这里使用了新的 DeepSeek 嵌入模型
        collection_name="guide_textbook"
    )
    return vectorstore

def initialize_vectorstore():
    # 如果没有数据，可以初始化加载
    """加载或创建 Chroma 向量库"""
    print("step 3")
    vectorstore = load_vectorstore()
    if vectorstore._collection.count() == 0:  # 最终还是得用这个
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("step 4")

        # 加载并切分你的教材文本
        # 1. 加载 PDF
        loader = PyPDFLoader("全国导游人员资格统一考试模拟试题汇编.pdf")
        docs = loader.load()  # 每一页变成一个 Document 对象
        for i, doc in enumerate(docs):
            doc.metadata["source"] = "全国导游人员资格统一考试模拟试题汇编"
            doc.metadata["chapter"] = detect_chapter(doc.page_content)  # 自定义章节检测
            doc.metadata["page"] = i + 1
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50,
                                                  separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""])
        chunks = splitter.split_documents(docs)

        # 3. 写入向量库（替换原来的纯文本读取）
        ids = [f"textbook_chunk_{i}" for i in range(len(chunks))]
        vectorstore.add_documents(chunks, ids=ids)
        return vectorstore
    else:
        print("向量数据库已存在数据")



@tool
def search_textbook(query: str) -> str:
    """
    从全国导游人员资格统一考试模拟试题汇编中检索任意相关内容，包括但不限于：知识点、章节目录、概念解释、流程步骤等。
    当用户询问任何与教材文本直接相关的问题时，**必须优先使用本工具**，严禁自行编造答案。
    参数 query: 用户问题的关键词或完整句子。
    """
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        print("未找到相关教材内容。")
        return "未找到相关教材内容。"
    for i, doc in enumerate(docs, 1):
        results = []
        results.append(f"【教材片段{i}】\n{doc.page_content}")
    return "\n\n".join(results)



def detect_chapter(text: str) -> str:
    """从文本中检测章节标题"""
    # 匹配“第X章”或“第XX章”或“第一章”等
    match = re.search(r'第[一二三四五六七八九十\d]+章\s*[^\n]*', text)
    if match:
        return match.group().strip()
    # 匹配“Chapter X”这种英文格式（如果有）
    match = re.search(r'Chapter\s+\d+', text, re.IGNORECASE)
    if match:
        return match.group().strip()
    return "未知章节"


@tool
def grade_answer(question_id: str, student_answer: str) -> str:
    """
    根据题库标准答案批改学员的答案，返回对错、解析，并推荐相关知识点。
    当用户要求批改题目、对答案、判分时，**必须**调用本工具。
    参数：
        question_id: 题目的唯一ID（如 q_001）
        student_answer: 学员的答案（如 A、B、C、D，或判断题的“对/错”）
    """
    import json
    try:
        with open("question_bank.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        return "题库文件未找到，请联系管理员。"

    # 查找题目
    question = next((q for q in questions if q["id"] == question_id), None)
    if not question:
        return f"未找到ID为 {question_id} 的题目，请检查题目ID是否正确。"

    correct = question["answer"].strip().upper()
    student = student_answer.strip().upper()

    # 批改结果
    if correct == student:
        result = (
            f"✅ **回答正确！**\n"
            f"题目：{question['question']}\n"
            f"你的答案：{student}\n"
            f"解析：{question.get('explanation', '暂无解析')}"
        )
    else:
        result = (
            f"❌ **回答错误。**\n"
            f"题目：{question['question']}\n"
            f"你的答案：{student}\n"
            f"正确答案：{correct}\n"
            f"解析：{question.get('explanation', '暂无解析')}\n"
            f"💡 建议：你可以让我帮你查找「{question['chapter']}」的相关知识点来巩固学习。"
        )

    return result

# # 初始化 BM25（在启动时做一次）
# def build_bm25_index(chunks):
#     """建立关键词索引"""
#     tokenized_corpus = [list(jieba.cut(chunk)) for chunk in chunks]
#     return BM25Okapi(tokenized_corpus)
#
# # 假设你有 chunks 列表（教材所有片段）
# bm25 = build_bm25_index(chunks)
#
# @tool
# def hybrid_search(query: str, k: int = 5) -> str:
#     """混合检索教材内容：结合语义和关键词。"""
#     # 1. 语义检索（你已有的向量检索）
#     semantic_docs = vectorstore.similarity_search(query, k=k)
#
#     # 2. 关键词检索（BM25）
#     tokenized_query = list(jieba.cut(query))
#     keyword_scores = bm25.get_scores(tokenized_query)
#     top_kw_indices = sorted(range(len(keyword_scores)), key=lambda i: keyword_scores[i], reverse=True)[:k]
#     keyword_docs = [chunks[i] for i in top_kw_indices]
#
#     # 3. 结果融合（简单合并，实际可用 RRF 算法）
#     all_docs = semantic_docs + keyword_docs
#     # 去重（按文本相似度）
#     unique_docs = deduplicate_docs(all_docs)
#     # 限制最终返回数量
#     unique_docs = unique_docs[:k]
#
#     if not unique_docs:
#         return "未找到相关内容。"
#     return "\n\n".join([d.page_content for d in unique_docs])
#
# # 1. 加载 Reranker 模型（中文优化版）
# model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
# compressor = CrossEncoderReranker(model=model, top_n=3)
#
# # 2. 把 Chroma 包装成 Retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 先召回10个
#
# # 3. 创建压缩检索器（Reranker 二次筛选）
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=retriever
# )
#
# # 4. 在工具中使用
# @tool
# def search_with_rerank(query: str) -> str:
#     """检索教材并重排序，确保最相关的结果排在前面。"""
#     docs = compression_retriever.get_relevant_documents(query)
#     if not docs:
#         return "未找到相关内容。"
#     return "\n\n".join([doc.page_content for doc in docs])