import json
import os
import re

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
    vectorstore = load_vectorstore()
    print("step 3")

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


@tool
def grade_answer(question_id: str, student_answer: str) -> str:
    """根据题库标准答案批改学员答案，返回对错及解析。"""
    with open("question_bank.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    question = next((q for q in questions if q["id"] == question_id), None)
    if not question:
        return f"未找到ID为 {question_id} 的题目。"

    correct = question["answer"].strip().upper()
    student = student_answer.strip().upper()

    if correct == student:
        return f"✅ 回答正确！\n解析：{question.get('explanation', '暂无解析')}"
    else:
        return f"❌ 回答错误。你的答案：{student}，正确答案：{correct}。\n解析：{question.get('explanation', '暂无解析')}"

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