# 🎓 导游考试 AI Agent Demo

> 一个基于 LangChain + LangGraph 的智能导游考试助手，支持教材知识检索、智能出卷、自动阅卷和多轮对话记忆。
> [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## ✨ 功能特性

- **📖 教材知识问答**：基于 RAG 技术，从本地教材向量库中检索知识点，回答学员的疑问。
- **📝 智能出卷**：支持自然语言描述出题需求，自动从题库中检索真实题目并生成试卷。
- **📊 自动阅卷**：批改学员提交的答案，返回对错结果、解析，并联动推荐薄弱知识点。
- **🧠 多轮对话记忆**：可选的连续对话模式，让 Agent 记住上下文，支持连续追问。
- **🔧 工具调用可视化**：提供折叠面板，展示 Agent 的每一步思考过程和工具调用记录。

## 🛠️ 技术栈

| 层次           | 技术                                     |
| -------------- | ---------------------------------------- |
| **框架**       | LangChain, LangGraph                     |
| **大语言模型** | DeepSeek (Chat)                          |
| **向量数据库** | ChromaDB                                 |
| **嵌入模型**   | 阿里云百炼 DashScope (text-embedding-v3) |
| **前端**       | Streamlit                                |
| **文档处理**   | PyPDF, RecursiveCharacterTextSplitter    |
| **工具调用**   | @tool 装饰器, 自定义工具函数             |
| **环境管理**   | Python 3.11, venv, pip                   |

## 🏗️ 架构图

```mermaid
flowchart TD
    A[用户输入] --> B[Streamlit 前端]
    B --> C{Agent 决策}
    C -->|教材知识| D[search_textbook]
    C -->|出卷需求| E[search_questions]
    C -->|批改需求| F[grade_answer]
    D --> G[Chroma 向量库]
    G --> H[阿里云 Embedding]
    E --> I[JSON 题库]
    F --> I
    D --> J[返回回答]
    E --> J
    F --> J
    J --> B