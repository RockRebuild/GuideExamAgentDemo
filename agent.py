import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools import search_questions, search_textbook, grade_answer  # 导入你的工具
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()




llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",  # 必须加 /v1
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
    streaming=True
)

# 2. 告诉 Agent 它能用什么工具
tools = [search_textbook, grade_answer]

SYSTEM_PROMPT = """
你是一个导游考试智能助手。你拥有以下工具：
- search_textbook: 从教材中检索任何内容（知识点、目录、习题等）。
- grade_answer: 批改学员的答案。

重要规则：
1. 当用户询问任何与教材相关的内容时，**必须**先调用 search_textbook 工具获取真实内容，不得凭记忆编造。
2. 如果需要批改，调用 grade_answer。
4. 严禁在未调用工具的情况下直接回答与教材或题库有关的问题。
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

agent = create_react_agent(llm, tools, checkpointer=memory,prompt=prompt_template)

