import os
import sys

import redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langfuse._client.observe import observe
from langgraph.checkpoint.redis import RedisSaver
from langgraph.prebuilt import create_react_agent
from tools import search_questions, search_textbook, grade_answer, hybrid_search, multi_search, rewritten_search, \
    parent_child_search, load_mcp_tools_http  # 导入你的工具
from dotenv import load_dotenv
import asyncio

load_dotenv()  # 自动查找并加载项目根目录下的 .env 文件
memory = RedisSaver(redis_url="redis://redis:6379")
memory.setup()


llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",  # 必须加 /v1
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
    streaming=True
)

def load_system_prompt(filepath="prompts/system_prompt.md"):
    """从 Markdown 文件加载 System Prompt"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
SYSTEM_PROMPT = load_system_prompt()

# 2. 告诉 Agent 它能用什么工具
def get_agent_for_mode(mode: str):
    """根据模式动态创建 Agent，实现工具懒加载"""
    tools = []
    final_prompt = SYSTEM_PROMPT
    if mode == "📖 教材知识问答":
        tools = [search_textbook, hybrid_search, multi_search,
                 rewritten_search, parent_child_search]
    elif mode == "📝 智能出卷":
        tools = [search_questions]
    elif mode == "📊 阅卷批改":
        tools = [search_textbook, grade_answer]   # ← 确保这里有 grade_answer
    # 加载自己写的天气 MCP Server（假设运行在 weather-mcp 容器，端口 8000）
    weather_tools = load_mcp_tools_http("http://weather-mcp:8000")
    tools.extend(weather_tools)
    print(f"✅ 加载了 {len(weather_tools)} 个 MCP 工具: {[t.name for t in weather_tools]}")
    return create_react_agent(llm, tools, checkpointer=memory, prompt=final_prompt)





prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

# agent = create_react_agent(llm, tools, checkpointer=memory, prompt=prompt_template)

from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from openai import RateLimitError, APIConnectionError, InternalServerError, APITimeoutError
import logging

logger = logging.getLogger(__name__)

RETRYABLE = (RateLimitError, APIConnectionError, InternalServerError, APITimeoutError)

@observe()
def stream_agent_with_retry(agent, messages, config=None):
    """带重试的流式调用生成器"""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _stream():
        # 每次重试都重新调用 agent.stream() 获取全新生成器
        for chunk in agent.stream({"messages": messages}, config=config):
            yield chunk

    yield from _stream()



