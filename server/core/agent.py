import os
import sys

import redis
from langchain_openai import ChatOpenAI
from langfuse._client.observe import observe
from langgraph.checkpoint.redis import RedisSaver
from langgraph.prebuilt import create_react_agent
from server.core.tools import search_questions, search_textbook, grade_answer, hybrid_search, multi_search, rewritten_search, \
    parent_child_search, load_mcp_tools_http
from dotenv import load_dotenv
import asyncio

load_dotenv()  # 自动查找并加载项目根目录下的 .env 文件
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")

_memory = None
_memory_tried = False


def get_memory():
    """
    懒加载 RedisSaver。
    - Docker 环境：连接 redis 容器，提供会话记忆
    - 本地环境：Redis 不可用时返回 None，Agent 无记忆但仍可工作
    """
    global _memory, _memory_tried
    if _memory is None and not _memory_tried:
        _memory_tried = True
        try:
            _memory = RedisSaver(redis_url=REDIS_URL)
            _memory.setup()
            print(f"✅ Redis 连接成功 ({REDIS_URL})，会话记忆已启用")
        except Exception as e:
            print(f"⚠️ Redis 不可用 ({e})，会话记忆已禁用，但聊天功能正常")
            _memory = None
    return _memory


# 模型名集中由 DEEPSEEK_MODEL 控制，默认 deepseek-v4-flash（旧 deepseek-chat 2026-07-24 下线）
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    base_url="https://api.deepseek.com/v1",  # 必须加 /v1
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
    streaming=True,
    # 显式关闭思考模式，等价于旧 deepseek-chat 的非思考行为
    extra_body={"thinking": {"type": "disabled"}},
)

def load_system_prompt(filepath="prompts/system_prompt.md"):
    """从 Markdown 文件加载 System Prompt"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
SYSTEM_PROMPT = load_system_prompt()

# 2. 告诉 Agent 它能用什么工具
# Agent 缓存：key=mode，value=(agent, weather_loaded)
# 如果天气 MCP 首次加载失败，下次请求会重试，直到成功后再缓存
_agent_cache = {}

def get_agent_for_mode(mode: str):
    """根据模式动态创建 Agent，实现工具懒加载（带缓存，天气 MCP 失败时会重试）"""
    # 如果已缓存且天气工具已成功加载，直接返回
    if mode in _agent_cache:
        cached_agent, was_weather_loaded = _agent_cache[mode]
        if was_weather_loaded:
            return cached_agent
        # 天气之前没加载成功，返回缓存的 agent 但这次尝试重新创建（重试 MCP）

    tools = []
    final_prompt = SYSTEM_PROMPT
    if mode == "📖 教材知识问答":
        tools = [search_textbook, hybrid_search, multi_search,
                 rewritten_search, parent_child_search]
    elif mode == "📝 智能出卷":
        tools = [search_questions]
    elif mode == "📊 阅卷批改":
        tools = [search_textbook, grade_answer]   # ← 确保这里有 grade_answer
    # 加载天气 MCP Server（依次尝试 Docker 容器名 / localhost / 127.0.0.1）
    weather_loaded = False
    for host in ["weather-mcp", "localhost", "127.0.0.1"]:
        url = f"http://{host}:8000"
        try:
            weather_tools = load_mcp_tools_http(url)
            tools.extend(weather_tools)
            print(f"✅ 从 {url} 加载了 {len(weather_tools)} 个 MCP 工具: {[t.name for t in weather_tools]}")
            weather_loaded = True
            break
        except Exception as e:
            print(f"⚠️ 连接 {url} 失败: {e}")
            continue
    if not weather_loaded:
        print("⚠️ 天气 MCP 工具不可用（所有主机均连接失败），跳过")
        # 从 prompt 中移除 get_weather 相关描述，避免 LLM 调用一个不存在的工具
        import re
        final_prompt = re.sub(r'- get_weather[^\n]*\n', '', final_prompt)
        final_prompt = re.sub(r'- 查询天气[^\n]*\n', '', final_prompt)
    agent = create_react_agent(llm, tools, checkpointer=get_memory(), prompt=final_prompt)
    _agent_cache[mode] = (agent, weather_loaded)
    return agent


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
    """带重试的流式调用生成器。

    如果检测到 checkpoint 中有孤儿 tool_calls（上次请求中断），
    自动切换到新 thread_id 重试，避免 LangGraph 状态校验失败。
    """
    import time
    import copy
    last_error = None
    orphan_fixed = False  # 只允许一次孤儿修复，防止死循环
    attempt = 0

    while attempt < 3:
        try:
            for chunk in agent.stream({"messages": messages}, config=config):
                yield chunk
            return  # 成功
        except RETRYABLE as e:
            last_error = e
            attempt += 1
            logger.warning(f"Agent stream 失败 (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(min(2 ** attempt, 10))
        except Exception as e:
            err_msg = str(e)
            # 孤儿 tool_calls：LangGraph checkpoint 中有未完成的工具调用
            if "tool_calls" in err_msg and "ToolMessage" in err_msg:
                if orphan_fixed:
                    raise  # 修复一次后仍失败，不再重试
                orphan_fixed = True
                logger.warning(
                    f"检测到孤儿 tool_calls，切换新会话重试: {err_msg[:200]}"
                )
                # 创建新 thread_id 绕过损坏的 checkpoint
                old_tid = config.get("configurable", {}).get("thread_id", "default")
                new_tid = f"{old_tid}_{int(time.time())}"
                config = copy.deepcopy(config) if config else {}
                config.setdefault("configurable", {})["thread_id"] = new_tid
                # 补上 SystemPrompt（孤儿状态下的新会话需要完整上下文）
                from langchain_core.messages import SystemMessage
                if isinstance(messages, list) and not any(
                    isinstance(m, SystemMessage) for m in messages
                ):
                    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
                # 孤儿修复不消耗重试次数
                logger.info(f"已切换到新 thread_id: {new_tid}")
                continue
            raise



