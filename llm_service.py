# llm_service.py

import asyncio
import logging
from datetime import date
from typing import Any, Dict

import redis
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import RateLimitError, APIConnectionError, InternalServerError, APITimeoutError

logger = logging.getLogger(__name__)

RETRYABLE = (RateLimitError, APIConnectionError, InternalServerError, APITimeoutError)

DEEPSEEK_V4_FLASH_INPUT_PRICE = 1.0  # $0.14 / 百万 tokens
DEEPSEEK_V4_FLASH_OUTPUT_PRICE = 2.0
class LLMService:
    """LLM API 统一服务层：重试、超时、并发控制、成本记录"""

    def __init__(
        self,
        agent,
        redis_client: redis.Redis,
        max_concurrency: int = 3,
        timeout_seconds: int = 30,
    ):
        self.agent = agent
        self.redis = redis_client
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout = timeout_seconds

    @staticmethod
    def _extract_token_usage(response) -> tuple[int, int]:
        """从 Agent 返回值中提取 Token 用量"""
        messages = response.get("messages", [])
        total_input = sum(
            msg.usage_metadata.get("input_tokens", 0)
            for msg in messages
            if hasattr(msg, "usage_metadata")
        )
        total_output = sum(
            msg.usage_metadata.get("output_tokens", 0)
            for msg in messages
            if hasattr(msg, "usage_metadata")
        )
        return total_input, total_output

    def _record_usage(self, input_tokens: int, output_tokens: int):
        """写入 Redis，按日聚合"""
        today = str(date.today())
        key = f"token_usage:{today}"
        current = self.redis.hgetall(key)
        new_input = int(current.get("input", 0)) + input_tokens
        new_output = int(current.get("output", 0)) + output_tokens
        self.redis.hset(key, mapping={
            "input": new_input,
            "output": new_output,
            "total": new_input + new_output,
        })
        self.redis.expire(key, 60 * 60 * 48)

    # ========== 同步调用（带重试 + 成本记录） ==========
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def invoke(self, messages: list, config: Dict[str, Any] | None = None):
        """同步调用 Agent，自动重试并记录成本"""
        response = self.agent.invoke({"messages": messages}, config=config)
        input_tok, output_tok = self._extract_token_usage(response)
        self._record_usage(input_tok, output_tok)
        return response

    # ========== 异步调用（带 Semaphore + 重试 + 成本记录） ==========
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def ainvoke(self, messages: list, config: Dict[str, Any] | None = None):
        """异步调用 Agent，受 Semaphore 控制，自动重试并记录成本"""
        async with self.semaphore:
            response = await self.agent.ainvoke(
                {"messages": messages}, config=config
            )
        input_tok, output_tok = self._extract_token_usage(response)
        self._record_usage(input_tok, output_tok)
        return response

    # ========== 流式调用（带重试） ==========
    def stream(self, messages, config=None):
        """流式调用，带重试"""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(RETRYABLE),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        def _stream():
            yield from self.agent.stream(
                {"messages": messages}, config=config
            )
        yield from _stream()

    async def astream(self, messages, config=None):
        """异步流式调用，带 Semaphore 控制"""
        async with self.semaphore:
            async for chunk in self.agent.astream(
                {"messages": messages}, config=config
            ):
                yield chunk


    # --- 侧边栏 Token 展示 ---
    def sidebar_usage(self, budget=1_000_000):
        import streamlit as st
        today = str(date.today())
        usage = self.redis.hgetall(f"token_usage:{today}")
        total = int(usage.get("total", 0))
        ratio = total / budget * 100
        cost = self.get_daily_cost()
        st.sidebar.metric("今日 API 费用", f"¥{cost:.2f}")
        if ratio > 70:
            st.sidebar.warning(f"⚠️ Token 用量 {ratio:.0f}%")
        else:
            st.sidebar.caption(f"📊 今日 Token：{total:,}")

    @staticmethod
    def extract_token_usage_from_stream(chunks):
        total_input = 0
        total_output = 0
        for chunk in chunks:
            if "agent" not in chunk:
                continue
            messages = chunk["agent"].get("messages", [])
            for msg in messages:
                if hasattr(msg, "usage_metadata"):
                    usage = msg.usage_metadata
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
        return total_input, total_output

    def get_daily_cost(self):
        today = str(date.today())
        key = f"token_usage:{today}"
        usage = self.redis.hgetall(key)
        input_tok = int(usage.get("input", 0))
        output_tok = int(usage.get("output", 0))

        # 转换为百万 tokens
        input_millions = input_tok / 1_000_000
        output_millions = output_tok / 1_000_000

        cost = (input_millions * DEEPSEEK_V4_FLASH_INPUT_PRICE +
                output_millions * DEEPSEEK_V4_FLASH_OUTPUT_PRICE)

        # 返回人民币金额，保留两位小数
        return round(cost, 2)