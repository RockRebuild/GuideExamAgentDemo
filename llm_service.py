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

# ============================================================
# API 定价（¥ / 百万 tokens）
# 最后更新: 2026-07
# ============================================================

# DeepSeek 模型定价  https://api-docs.deepseek.com/zh-cn/quick_start/pricing
PRICES = {
    # Agent 主模型
    "deepseek-v4-flash":     {"input": 1.0,  "output": 2.0},   # ¥1.0 / 2.0 每百万 token
    # RAGAS Judge 模型（pro 版精度更高）
    "deepseek-v4-pro":      {"input": 2.0,  "output": 8.0},   # ¥2.0 / 8.0 每百万 token
}

# DashScope 嵌入模型定价  https://help.aliyun.com/zh/model-studio
DASHSCOPE_EMBEDDING_PRICE = 0.0007  # ¥ / 千 token → ¥0.7 / 百万 token


class LLMService:
    """LLM API 统一服务层：重试、超时、并发控制、分模型成本记录"""

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

    # ============================================================
    # Token 提取
    # ============================================================
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

    # ============================================================
    # 分模型成本记录（Redis 按日聚合）
    # ============================================================
    def _record_usage(self, input_tokens: int, output_tokens: int,
                       model: str = "deepseek-v4-flash"):
        """写入 Redis，按日 + 模型聚合"""
        today = str(date.today())
        key = f"token_usage:{today}:{model}"
        current = self.redis.hgetall(key)
        new_input = int(current.get("input", 0)) + input_tokens
        new_output = int(current.get("output", 0)) + output_tokens
        self.redis.hset(key, mapping={
            "input": new_input,
            "output": new_output,
            "total": new_input + new_output,
        })
        self.redis.expire(key, 60 * 60 * 48)

    @staticmethod
    def record_embedding_usage(redis_client: redis.Redis,
                                token_count: int,
                                model: str = "text-embedding-v4"):
        """静态方法：记录嵌入模型的 token 用量（无 input/output 区分）"""
        if token_count <= 0:
            return
        today = str(date.today())
        key = f"token_usage:{today}:{model}"
        current = redis_client.hgetall(key)
        new_total = int(current.get("total", 0)) + token_count
        redis_client.hset(key, mapping={
            "input": new_total,   # 嵌入模型只有 input
            "output": 0,
            "total": new_total,
        })
        redis_client.expire(key, 60 * 60 * 48)

    # ============================================================
    # 同步调用（带重试 + 成本记录）
    # ============================================================
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

    # ============================================================
    # 异步调用（带 Semaphore + 重试 + 成本记录）
    # ============================================================
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

    # ============================================================
    # 流式调用（带重试）
    # ============================================================
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

    # ============================================================
    # 流式 Token 提取
    # ============================================================
    @staticmethod
    def extract_token_usage_from_stream(chunks) -> tuple[int, int]:
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

    # ============================================================
    # 费用计算
    # ============================================================
    @staticmethod
    def _calc_cost(input_tokens: int, output_tokens: int,
                    model: str = "deepseek-v4-flash") -> float:
        """根据模型定价计算单次调用费用（¥）"""
        price = PRICES.get(model, {"input": 1.0, "output": 2.0})
        input_m = input_tokens / 1_000_000
        output_m = output_tokens / 1_000_000
        return round(input_m * price["input"] + output_m * price["output"], 4)

    @staticmethod
    def _calc_embedding_cost(token_count: int) -> float:
        """计算嵌入模型费用（¥）"""
        return round(token_count / 1_000_000 * DASHSCOPE_EMBEDDING_PRICE, 4)

    @staticmethod
    def get_daily_cost_detail(redis_client: redis.Redis) -> dict:
        """
        返回当日各模型费用明细:
        {
          "models": {"deepseek-v4-flash": {...}, ...},
          "total_cost": float
        }
        """
        today = str(date.today())
        result = {"models": {}, "total_cost": 0.0}
        total_cost = 0.0

        for model, price in PRICES.items():
            key = f"token_usage:{today}:{model}"
            usage = redis_client.hgetall(key)
            if not usage:
                continue
            inp = int(usage.get("input", 0))
            out = int(usage.get("output", 0))
            cost = LLMService._calc_cost(inp, out, model)
            result["models"][model] = {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out,
                "cost": cost,
            }
            total_cost += cost

        # 嵌入模型
        emb_key = f"token_usage:{today}:text-embedding-v4"
        emb_usage = redis_client.hgetall(emb_key)
        if emb_usage:
            emb_tokens = int(emb_usage.get("total", 0))
            emb_cost = LLMService._calc_embedding_cost(emb_tokens)
            result["models"]["text-embedding-v4"] = {
                "input_tokens": emb_tokens,
                "output_tokens": 0,
                "total_tokens": emb_tokens,
                "cost": emb_cost,
            }
            total_cost += emb_cost

        result["total_cost"] = round(total_cost, 4)
        return result

    @staticmethod
    def get_daily_cost(redis_client: redis.Redis) -> float:
        """获取当日总费用（¥），向后兼容"""
        return LLMService.get_daily_cost_detail(redis_client)["total_cost"]

    # ============================================================
    # 侧边栏展示
    # ============================================================
    def sidebar_usage(self, budget: float = 5.0):
        """侧边栏显示当日 API 用量与费用明细"""
        import streamlit as st

        detail = self.get_daily_cost_detail(self.redis)
        models = detail["models"]
        total_cost = detail["total_cost"]

        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 API 费用")

        # 总额
        ratio = total_cost / budget * 100 if budget > 0 else 0
        if ratio > 70:
            st.sidebar.warning(f"⚠️ 今日 ¥{total_cost:.2f} / ¥{budget:.0f} ({ratio:.0f}%)")
        else:
            st.sidebar.metric("今日费用", f"¥{total_cost:.4f}")

        # 分模型明细（折叠）
        if models:
            with st.sidebar.expander("📋 费用明细", expanded=False):
                for model, info in sorted(models.items()):
                    cost = info["cost"]
                    tokens = info["total_tokens"]
                    if cost < 0.0001 and tokens == 0:
                        continue
                    short_name = model.replace("deepseek-", "").replace("text-embedding-", "")
                    st.caption(
                        f"**{short_name}**  "
                        f"{tokens:,} tokens  →  ¥{cost:.4f}"
                    )

        # 历史趋势（最近 5 天）
        with st.sidebar.expander("📈 近 5 日趋势", expanded=False):
            from datetime import timedelta
            today = date.today()
            total_5d = 0.0
            for i in range(5):
                d = today - timedelta(days=i)
                day_cost = 0.0
                for model in PRICES:
                    key = f"token_usage:{d}:{model}"
                    usage = self.redis.hgetall(key)
                    if usage:
                        inp = int(usage.get("input", 0))
                        out = int(usage.get("output", 0))
                        day_cost += self._calc_cost(inp, out, model)
                emb_key = f"token_usage:{d}:text-embedding-v4"
                emb_usage = self.redis.hgetall(emb_key)
                if emb_usage:
                    day_cost += self._calc_embedding_cost(int(emb_usage.get("total", 0)))
                marker = "← 今天" if i == 0 else ""
                st.caption(f"{d}  ¥{day_cost:.4f} {marker}")
                total_5d += day_cost
            st.caption(f"**5 日合计: ¥{total_5d:.4f}**")
