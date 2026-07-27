# server/core/concurrency/__init__.py
# ── 并发控制管理器 ──
# 串联限流 → 排队 → 熔断 → 执行四层防护。
# 用法:
#   from server.core.concurrency import ConcurrencyManager, get_manager
#   manager = get_manager()
#   result = await manager.guard(user_id, ip, priority)

import asyncio
import hashlib
import json
import logging
import time
from typing import Optional, AsyncIterator

from .config import ConcurrencyConfig, DegradationLevel, Priority
from .rate_limiter import RateLimiter, RateLimitResult
from .request_queue import RequestQueue
from .circuit_breaker import CircuitBreaker, CircuitOpenError, create_circuits

logger = logging.getLogger(__name__)

# 模块级单例
_manager: Optional["ConcurrencyManager"] = None


class GuardResult:
    """准入检查结果。"""
    # action: "pass" | "rate_limited" | "queued" | "circuit_open" | "queue_full"
    def __init__(self, action: str, **kwargs):
        self.action = action
        self.__dict__.update(kwargs)


class ConcurrencyManager:
    """并发控制编排器。

    在 FastAPI lifespan 中初始化一次，挂在 app.state.concurrency 上。
    """

    def __init__(
        self,
        config: ConcurrencyConfig = None,
        redis_client=None,
    ):
        self.config = config or ConcurrencyConfig.from_env()
        self.redis = redis_client

        # 子组件
        self.rate_limiter = RateLimiter(self.config, redis_client)
        self.queue = RequestQueue(self.config, redis_client)
        self.circuits = create_circuits(self.config, redis_client)
        self.degradation_level = DegradationLevel.FULL

        # 统计
        self.total_guards = 0
        self.total_passed = 0
        self.total_limited = 0
        self.total_queued = 0
        self.total_circuit_blocked = 0

    # ── 主入口: guard 链 ──────────────────────────────

    async def guard(
        self,
        user_id: str = "anonymous",
        ip: str = "",
        priority: int = Priority.NORMAL,
        prompt: str = "",
    ) -> GuardResult:
        """执行准入检查链。

        返回 GuardResult:
          - action="pass" → 放行，可执行 LLM 调用
          - action="rate_limited" → 限流拒绝，含 retry_after
          - action="queued" → 已入队，含 queue_token
          - action="circuit_open" → 熔断打开，含 retry_after
          - action="queue_full" → 队列已满
        """
        self.total_guards += 1
        prompt_hash = _hash_prompt(prompt)

        # ── Layer 2: 熔断检查（先于限流，快速失败）──
        # ChromaDB 熔断器: 知识库不可用时立即拒绝（不走排队，直接返回）
        chroma_cb = self.circuits.get("chromadb")
        if chroma_cb and chroma_cb.is_open:
            self.total_circuit_blocked += 1
            self.degradation_level = DegradationLevel.DEGRADED
            logger.warning("ChromaDB circuit OPEN — rejecting request")
            return GuardResult(
                action="circuit_open",
                retry_after=self.config.cb_timeout_seconds,
                circuit_name="chromadb",
            )

        deepseek_cb = self.circuits.get("deepseek")
        if deepseek_cb and deepseek_cb.is_open:
            self.total_circuit_blocked += 1
            self.degradation_level = DegradationLevel.DEGRADED
            return GuardResult(
                action="circuit_open",
                retry_after=self.config.cb_timeout_seconds,
                circuit_name="deepseek",
            )

        # ── Layer 3: 限流 ──
        limit_result = await self.rate_limiter.check_all(user_id, ip)
        if not limit_result.allowed:
            self.total_limited += 1

            # 开启了排队 → 尝试入队
            if self.config.queue_enabled and not await self.queue.is_full():
                queue_result = await self.queue.enqueue(
                    user_id, priority=priority, prompt_hash=prompt_hash,
                )
                self.total_queued += 1
                return GuardResult(
                    action="queued",
                    queue_position=queue_result.position,
                    estimated_wait_s=queue_result.estimated_wait_s,
                    queue_token=queue_result.queue_token,
                    retry_after=limit_result.retry_after,
                )

            # 不能排队 → 直接拒绝
            if await self.queue.is_full():
                return GuardResult(
                    action="queue_full",
                    retry_after=limit_result.retry_after,
                )

            return GuardResult(
                action="rate_limited",
                retry_after=limit_result.retry_after,
                blocked_by=limit_result.blocked_by,
            )

        # 全部通过
        self.total_passed += 1
        if self.degradation_level == DegradationLevel.DEGRADED:
            self.degradation_level = DegradationLevel.FULL

        # 获取本地信号量槽位
        await self.rate_limiter.acquire_local()

        return GuardResult(action="pass")

    def release(self):
        """释放本地信号量槽位（Agent 执行完毕后调用）。"""
        self.rate_limiter.release_local()

    # ── 排队位置 SSE 流 ──────────────────────────────

    async def stream_queue_position(
        self,
        queue_token: str,
        cancel_event: asyncio.Event = None,
    ) -> AsyncIterator[str]:
        """SSE 生成器：推送排队位置变化。

        Args:
            queue_token: 入队时返回的令牌
            cancel_event: 外部取消信号

        Yields:
            SSE 格式字符串
        """
        interval = self.config.queue_poll_interval_seconds
        last_position = None
        timeout_at = time.monotonic() + self.config.queue_timeout_seconds

        while time.monotonic() < timeout_at:
            if cancel_event and cancel_event.is_set():
                yield _sse_queue("cancelled", position=0, message="排队已取消")
                return

            pos_info = await self.queue.get_position(queue_token)

            if pos_info is None:
                # 已出队（被 worker 消费了）
                yield _sse_queue("ready", position=0, message="轮到你了，正在处理...")
                return

            if pos_info.position != last_position:
                last_position = pos_info.position
                yield _sse_queue(
                    "waiting",
                    position=pos_info.position,
                    ahead=pos_info.ahead_count,
                    estimated_wait_s=pos_info.estimated_wait_s,
                )

            await asyncio.sleep(interval)

        # 超时
        await self.queue.dequeue(queue_token)
        yield _sse_queue("timeout", position=-1, message="排队超时，请稍后重试")

    # ── 记录 LLM 调用结果 ────────────────────────────

    async def record_result(self, success: bool):
        """Agent 执行完毕后记录结果，更新熔断器状态。"""
        cb = self.circuits.get("deepseek")
        if cb is None:
            return
        if success:
            await cb.on_success()
        else:
            await cb.on_failure()

    # ── 健康检查 ──────────────────────────────────────

    def get_health(self) -> dict:
        return {
            "degradation_level": self.degradation_level,
            "circuits": {name: cb.get_health() for name, cb in self.circuits.items()},
            "rate_limiter": self.rate_limiter.get_status() if self.redis else {"mode": "local"},
            "queue_depth": 0,  # 异步取，health 端点用同步方法
            "stats": {
                "total_guards": self.total_guards,
                "total_passed": self.total_passed,
                "total_limited": self.total_limited,
                "total_queued": self.total_queued,
                "total_circuit_blocked": self.total_circuit_blocked,
            },
        }


# ── 模块级便捷函数 ──────────────────────────────────

def init_manager(config: ConcurrencyConfig = None, redis_client=None) -> ConcurrencyManager:
    """初始化全局 ConcurrencyManager 单例。"""
    global _manager
    _manager = ConcurrencyManager(config=config, redis_client=redis_client)
    logger.info("ConcurrencyManager initialized (redis=%s)", "yes" if redis_client else "no")
    return _manager


def get_manager() -> Optional[ConcurrencyManager]:
    """获取全局 ConcurrencyManager 单例。"""
    return _manager


# ── 内部工具 ─────────────────────────────────────────

def _hash_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


def _sse_queue(status: str, **kwargs) -> str:
    """生成 queue SSE 事件字符串。"""
    data = {"status": status, **kwargs}
    return f"event: queue\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
