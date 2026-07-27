# server/core/concurrency/circuit_breaker.py
# ── 滑动窗口熔断器 ──
# 状态机: CLOSED → OPEN → HALF_OPEN → CLOSED
# Redis 不可用时自动降级为内存模式。

import logging
import time
from typing import Optional, Callable, Any

from .config import CircuitState, ConcurrencyConfig

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """熔断打开时抛出的异常。"""
    def __init__(self, name: str, retry_after: float = 30):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is OPEN, retry after {retry_after:.0f}s")


class CircuitBreaker:
    """滑动窗口熔断器。

    支持 Redis 分布式模式（多进程安全）和本地内存模式（开发环境降级）。
    """

    def __init__(
        self,
        name: str,
        config: ConcurrencyConfig,
        redis_client=None,
    ):
        self.name = name
        self.config = config
        self.redis = redis_client

        # 本地状态（Redis 模式下也缓存一份做快速判断）
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._opened_at: float = 0
        self._half_open_count = 0

        # 统计
        self.total_failures = 0
        self.total_successes = 0

    @property
    def state(self) -> str:
        self._maybe_transition()
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    # ── 公共接口 ──────────────────────────────────────

    async def before_call(self) -> bool:
        """在调用 LLM 之前检查。返回 False 表示熔断打开，应快速失败。"""
        if not self.config.circuit_breaker_enabled:
            return True

        self._maybe_transition()

        if self._state == CircuitState.OPEN:
            logger.warning("Circuit '%s' OPEN — fast-failing request", self.name)
            return False

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_count >= self.config.cb_half_open_max:
                logger.warning("Circuit '%s' HALF_OPEN at max probes (%d) — rejecting",
                               self.name, self._half_open_count)
                return False
            self._half_open_count += 1
            logger.info("Circuit '%s' HALF_OPEN — allowing probe %d/%d",
                        self.name, self._half_open_count, self.config.cb_half_open_max)

        return True

    async def on_success(self):
        """调用成功的回调。"""
        self.total_successes += 1

        if self._state == CircuitState.HALF_OPEN:
            # 所有探测请求成功 → 关闭熔断
            if self._half_open_count >= self.config.cb_half_open_max:
                self._transition_to(CircuitState.CLOSED)
                logger.info("Circuit '%s' HALF_OPEN probes all succeeded → CLOSED", self.name)

    async def on_failure(self, error: Exception = None):
        """调用失败的回调。"""
        self.total_failures += 1
        now = time.monotonic()

        # 滑动窗口：丢弃窗口外的旧失败
        window = self.config.cb_window_seconds
        if now - self._last_failure_time > window:
            self._failure_count = 0

        self._failure_count += 1
        self._last_failure_time = now

        if self._state == CircuitState.HALF_OPEN:
            # 半开状态任何失败立即重新打开
            self._transition_to(CircuitState.OPEN)
            logger.warning("Circuit '%s' HALF_OPEN probe failed → OPEN (error: %s)",
                           self.name, str(error)[:100] if error else "unknown")
            return

        if self._failure_count >= self.config.cb_failure_threshold:
            self._transition_to(CircuitState.OPEN)
            logger.warning("Circuit '%s' failure threshold reached (%d/%d) → OPEN",
                           self.name, self._failure_count, self.config.cb_failure_threshold)

    async def call(
        self,
        fn: Callable,
        fallback: Callable = None,
        *args,
        **kwargs,
    ) -> Any:
        """包装一个异步函数调用，自动处理熔断和降级。

        Args:
            fn: 主调用异步函数
            fallback: 降级函数（熔断打开或主调用失败时调用）
            *args, **kwargs: 传递给 fn 和 fallback

        Returns:
            fn 或 fallback 的返回值

        Raises:
            CircuitOpenError: 熔断打开且没有 fallback 时
        """
        if not await self.before_call():
            if fallback:
                return await fallback(*args, **kwargs)
            raise CircuitOpenError(self.name, self.config.cb_timeout_seconds)

        try:
            result = await fn(*args, **kwargs)
            await self.on_success()
            return result
        except Exception as e:
            await self.on_failure(e)
            if fallback:
                return await fallback(*args, **kwargs)
            raise

    def get_health(self) -> dict:
        """返回熔断器健康状态（供 /health 端点）。"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "opened_at": self._opened_at if self._state == CircuitState.OPEN else None,
        }

    # ── 内部方法 ──────────────────────────────────────

    def _maybe_transition(self):
        """检查是否应该从 OPEN 转入 HALF_OPEN。"""
        if self._state != CircuitState.OPEN:
            return
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self.config.cb_timeout_seconds:
            self._transition_to(CircuitState.HALF_OPEN)
            logger.info("Circuit '%s' timeout elapsed (%.1fs) → HALF_OPEN", self.name, elapsed)

    def _transition_to(self, new_state: str):
        """状态转换。"""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_count = 0
        elif new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_count = 0

        # 同步到 Redis（如果可用）
        if self.redis:
            try:
                pipe = self.redis.pipeline()
                key = f"cb:{self.name}"
                pipe.hset(key, mapping={
                    "state": new_state,
                    "failure_count": str(self._failure_count),
                    "opened_at": str(self._opened_at),
                    "updated_at": str(time.time()),
                })
                pipe.expire(key, self.config.cb_timeout_seconds * 3)
                pipe.execute()
            except Exception as e:
                logger.debug("Redis sync for circuit '%s' failed: %s", self.name, e)


# ── 预置熔断器实例 ───────────────────────────────────

def create_circuits(config: ConcurrencyConfig, redis_client=None) -> dict[str, CircuitBreaker]:
    """创建所有预置熔断器实例。"""
    return {
        "deepseek": CircuitBreaker("deepseek", config, redis_client),
        "chromadb": CircuitBreaker("chromadb", config, redis_client),
    }
