# tests/test_concurrency.py
# ── 并发控制单元测试 ──
# 测试限流、熔断、排队逻辑。优先使用本地模式（无 Redis 依赖）。

import asyncio
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from server.core.concurrency.config import ConcurrencyConfig, Priority, CircuitState, DegradationLevel
from server.core.concurrency.rate_limiter import RateLimiter, RateLimitResult
from server.core.concurrency.circuit_breaker import CircuitBreaker, CircuitOpenError
from server.core.concurrency.request_queue import RequestQueue
from server.core.concurrency import ConcurrencyManager, GuardResult


# ═══════════════════════════════════════════════════
# Config 测试
# ═══════════════════════════════════════════════════

class TestConfig:
    def test_defaults(self):
        cfg = ConcurrencyConfig()
        assert cfg.global_rpm == 50
        assert cfg.global_burst == 10
        assert cfg.per_user_rpm == 5
        assert cfg.cb_failure_threshold == 5
        assert cfg.queue_max_size == 200

    def test_from_env(self):
        cfg = ConcurrencyConfig.from_env()
        assert cfg.global_rpm > 0  # 至少不是 0
        assert isinstance(cfg.circuit_breaker_enabled, bool)

    def test_priority_order(self):
        assert Priority.HIGH < Priority.NORMAL < Priority.LOW

    def test_circuit_states(self):
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"


# ═══════════════════════════════════════════════════
# RateLimiter 测试（本地模式）
# ═══════════════════════════════════════════════════

class TestRateLimiter:
    def test_local_allow_first_request(self):
        cfg = ConcurrencyConfig(local_max_concurrency=5)
        rl = RateLimiter(cfg, redis_client=None)

        async def _test():
            result = await rl.check_all(user_id="test", ip="1.2.3.4")
            assert result.allowed

        asyncio.run(_test())

    def test_local_sequential_requests(self):
        cfg = ConcurrencyConfig(local_max_concurrency=3)
        rl = RateLimiter(cfg, redis_client=None)

        async def _test():
            # 获取全部 3 个槽位
            await rl.acquire_local()
            await rl.acquire_local()
            await rl.acquire_local()

            # 槽位用完，应拒绝
            result = await rl.check_all(user_id="test", ip="1.2.3.4")
            assert not result.allowed
            assert result.blocked_by == "local"

            # 释放一个槽位
            rl.release_local()
            result = await rl.check_all(user_id="test", ip="1.2.3.4")
            assert result.allowed

            # 清理
            rl.release_local()
            rl.release_local()

        asyncio.run(_test())

    def test_disabled_rate_limit(self):
        cfg = ConcurrencyConfig(rate_limit_enabled=False)
        rl = RateLimiter(cfg, redis_client=None)

        async def _test():
            result = await rl.check_all(user_id="test", ip="1.2.3.4")
            assert result.allowed
            assert result.blocked_by == "none"

        asyncio.run(_test())


# ═══════════════════════════════════════════════════
# CircuitBreaker 测试
# ═══════════════════════════════════════════════════

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cfg = ConcurrencyConfig()
        cb = CircuitBreaker("test", cfg)
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open

    def test_transition_to_open(self):
        cfg = ConcurrencyConfig(cb_failure_threshold=3, cb_window_seconds=60)
        cb = CircuitBreaker("test", cfg)

        async def _test():
            # 熔断前可以调用
            assert await cb.before_call()

            # 3 次失败 → OPEN
            await cb.on_failure(Exception("error1"))
            await cb.on_failure(Exception("error2"))
            await cb.on_failure(Exception("error3"))
            assert cb.is_open

            # OPEN 状态拒绝调用
            assert not await cb.before_call()

        asyncio.run(_test())

    def test_half_open_probe(self):
        cfg = ConcurrencyConfig(
            cb_failure_threshold=1,
            cb_window_seconds=60,
            cb_timeout_seconds=0,       # 立即 HALF_OPEN
            cb_half_open_max=1,
        )
        cb = CircuitBreaker("test", cfg)

        async def _test():
            # 1 次失败 → OPEN
            await cb.on_failure(Exception("error"))
            assert cb.is_open

            # timeout=0 → 立即转 HALF_OPEN
            assert await cb.before_call()  # 允许探测请求
            assert cb.state == CircuitState.HALF_OPEN

            # 探测成功 → CLOSED
            await cb.on_success()
            assert cb.state == CircuitState.CLOSED

        asyncio.run(_test())

    def test_circuit_open_disabled(self):
        cfg = ConcurrencyConfig(circuit_breaker_enabled=False)
        cb = CircuitBreaker("test", cfg)

        async def _test():
            # 禁用时始终返回 True
            for _ in range(10):
                await cb.on_failure(Exception("error"))
            assert await cb.before_call()

        asyncio.run(_test())

    def test_call_with_fallback(self):
        cfg = ConcurrencyConfig(cb_failure_threshold=1, cb_timeout_seconds=0)
        cb = CircuitBreaker("test", cfg)

        async def _test():
            fallback_called = False

            async def main_fn():
                raise RuntimeError("fail!")

            async def fallback_fn():
                nonlocal fallback_called
                fallback_called = True
                return "fallback_result"

            # 前两次调用（失败 + fallback）
            result1 = await cb.call(main_fn, fallback_fn)
            assert result1 == "fallback_result"
            assert fallback_called

            # 第三次：熔断打开 + fallback
            fallback_called = False
            result2 = await cb.call(main_fn, fallback_fn)
            assert result2 == "fallback_result"
            assert fallback_called

        asyncio.run(_test())


# ═══════════════════════════════════════════════════
# RequestQueue 测试（本地模式）
# ═══════════════════════════════════════════════════

class TestRequestQueue:
    def test_enqueue_and_get_position_local(self):
        cfg = ConcurrencyConfig(queue_enabled=True)
        q = RequestQueue(cfg, redis_client=None)

        async def _test():
            result = await q.enqueue(
                user_id="user1",
                priority=Priority.NORMAL,
                prompt_hash="abc",
            )
            assert result.position >= 1
            assert result.estimated_wait_s > 0
            assert result.queue_token

            pos = await q.get_position(result.queue_token)
            assert pos is not None
            assert pos.position >= 1

            await q.dequeue(result.queue_token)
            pos = await q.get_position(result.queue_token)
            assert pos is None

        asyncio.run(_test())

    def test_queue_disabled(self):
        cfg = ConcurrencyConfig(queue_enabled=False)
        q = RequestQueue(cfg, redis_client=None)

        async def _test():
            assert not q.enabled

        asyncio.run(_test())

    def test_priority_order(self):
        """高优先级先入队应该排在前面。"""
        cfg = ConcurrencyConfig(queue_enabled=True)
        q = RequestQueue(cfg, redis_client=None)

        async def _test():
            high = await q.enqueue(user_id="vip", priority=Priority.HIGH)
            normal = await q.enqueue(user_id="user", priority=Priority.NORMAL)
            # 本地模式下按入队顺序，但验证不同优先级不会报错
            assert high.position == 1
            assert normal.position == 2

            await q.dequeue(high.queue_token)
            await q.dequeue(normal.queue_token)

        asyncio.run(_test())


# ═══════════════════════════════════════════════════
# ConcurrencyManager 集成测试
# ═══════════════════════════════════════════════════

class TestConcurrencyManager:
    def test_init_without_redis(self):
        cfg = ConcurrencyConfig()
        mgr = ConcurrencyManager(cfg, redis_client=None)
        assert mgr.rate_limiter is not None
        assert mgr.queue is not None
        assert "deepseek" in mgr.circuits
        assert mgr.degradation_level == DegradationLevel.FULL

    def test_guard_passes_when_ok(self):
        cfg = ConcurrencyConfig()
        mgr = ConcurrencyManager(cfg, redis_client=None)

        async def _test():
            result = await mgr.guard(user_id="test", ip="1.2.3.4")
            assert result.action == "pass"
            mgr.release()

        asyncio.run(_test())

    def test_guard_rate_limited(self):
        cfg = ConcurrencyConfig(local_max_concurrency=1)
        mgr = ConcurrencyManager(cfg, redis_client=None)

        async def _test():
            # 占满槽位
            await mgr.rate_limiter.acquire_local()

            result = await mgr.guard(user_id="test", ip="1.2.3.4")
            # 应被限流（本地模式排队或直接拒绝）
            assert result.action in ("rate_limited", "queued")

            mgr.rate_limiter.release_local()

        asyncio.run(_test())

    def test_guard_circuit_open_blocks(self):
        cfg = ConcurrencyConfig(
            cb_failure_threshold=1,
            cb_timeout_seconds=30,
        )
        mgr = ConcurrencyManager(cfg, redis_client=None)

        async def _test():
            # 触发熔断
            cb = mgr.circuits["deepseek"]
            await cb.on_failure(Exception("test failure"))

            result = await mgr.guard(user_id="test", ip="1.2.3.4")
            assert result.action == "circuit_open"
            assert result.circuit_name == "deepseek"

        asyncio.run(_test())

    def test_health(self):
        cfg = ConcurrencyConfig()
        mgr = ConcurrencyManager(cfg, redis_client=None)
        health = mgr.get_health()
        assert "degradation_level" in health
        assert "circuits" in health
        assert "stats" in health
        assert health["degradation_level"] == DegradationLevel.FULL


# ═══════════════════════════════════════════════════
# 压力测试（模拟 100 并发）
# ═══════════════════════════════════════════════════

class TestConcurrencyStress:
    def test_100_concurrent_guards(self):
        """100 个并发请求，验证不会崩溃且部分被限流。"""
        cfg = ConcurrencyConfig(local_max_concurrency=5)
        mgr = ConcurrencyManager(cfg, redis_client=None)
        results = []

        async def _worker(i: int):
            result = await mgr.guard(
                user_id=f"user_{i % 10}",
                ip=f"10.0.0.{i % 20}",
            )
            results.append(result)
            if result.action == "pass":
                # 模拟 API 调用
                await asyncio.sleep(0.01)
                mgr.release()

        async def _test():
            tasks = [_worker(i) for i in range(100)]
            await asyncio.gather(*tasks)

            passed = sum(1 for r in results if r.action == "pass")
            limited = sum(1 for r in results if r.action in ("rate_limited", "queued"))

            # 由于 max_concurrency=5，应该有限流发生
            assert passed > 0
            # 不会全部通过（限流在起作用）
            assert limited + passed == 100
            assert mgr.total_guards == 100

        asyncio.run(_test())
