# server/core/concurrency/rate_limiter.py
# ── Redis Token Bucket 分布式限流 + 本地 Semaphore 降级 ──
# GCRA (Generic Cell Rate Algorithm)：O(1) 内存，边界平滑，无窗口尖刺问题。

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .config import ConcurrencyConfig

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after: float       # 建议重试秒数（0 表示通过）
    remaining: int          # 剩余配额
    blocked_by: str         # "global" | "user" | "ip" | "none"


class RateLimiter:
    """基于 Redis GCRA 的分布式限流器。

    Redis 不可用时自动降级为本地 asyncio.Semaphore（单进程不准确但比没有好）。
    """

    def __init__(self, config: ConcurrencyConfig, redis_client=None):
        self.config = config
        self.redis = redis_client
        self._lua_sha = {}       # {name: sha}
        self._local_buckets: dict[str, dict] = {}  # 本地降级用
        self._local_semaphore = asyncio.Semaphore(config.local_max_concurrency)
        self._scripts_registered = False

    async def _ensure_scripts(self):
        """懒注册 Lua 脚本（Redis 连接可能晚于对象创建）。"""
        if self._scripts_registered or self.redis is None:
            return
        try:
            from .redis_scripts import register_scripts
            self._lua_sha = register_scripts(self.redis)
            self._scripts_registered = True
        except Exception as e:
            logger.warning("Lua script registration failed: %s", e)

    # ── 公共接口 ──────────────────────────────────────

    async def check_all(
        self,
        user_id: str = "",
        ip: str = "",
    ) -> RateLimitResult:
        """三层限流联合检查：global + user + ip，返回最严格的限制。"""
        if not self.config.rate_limit_enabled:
            return RateLimitResult(allowed=True, retry_after=0, remaining=-1, blocked_by="none")

        if self.redis is None:
            return await self._check_local()

        await self._ensure_scripts()

        if "multi_check" not in self._lua_sha:
            return await self._check_local()

        cfg = self.config
        try:
            global_key = "rl:global:deepseek"
            user_key = f"rl:user:{user_id}" if user_id else ""
            ip_key = f"rl:ip:{_hash_ip(ip)}" if ip else ""

            result = await self.redis.evalsha(
                self._lua_sha["multi_check"],
                3,
                global_key, user_key, ip_key,
                # global: period, burst, ttl
                60.0 / cfg.global_rpm, cfg.global_burst, cfg.global_rpm * 2,
                # user: period, burst, ttl
                60.0 / cfg.per_user_rpm, cfg.per_user_burst, cfg.per_user_rpm * 2,
                # ip: period, burst, ttl
                60.0 / cfg.per_ip_rpm, cfg.per_ip_burst, cfg.per_ip_rpm * 2,
            )

            allowed = bool(result[0])
            retry_after = float(result[1]) if len(result) > 1 else 0
            blocked_by = result[2].decode() if len(result) > 2 and isinstance(result[2], bytes) else str(result[2]) if len(result) > 2 else "none"

            return RateLimitResult(
                allowed=allowed,
                retry_after=retry_after,
                remaining=-1,
                blocked_by=blocked_by,
            )
        except Exception as e:
            logger.warning("Redis rate limit check failed, falling back to local: %s", e)
            return await self._check_local()

    async def get_status(self) -> dict:
        """返回各桶当前状态（供 /health 端点使用）。"""
        if self.redis is None:
            return {"mode": "local", "max_concurrency": self.config.local_max_concurrency}
        try:
            keys = ["rl:global:deepseek"]
            status = {}
            for key in keys:
                val = await self.redis.get(key)
                status[key] = float(val) if val else 0
            return {"mode": "redis", "buckets": status}
        except Exception:
            return {"mode": "redis_error"}

    # ── 本地降级 ──────────────────────────────────────

    async def _check_local(self) -> RateLimitResult:
        """本地 Semaphore 降级（单进程，不准确但比没有好）。"""
        if self._local_semaphore.locked():
            # 所有槽位都被占用
            return RateLimitResult(
                allowed=False,
                retry_after=2.0,
                remaining=0,
                blocked_by="local",
            )
        return RateLimitResult(allowed=True, retry_after=0, remaining=-1, blocked_by="none")

    async def acquire_local(self):
        """获取本地信号量槽位。调用方在请求结束时必须 release_local()。"""
        await self._local_semaphore.acquire()

    def release_local(self):
        """释放本地信号量槽位。"""
        try:
            self._local_semaphore.release()
        except ValueError:
            pass  # 可能已经释放过


def _hash_ip(ip: str) -> str:
    """IP 哈希脱敏。"""
    if not ip:
        return ""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]
