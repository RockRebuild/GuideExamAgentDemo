# server/core/concurrency/request_queue.py
# ── 优先级请求排队 ──
# Redis Sorted Set 实现，优先级高的先出队。
# Redis 不可用时降级为本地 asyncio.Queue。

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from .config import ConcurrencyConfig, Priority

logger = logging.getLogger(__name__)


@dataclass
class QueuePosition:
    """排队位置信息。"""
    position: int           # 当前位置（0=轮到你了）
    ahead_count: int        # 前方人数
    estimated_wait_s: float # 预计等待秒数


@dataclass
class EnqueueResult:
    """入队结果。"""
    request_id: str
    position: int
    estimated_wait_s: float
    queue_token: str


class RequestQueue:
    """基于 Redis Sorted Set 的优先级请求队列。

    score = priority * 1e12 + timestamp（越小越优先出队）
    """

    QUEUE_KEY = "q:requests"        # Sorted Set: member=rid, score=priority*1e12+timestamp
    META_PREFIX = "q:meta:"         # Hash: user_id, priority, entered_at, prompt_hash

    def __init__(self, config: ConcurrencyConfig, redis_client=None):
        self.config = config
        self.redis = redis_client
        # 本地降级
        self._local_queue: asyncio.Queue = asyncio.Queue()
        self._local_position: dict[str, float] = {}  # rid -> entered_at

    @property
    def enabled(self) -> bool:
        return self.config.queue_enabled

    # ── 入队 / 出队 ──────────────────────────────────

    async def enqueue(
        self,
        user_id: str,
        priority: int = Priority.NORMAL,
        prompt_hash: str = "",
    ) -> EnqueueResult:
        """将请求加入优先级队列。返回排队令牌。

        Args:
            user_id: 用户标识
            priority: Priority 枚举值（越小越优先）
            prompt_hash: 问题哈希（用于缓存预热）

        Returns:
            EnqueueResult 含排队位置和令牌
        """
        request_id = f"{user_id}:{uuid.uuid4().hex[:8]}"
        now = time.time()
        score = priority * 1e12 + now  # priority 低 = 排在前面

        if self.redis and self.config.queue_enabled:
            try:
                pipe = self.redis.pipeline()
                # 入队
                pipe.zadd(self.QUEUE_KEY, {request_id: score})
                # 元数据
                meta = {
                    "user_id": user_id,
                    "priority": str(priority),
                    "entered_at": str(now),
                    "prompt_hash": prompt_hash,
                }
                pipe.hset(f"{self.META_PREFIX}{request_id}", mapping=meta)
                pipe.expire(f"{self.META_PREFIX}{request_id}", self.config.queue_timeout_seconds * 2)
                # 获取当前位置
                pipe.zrank(self.QUEUE_KEY, request_id)
                pipe.zcard(self.QUEUE_KEY)
                results = await pipe.execute()
                position = (results[-2] or 0) + 1   # zrank (0-based → 1-based)
                depth = results[-1] or 0
            except Exception as e:
                logger.warning("Redis enqueue failed, using local: %s", e)
                return await self._enqueue_local(user_id, request_id)
        else:
            return await self._enqueue_local(user_id, request_id)

        estimated_wait = self._estimate_wait(position)

        logger.info("Request %s enqueued: position=%d, depth=%d, wait=%.1fs",
                     request_id, position, depth, estimated_wait)

        return EnqueueResult(
            request_id=request_id,
            position=position,
            estimated_wait_s=estimated_wait,
            queue_token=request_id,  # 简化：token = request_id
        )

    async def get_position(self, request_id: str) -> Optional[QueuePosition]:
        """获取排队位置。"""
        if self.redis and self.config.queue_enabled:
            try:
                pipe = self.redis.pipeline()
                pipe.zrank(self.QUEUE_KEY, request_id)
                pipe.zcard(self.QUEUE_KEY)
                rank, depth = await pipe.execute()

                if rank is None:
                    # 不在队列中（已出队或超时）
                    return None

                position = rank + 1
                return QueuePosition(
                    position=position,
                    ahead_count=rank,
                    estimated_wait_s=self._estimate_wait(position),
                )
            except Exception as e:
                logger.warning("Redis get_position failed: %s", e)

        # 本地降级
        if request_id in self._local_position:
            pos = len(self._local_position)
            return QueuePosition(position=pos, ahead_count=pos - 1,
                                 estimated_wait_s=pos * 1.5)
        return None

    async def dequeue(self, request_id: str):
        """移除排队（用户取消或超时）。"""
        if self.redis and self.config.queue_enabled:
            try:
                pipe = self.redis.pipeline()
                pipe.zrem(self.QUEUE_KEY, request_id)
                pipe.delete(f"{self.META_PREFIX}{request_id}")
                await pipe.execute()
            except Exception as e:
                logger.warning("Redis dequeue failed: %s", e)
        # 本地降级
        self._local_position.pop(request_id, None)

    async def get_depth(self) -> int:
        """获取当前队列深度。"""
        if self.redis and self.config.queue_enabled:
            try:
                return await self.redis.zcard(self.QUEUE_KEY) or 0
            except Exception:
                pass
        return len(self._local_position)

    async def is_full(self) -> bool:
        """队列是否已满。"""
        if not self.config.queue_enabled:
            return False
        depth = await self.get_depth()
        return depth >= self.config.queue_max_size

    # ── 内部方法 ──────────────────────────────────────

    def _estimate_wait(self, position: int) -> float:
        """估算等待秒数：前方人数 × 平均每次耗时 / 并发槽位数。"""
        avg_call_time = 8.0    # LLM 平均响应时间（秒）
        slots = self.config.local_max_concurrency
        return round(position * avg_call_time / slots, 1)

    async def _enqueue_local(self, user_id: str, request_id: str) -> EnqueueResult:
        """本地异步队列降级。"""
        now = time.time()
        self._local_position[request_id] = now
        pos = len(self._local_position)

        # 设置超时自动清理
        async def _auto_dequeue():
            await asyncio.sleep(self.config.queue_timeout_seconds)
            self._local_position.pop(request_id, None)

        asyncio.create_task(_auto_dequeue())

        return EnqueueResult(
            request_id=request_id,
            position=pos,
            estimated_wait_s=self._estimate_wait(pos),
            queue_token=request_id,
        )
