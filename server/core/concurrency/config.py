# server/core/concurrency/config.py
# ── 并发控制配置 ──
# 全部可通过环境变量覆盖，Redis 不可用时自动降级为本地模式。

import os
from dataclasses import dataclass, field
from enum import IntEnum


# ── 优先级 ──

class Priority(IntEnum):
    HIGH = 0      # 付费用户 / 内部测试 / HITL 恢复
    NORMAL = 5    # 普通用户
    LOW = 10      # 爬虫 / 高频用户


# ── 熔断状态 ──

class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ── 降级级别 ──

class DegradationLevel:
    FULL = "full"              # 全功能
    THROTTLED = "throttled"    # 限流收紧
    DEGRADED = "degraded"      # 熔断打开，仅缓存
    MINIMAL = "minimal"        # 仅健康检查 + 静态文件


@dataclass
class ConcurrencyConfig:
    """并发控制全局配置。"""

    # ── 限流 ──
    rate_limit_enabled: bool = True
    global_rpm: int = 50               # DeepSeek API 全局每分钟
    global_burst: int = 10             # 全局突发容忍
    global_tpm: int = 100_000          # DeepSeek 全局每分钟 Token（保留字段）
    per_user_rpm: int = 5              # 每用户每分钟
    per_user_burst: int = 2
    per_ip_rpm: int = 10               # 每 IP 每分钟
    per_ip_burst: int = 3

    # ── 排队 ──
    queue_enabled: bool = True
    queue_max_size: int = 200          # 超过则 503
    queue_timeout_seconds: int = 120   # 排队最大等待
    queue_poll_interval_seconds: float = 1.5

    # ── 熔断 ──
    circuit_breaker_enabled: bool = True
    cb_failure_threshold: int = 5      # 滑动窗口内错误数 → OPEN
    cb_window_seconds: int = 60        # 滑动窗口大小
    cb_timeout_seconds: int = 30       # OPEN → HALF_OPEN 等待
    cb_half_open_max: int = 1          # HALF_OPEN 允许的探测请求数

    # ── 本地降级（Redis 不可用时） ──
    local_max_concurrency: int = 5     # asyncio.Semaphore

    @classmethod
    def from_env(cls) -> "ConcurrencyConfig":
        """从环境变量加载配置，未设置时使用默认值。"""
        return cls(
            rate_limit_enabled=_env_bool("CONCURRENCY_RATE_LIMIT_ENABLED", True),
            global_rpm=_env_int("CONCURRENCY_GLOBAL_RPM", 50),
            global_burst=_env_int("CONCURRENCY_GLOBAL_BURST", 10),
            global_tpm=_env_int("CONCURRENCY_GLOBAL_TPM", 100_000),
            per_user_rpm=_env_int("CONCURRENCY_PER_USER_RPM", 5),
            per_user_burst=_env_int("CONCURRENCY_PER_USER_BURST", 2),
            per_ip_rpm=_env_int("CONCURRENCY_PER_IP_RPM", 10),
            per_ip_burst=_env_int("CONCURRENCY_PER_IP_BURST", 3),
            queue_enabled=_env_bool("CONCURRENCY_QUEUE_ENABLED", True),
            queue_max_size=_env_int("CONCURRENCY_QUEUE_MAX_SIZE", 200),
            queue_timeout_seconds=_env_int("CONCURRENCY_QUEUE_TIMEOUT_SECONDS", 120),
            queue_poll_interval_seconds=_env_float("CONCURRENCY_QUEUE_POLL_INTERVAL", 1.5),
            circuit_breaker_enabled=_env_bool("CONCURRENCY_CB_ENABLED", True),
            cb_failure_threshold=_env_int("CONCURRENCY_CB_FAILURE_THRESHOLD", 5),
            cb_window_seconds=_env_int("CONCURRENCY_CB_WINDOW_SECONDS", 60),
            cb_timeout_seconds=_env_int("CONCURRENCY_CB_TIMEOUT_SECONDS", 30),
            cb_half_open_max=_env_int("CONCURRENCY_CB_HALF_OPEN_MAX", 1),
            local_max_concurrency=_env_int("CONCURRENCY_LOCAL_MAX_CONCURRENCY", 5),
        )


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, ""))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, ""))
    except (ValueError, TypeError):
        return default
