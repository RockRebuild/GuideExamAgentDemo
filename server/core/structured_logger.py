# server/core/structured_logger.py
# ── 结构化日志 + 请求级 Trace ID ──
#
# 原理:
#   使用 Python contextvars 实现请求级的 Trace ID 自动传播。
#   contextvars 在 asyncio 和线程池中都能正确传递上下文，
#   不需要在函数签名中显式传递 trace_id。
#
# 架构:
#   ┌─────────────────────────────────────────────────┐
#   │  FastAPI Middleware (chat_guard)                 │
#   │    → set_trace_id(uuid)                          │
#   │    → 所有后续日志自动带上该 trace_id            │
#   ├─────────────────────────────────────────────────┤
#   │  agent_service.stream_chat()                     │
#   │    → logger.info("stream start")   [trace_id ✅] │
#   │    → tool 调用                      [trace_id ✅] │
#   │    → RAGAS 评估                     [trace_id ✅] │
#   ├─────────────────────────────────────────────────┤
#   │  tools.py (各检索工具)                           │
#   │    → logger.debug("searching...")  [trace_id ✅] │
#   └─────────────────────────────────────────────────┘
#
# 输出格式 (JSON Lines):
#   {"timestamp":"2026-07-27T10:30:01.123Z","level":"INFO","trace_id":"a1b2c3d4",
#    "logger":"server.services.agent_service","message":"stream start",
#    "extra":{"mode":"📖 教材知识问答","thread_id":"guide_exam_knowledge"}}

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional


# ── Trace ID ──────────────────────────────────────────
# 使用 contextvars 实现请求级上下文传播。
# 在 asyncio Task / ThreadPoolExecutor 中都会自动继承。

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)

# 额外的请求上下文（可选，用于结构化日志附加上下文）
_request_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "request_context", default={}
)


def set_trace_id(tid: str = None) -> str:
    """设置当前请求的 trace_id。如不传则自动生成一个 UUID8。"""
    if tid is None:
        tid = uuid.uuid4().hex[:8]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """获取当前请求的 trace_id。"""
    return _trace_id.get()


def set_request_context(**kwargs):
    """设置当前请求的附加上下文（mode, user_id, thread_id 等）。"""
    _request_context.set(kwargs)


def get_request_context() -> dict:
    """获取当前请求的附加上下文。"""
    return _request_context.get()


# ── JSON 格式化器 ─────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """将日志记录格式化为 JSON Lines，便于日志聚合系统（ELK/Loki/ClickHouse）索引。

    每条日志输出的 JSON 结构:
    {
        "timestamp": "ISO 8601 UTC",
        "level": "INFO|WARNING|ERROR|DEBUG",
        "trace_id": "请求追踪 ID（空字符串表示无请求上下文）",
        "logger": "模块路径",
        "message": "日志文本",
        "module": "文件名",
        "line": 行号,
        "extra": { ... 自定义字段 }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "trace_id": _trace_id.get(),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # 附加请求上下文
        ctx = _request_context.get()
        if ctx:
            log_obj["context"] = ctx

        # 附加用户自定义的 extra 字段（logger.info("msg", extra={...})）
        if hasattr(record, "extra_fields") and record.extra_fields:
            log_obj["extra"] = record.extra_fields

        # 异常信息
        if record.exc_info and record.exc_info[0]:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False, default=str)


# ── Logger 适配器（支持 extra 字段）───────────────────

class StructuredLogger(logging.LoggerAdapter):
    """自定义 LoggerAdapter，支持 extra_fields 传递。

    用法:
        logger = get_logger(__name__)
        logger.info("search completed", extra_fields={"tool": "hybrid_search", "latency_ms": 120})

    输出的 JSON 中:
        {"extra": {"tool": "hybrid_search", "latency_ms": 120}, ...}
    """

    def process(self, msg, kwargs):
        extra_fields = kwargs.pop("extra_fields", None)
        if extra_fields:
            kwargs["extra"] = {"extra_fields": extra_fields}
        return msg, kwargs


# ── 初始化工厂 ────────────────────────────────────────

_initialized = False


def setup_structured_logging(
    level: int = None,
    log_file: str = None,
    console: bool = True,
):
    """初始化全局结构化日志配置。

    Args:
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取，兜底 INFO
        log_file: 可选的文件输出路径（JSON Lines）
        console: 是否输出到 stdout
    """
    global _initialized
    if _initialized:
        return

    if level is None:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有的处理器（避免重复）
    root_logger.handlers.clear()

    formatter = JsonFormatter()

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # 降低第三方库的日志噪音
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "redis"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> StructuredLogger:
    """获取结构化 Logger。

    用法:
        from server.core.structured_logger import get_logger
        logger = get_logger(__name__)
        logger.info("message", extra_fields={"key": "value"})
    """
    raw_logger = logging.getLogger(name)
    return StructuredLogger(raw_logger, {})


# ── RequestIdMiddleware（FastAPI 中间件）───────────────
# 用法（在 main.py 中）:
#   from server.core.structured_logger import RequestIdMiddleware
#   app.add_middleware(RequestIdMiddleware)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestIdMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件：为每个 HTTP 请求自动生成并设置 trace_id。

    - 如果客户端传了 X-Request-Id 头，则复用（便于跨服务追踪）
    - 否则自动生成一个 8 位 UUID
    - 在响应头中返回 X-Request-Id
    - 请求结束时记录一条汇总日志（状态码 + 耗时）
    """

    async def dispatch(self, request: Request, call_next):
        # 优先复用客户端传入的 request_id
        tid = request.headers.get("X-Request-Id", "") or uuid.uuid4().hex[:8]
        set_trace_id(tid)

        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        response.headers["X-Request-Id"] = tid
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        # 汇总日志
        logger = get_logger("server.middleware.request_id")
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code}",
            extra_fields={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        return response
