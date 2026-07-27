# server/middleware/concurrency.py
# ── 并发控制中间件（路由 guard 函数）──
# 用于在 chat 路由中串联缓存→限流→排队→熔断。
# 不破坏 SSE 流式体验。

import json
import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from server.core.concurrency import ConcurrencyManager, get_manager, GuardResult
from server.core.concurrency.config import Priority
from server.core.semantic_cache import lookup as cache_lookup

logger = logging.getLogger(__name__)


async def chat_guard(
    request: Request,
    prompt: str,
    mode: str = "",
) -> Optional[JSONResponse]:
    """chat 路由准入 guard。

    Args:
        request: FastAPI Request 对象
        prompt: 用户输入（已 sanitize）
        mode: 聊天模式

    Returns:
        None → 放行，继续执行 Agent
        JSONResponse → 被拦截（限流/熔断/排队），直接返回给客户端
    """
    manager = get_manager()
    if manager is None:
        return None  # concurrency 未初始化，放行

    user_id = _get_user_id(request)
    ip = _get_client_ip(request)

    # ── Layer 1: 语义缓存（提前检查，命中则绕过一切）──
    try:
        cached = cache_lookup(prompt)
        if cached:
            # 返回缓存命中信号 → 让路由层处理（StreamingResponse 不能在这里返回）
            return _json_429({
                "error": "cache_hit",
                "message": "使用缓存结果",
                "cached_result": cached,
            })
    except Exception:
        pass  # 缓存不可用，继续正常流程

    # ── Layer 2-4: 熔断 + 限流 + 排队 ──
    # 多Agent模式给更高的优先级（通常是复杂查询）
    priority = Priority.HIGH if "多Agent" in mode else Priority.NORMAL

    result = await manager.guard(
        user_id=user_id,
        ip=ip,
        priority=priority,
        prompt=prompt,
    )

    if result.action == "pass":
        return None  # 放行

    if result.action == "circuit_open":
        # 熔断打开 → 再试一次缓存
        try:
            cached = cache_lookup(prompt)
            if cached:
                return _json_429({
                    "error": "service_degraded",
                    "message": "AI 服务暂时繁忙，以下为缓存结果",
                    "cached_result": cached,
                    "retry_after": getattr(result, "retry_after", 30),
                })
        except Exception:
            pass
        return _json_429({
            "error": "service_unavailable",
            "message": "服务暂时不可用，请稍后重试",
            "retry_after": getattr(result, "retry_after", 30),
        })

    if result.action == "queued":
        return _json_429({
            "error": "queued",
            "message": f"请求已排队，前方 {result.queue_position} 人",
            "queue_position": result.queue_position,
            "estimated_wait_s": result.estimated_wait_s,
            "queue_token": result.queue_token,
            "retry_after": getattr(result, "retry_after", 5),
        })

    if result.action == "queue_full":
        return _json_429({
            "error": "queue_full",
            "message": "队列已满，请稍后重试",
            "retry_after": getattr(result, "retry_after", 30),
        })

    if result.action == "rate_limited":
        return _json_429({
            "error": "rate_limited",
            "message": f"请求频率过高，请 {result.retry_after:.0f} 秒后重试",
            "retry_after": getattr(result, "retry_after", 5),
            "blocked_by": getattr(result, "blocked_by", "unknown"),
        })

    return None


def release_guard():
    """Agent 执行完毕后释放资源。"""
    manager = get_manager()
    if manager:
        manager.release()


# ── 内部工具 ─────────────────────────────────────────

def _get_user_id(request: Request) -> str:
    """从请求中提取用户标识。"""
    # 优先使用自定义 header
    user_id = request.headers.get("X-User-Id", "")
    if user_id:
        return user_id
    # 次选 session cookie
    session = request.cookies.get("session_id", "")
    if session:
        return f"session:{session[:16]}"
    # 兜底 IP
    return f"ip:{_get_client_ip(request)}"


def _get_client_ip(request: Request) -> str:
    """获取客户端真实 IP。"""
    # 如果有反向代理（nginx/CDN），优先取 X-Forwarded-For
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # 直连取 client.host
    if request.client:
        return request.client.host
    return "127.0.0.1"


def _json_429(data: dict) -> JSONResponse:
    """生成 429 响应（带上 Retry-After 头）。"""
    retry_after = int(data.get("retry_after", 5))
    return JSONResponse(
        content=data,
        status_code=429,
        headers={"Retry-After": str(retry_after)},
    )
