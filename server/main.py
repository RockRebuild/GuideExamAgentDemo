# server/main.py
# FastAPI application entry point

import os
import locale
import signal

# ── 编码设置 ──────────────────────────────────────────
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    pass
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["PYTHONIOENCODING"] = "utf-8"

import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*NoSessionContext.*")

from contextlib import asynccontextmanager
from pathlib import Path

import redis
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

# ── 结构化日志初始化 ─────────────────────────────────
from server.core.structured_logger import setup_structured_logging, RequestIdMiddleware
setup_structured_logging()

# ── Langfuse (import to initialize) ──────────────────
from langfuse import Langfuse
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    # Startup — Redis（先试 Docker 容器名，再试 localhost）
    for host in ["redis", "localhost", "127.0.0.1"]:
        try:
            r = redis.Redis(host=host, port=6379, decode_responses=True, socket_connect_timeout=2)
            r.ping()
            app.state.redis = r
            print(f"✅ Redis 连接成功 (host={host})")
            break
        except Exception:
            app.state.redis = None
    if app.state.redis is None:
        print("⚠️ Redis 不可用（反馈统计和费用记录将不可用）")

    # Pre-load BM25 index and BGE reranker (warm-up)
    try:
        from server.core.tools import _init_bm25, get_embeddings
        _init_bm25()
        get_embeddings()
    except Exception as e:
        print(f"⚠️ 预热失败（首次请求时懒加载）: {e}")

    # ── 并发控制初始化 ──
    try:
        from server.core.concurrency import init_manager, ConcurrencyConfig
        config = ConcurrencyConfig.from_env()
        app.state.concurrency = init_manager(config=config, redis_client=app.state.redis)
        print("✅ ConcurrencyManager 已初始化")
    except Exception as e:
        print(f"⚠️ ConcurrencyManager 初始化失败: {e}")
        app.state.concurrency = None

    yield

    # ── Graceful Shutdown ──────────────────────────────
    # 1. 停止接收新请求（由 uvicorn 处理 SIGTERM）
    # 2. 等待现有 SSE 连接完成（最多 30s）
    # 3. 关闭全局线程池
    # 4. 关闭 Redis
    from server.core.structured_logger import get_logger
    logger = get_logger("server.main")
    logger.info("Starting graceful shutdown...")

    try:
        from server.core.executor import shutdown_executor
        shutdown_executor(wait=True, timeout=30)
        logger.info("ThreadPoolExecutor shutdown complete")
    except Exception as e:
        logger.warning(f"ThreadPoolExecutor shutdown error: {e}")
        shutdown_executor(wait=False)

    try:
        if app.state.redis:
            app.state.redis.close()
            logger.info("Redis connection closed")
    except Exception:
        pass


app = FastAPI(
    title="AI导游考试Agent-RAG智能问答系统",
    description="导游考试智能助手后端服务",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Import routes AFTER app is created to avoid circular imports ──
from server.routes import chat, feedback, evaluation, question_bank, cost, chat_log, wrong_book, agent_eval, hitl
from server.routes.cost import router as cost_router  # has two routes

app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(evaluation.router)
app.include_router(question_bank.router)
app.include_router(cost_router)
app.include_router(chat_log.router)
app.include_router(wrong_book.router)
app.include_router(agent_eval.router)
app.include_router(hitl.router)

# ── Structured Logging Middleware ──────────────────────
app.add_middleware(RequestIdMiddleware)

# ── Prometheus Metrics Endpoint ────────────────────────

@app.get("/metrics")
async def metrics():
    """Prometheus metrics 端点。
    暴露: request_duration, tool_call_duration, circuit_breaker_state,
          queue_depth, rate_limited, cache_hit_ratio 等关键指标。
    """
    from server.core.metrics import get_metrics_response, PROMETHEUS_AVAILABLE
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content=get_metrics_response(),
            media_type="text/plain",
            status_code=501,
        )
    return Response(
        content=get_metrics_response(),
        media_type="text/plain; version=0.0.4",
    )


# ── Modes endpoint ────────────────────────────────────

@app.get("/api/modes")
async def list_modes():
    """Get available modes and sample questions."""
    return {
        "modes": [
            {
                "name": "📖 教材知识问答",
                "samples": [
                    "政策与法律法规的第二章主要讲了什么？",
                    "查询未来五天的杭州天气",
                    "《旅游法》第35条是什么？",
                    "导游证的种类有哪些？"
                ]
            },
            {
                "name": "📝 智能出卷",
                "samples": [
                    "导游业务 团队导游服务规范 出3道单选题",
                    "合同法律制度出5道多选题",
                    "中国饮食文化 出4道判断题",
                ]
            },
            {
                "name": "📊 阅卷批改",
                "samples": [
                    "请批改题目 科目一的第一章单选第一题，我的答案是 B",
                    "帮我批改题目 科目四的第十章多选第三题，我选 A，B",
                ]
            },
            {
                "name": "🤖 多Agent协作",
                "samples": [
                    "对比一下导游业务第三章和法律法规第四章的主要知识点",
                    "出3道单选题并解释每道题涉及的教材知识点",
                    "帮我批改答案后推荐相关知识点复习",
                ]
            },
        ]
    }


# ── Static files + index ──────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    """Deep health check: Redis, ChromaDB, DeepSeek API, Reranker, Concurrency。

    设计原理:
    - Kubernetes readiness probe 用 GET /health?level=ready
    - Kubernetes liveness probe 用 GET /health?level=alive
    - 运维监控用 GET /health?level=deep (全部检查)
    """
    import time, psutil

    level = "deep"  # 默认深度检查

    checks = {}
    overall = "ok"

    # ── Reranker 状态 ──────────────────────────────
    reranker_status = "unknown"
    try:
        from server.core.retrieval_utils import _reranker_disabled, _is_reranker_permanently_disabled
        if _is_reranker_permanently_disabled():
            reranker_status = "permanently_disabled"
        elif _reranker_disabled:
            reranker_status = "disabled_this_session"
        else:
            reranker_status = "enabled"
    except Exception:
        pass
    checks["reranker"] = reranker_status

    # ── Redis (深度检查: 实际 ping) ─────────────────
    redis_status = "not_available"
    redis_latency_ms = None
    if hasattr(app.state, "redis") and app.state.redis:
        try:
            t0 = time.monotonic()
            app.state.redis.ping()
            redis_latency_ms = (time.monotonic() - t0) * 1000
            redis_status = "ok"
        except Exception:
            redis_status = "error"
    checks["redis"] = {
        "status": redis_status,
        "latency_ms": round(redis_latency_ms, 1) if redis_latency_ms else None,
    }

    # ── ChromaDB (深度检查: 集合数量 + 文档计数) ──
    chroma_status = "unknown"
    chroma_detail = {}
    try:
        from server.core.tools import get_vectorstore
        collections = ["guide_child", "guide_parent", "guide_summary", "guide_sentence", "semantic_cache"]
        total_docs = 0
        for coll_name in collections:
            try:
                vs = get_vectorstore(coll_name)
                count = vs._collection.count()
                chroma_detail[coll_name] = count
                total_docs += count
            except Exception:
                chroma_detail[coll_name] = -1
        chroma_status = "ok" if total_docs > 0 else "empty"
        chroma_detail["total_docs"] = total_docs
    except Exception as e:
        chroma_status = "error"
        chroma_detail["error"] = str(e)[:200]
    checks["chromadb"] = {"status": chroma_status, **chroma_detail}

    # ── DeepSeek API (深度检查: 实际 API 延迟) ─────
    deepseek_status = "not_checked"
    deepseek_latency_ms = None
    try:
        from langchain_openai import ChatOpenAI
        ping_llm = ChatOpenAI(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            temperature=0,
            max_tokens=1,
            extra_body={"thinking": {"type": "disabled"}},
        )
        t0 = time.monotonic()
        ping_llm.invoke("ping")
        deepseek_latency_ms = (time.monotonic() - t0) * 1000
        deepseek_status = "ok"
    except Exception as e:
        deepseek_status = "error"
        deepseek_latency_ms = (time.monotonic() - t0) * 1000 if 't0' in dir() else None
    checks["deepseek_api"] = {
        "status": deepseek_status,
        "latency_ms": round(deepseek_latency_ms, 1) if deepseek_latency_ms else None,
    }

    # ── 并发控制状态 ─────────────────────────────
    concurrency_status = "not_initialized"
    if hasattr(app.state, "concurrency") and app.state.concurrency:
        try:
            concurrency_status = app.state.concurrency.get_health()
        except Exception:
            concurrency_status = "error"
    checks["concurrency"] = concurrency_status

    # ── 系统资源 ─────────────────────────────────
    try:
        mem = psutil.virtual_memory()
        checks["system"] = {
            "memory_total_mb": round(mem.total / 1024 / 1024, 1),
            "memory_available_mb": round(mem.available / 1024 / 1024, 1),
            "memory_percent": mem.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }
    except Exception:
        checks["system"] = "unavailable"

    # ── 综合判定 ─────────────────────────────────
    if redis_status == "error":
        overall = "degraded"
    if chroma_status == "error":
        overall = "degraded"
    if deepseek_status == "error":
        overall = "degraded"

    return {
        "status": overall,
        "checks": checks,
    }


@app.get("/api/queue/status")
async def queue_status(request: Request, token: str = ""):
    """SSE 端点：推送排队位置变化。前端在收到 429+queue 后连接此端点。"""
    from fastapi.responses import StreamingResponse

    manager = getattr(request.app.state, "concurrency", None)
    if manager is None:
        return JSONResponse(
            {"error": "concurrency not available"}, status_code=503
        )

    async def event_stream():
        async for sse in manager.stream_queue_position(token):
            yield sse

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# Mount static files AFTER route definitions
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8080, reload=True)
