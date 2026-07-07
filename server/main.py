# server/main.py
# FastAPI application entry point

import os
import locale

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
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
        from tools import _init_bm25, get_embeddings
        _init_bm25()
        get_embeddings()
    except Exception as e:
        print(f"⚠️ 预热失败（首次请求时懒加载）: {e}")

    yield

    # Shutdown
    try:
        if app.state.redis:
            app.state.redis.close()
    except Exception:
        pass


app = FastAPI(
    title="导游考试 AI 助手 API",
    description="导游考试智能助手后端服务",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Import routes AFTER app is created to avoid circular imports ──
from server.routes import chat, feedback, evaluation, question_bank, cost, chat_log, wrong_book
from server.routes.cost import router as cost_router  # has two routes

app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(evaluation.router)
app.include_router(question_bank.router)
app.include_router(cost_router)
app.include_router(chat_log.router)
app.include_router(wrong_book.router)


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
                    "全陪导游的职责是什么？",
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
        ]
    }


# ── Static files + index ──────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mount static files AFTER route definitions
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8080, reload=True)
