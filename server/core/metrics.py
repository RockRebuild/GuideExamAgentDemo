# server/core/metrics.py
# ── Prometheus Metrics 暴露模块 ──
#
# 原理:
#   Prometheus 是 Cloud Native 生态的监控标准。本模块使用 prometheus_client
#   库定义关键指标，通过 FastAPI 的 /metrics 端点暴露给 Prometheus 抓取。
#
# 架构:
#   ┌──────────────┐   scrape /metrics   ┌─────────────┐
#   │  Prometheus  │ ◄────────────────── │  FastAPI     │
#   │  / Grafana   │   every 15s         │  :8080       │
#   └──────────────┘                     └─────────────┘
#
# 指标类型选择原理:
#   - Counter: 只增不减的累计值（请求数、错误数、token 数）
#     适合计算 rate() 得到 QPS、错误率
#   - Histogram: 分布统计（延迟），自动生成 _bucket/_sum/_count
#     适合计算 p50/p95/p99 分位数和平均值
#   - Gauge: 可增可减的瞬时值（并发数、队列深度）
#     适合看当前状态
#
# 关键指标设计:
#   ┌─────────────────────────────────────────────────────────────┐
#   │ 指标                       │ 类型      │ 标签              │
#   ├─────────────────────────────────────────────────────────────┤
#   │ request_total               │ Counter   │ mode, status      │
#   │ request_duration_seconds    │ Histogram │ mode              │
#   │ tool_call_duration_seconds  │ Histogram │ tool_name         │
#   │ tool_call_total             │ Counter   │ tool_name, status │
#   │ llm_api_errors_total        │ Counter   │ error_type        │
#   │ circuit_breaker_state       │ Gauge     │ circuit_name      │
#   │ queue_depth                 │ Gauge     │                   │
#   │ rate_limited_total          │ Counter   │ blocked_by        │
#   │ cache_hit_ratio             │ Gauge     │                   │
#   │ concurrent_requests         │ Gauge     │                   │
#   │ chromadb_query_duration     │ Histogram │ operation         │
#   │ rag_agent_active            │ Gauge     │ mode              │
#   │ token_usage_total           │ Counter   │ model, type       │
#   └─────────────────────────────────────────────────────────────┘

import os
import time
from contextlib import contextmanager
from typing import Optional

# prometheus_client 不是核心依赖，做可选导入
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # 创建空壳，避免 import 报错，运行时检查 PROMETHEUS_AVAILABLE
    class _FakeMetric:
        def labels(self, **kwargs): return self
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def set(self, value): pass
        def observe(self, value): pass
        def time(self): return _FakeContext()
    class _FakeContext:
        def __enter__(self): pass
        def __exit__(self, *a): pass
    Counter = Histogram = Gauge = Info = lambda *a, **kw: _FakeMetric()
    def generate_latest(*a, **kw): return b"# prometheus_client not installed\n"
    REGISTRY = None


METRICS_PREFIX = "guide_exam"

# ── Request Metrics ───────────────────────────────────

request_total = Counter(
    f"{METRICS_PREFIX}_request_total",
    "Total number of chat requests",
    ["mode", "status"],  # status: ok|error|rejected|timeout
)

request_duration_seconds = Histogram(
    f"{METRICS_PREFIX}_request_duration_seconds",
    "End-to-end request duration (seconds)",
    ["mode"],
    buckets=[0.5, 1, 2, 5, 10, 15, 30, 60, 120],
)

stream_first_token_seconds = Histogram(
    f"{METRICS_PREFIX}_stream_first_token_seconds",
    "Time to first token in SSE stream",
    ["mode"],
    buckets=[0.1, 0.25, 0.5, 1, 2, 3, 5, 10],
)

# ── Tool Call Metrics ─────────────────────────────────

tool_call_total = Counter(
    f"{METRICS_PREFIX}_tool_call_total",
    "Total number of tool calls",
    ["tool_name", "status"],  # status: success|error|cache_hit
)

tool_call_duration_seconds = Histogram(
    f"{METRICS_PREFIX}_tool_call_duration_seconds",
    "Tool call duration (seconds)",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
)

# ── LLM / API Metrics ─────────────────────────────────

llm_api_errors_total = Counter(
    f"{METRICS_PREFIX}_llm_api_errors_total",
    "Total LLM API errors",
    ["error_type"],  # RateLimitError|APIConnectionError|InternalServerError|APITimeoutError|other
)

llm_retry_total = Counter(
    f"{METRICS_PREFIX}_llm_retry_total",
    "Total LLM call retries",
    ["attempt"],  # 1|2|3
)

token_usage_total = Counter(
    f"{METRICS_PREFIX}_token_usage_total",
    "Total token usage",
    ["model", "type"],  # type: input|output
)

# ── Concurrency Metrics ───────────────────────────────

circuit_breaker_state = Gauge(
    f"{METRICS_PREFIX}_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["circuit_name"],
)

queue_depth = Gauge(
    f"{METRICS_PREFIX}_queue_depth",
    "Current request queue depth",
)

rate_limited_total = Counter(
    f"{METRICS_PREFIX}_rate_limited_total",
    "Total rate-limited requests",
    ["blocked_by"],  # global|user|ip|local
)

concurrent_requests = Gauge(
    f"{METRICS_PREFIX}_concurrent_requests",
    "Currently executing agent requests",
)

degradation_level = Gauge(
    f"{METRICS_PREFIX}_degradation_level",
    "Current degradation level (0=full, 1=throttled, 2=degraded, 3=minimal)",
)

# ── Cache Metrics ─────────────────────────────────────

cache_hits_total = Counter(
    f"{METRICS_PREFIX}_cache_hits_total",
    "Total semantic cache hits",
)

cache_misses_total = Counter(
    f"{METRICS_PREFIX}_cache_misses_total",
    "Total semantic cache misses",
)

cache_hit_ratio = Gauge(
    f"{METRICS_PREFIX}_cache_hit_ratio",
    "Semantic cache hit ratio (0.0~1.0)",
)

# ── Retrieval Metrics ─────────────────────────────────

chromadb_query_duration_seconds = Histogram(
    f"{METRICS_PREFIX}_chromadb_query_duration_seconds",
    "ChromaDB query duration (seconds)",
    ["operation"],  # semantic_search|hybrid_search|get_ids|bm25
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2],
)

retrieval_context_count = Histogram(
    f"{METRICS_PREFIX}_retrieval_context_count",
    "Number of contexts returned per retrieval",
    ["tool_name"],
    buckets=[0, 1, 2, 3, 5, 8, 12, 20],
)

# ── Health / Info ─────────────────────────────────────

app_info = Info(
    f"{METRICS_PREFIX}_app",
    "Application information",
)

# ── Convenience Functions ─────────────────────────────


def record_request(mode: str, status: str):
    """记录一次请求。"""
    request_total.labels(mode=mode, status=status).inc()


def record_tool_call(tool_name: str, status: str, duration_s: float):
    """记录一次工具调用。"""
    tool_call_total.labels(tool_name=tool_name, status=status).inc()
    tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration_s)


def record_llm_error(error_type: str):
    """记录 LLM API 错误。"""
    llm_api_errors_total.labels(error_type=error_type).inc()


def record_retry(attempt: int):
    """记录重试次数。"""
    llm_retry_total.labels(attempt=str(attempt)).inc()


def record_token_usage(model: str, input_tokens: int, output_tokens: int):
    """记录 token 用量。"""
    token_usage_total.labels(model=model, type="input").inc(input_tokens)
    token_usage_total.labels(model=model, type="output").inc(output_tokens)


def set_circuit_breaker_state(circuit_name: str, state: str):
    """更新熔断器状态指标。"""
    value = {"closed": 0, "open": 1, "half_open": 2}.get(state, -1)
    circuit_breaker_state.labels(circuit_name=circuit_name).set(value)


def set_queue_depth(depth: int):
    """更新排队深度。"""
    queue_depth.set(depth)


def record_rate_limited(blocked_by: str):
    """记录一次限流拒绝。"""
    rate_limited_total.labels(blocked_by=blocked_by).inc()


def update_cache_stats(hits: int, misses: int):
    """更新缓存命中统计。"""
    total = hits + misses
    cache_hits_total.inc(hits - _cache_hits_last if hasattr(update_cache_stats, "_last_hits") else hits)
    cache_misses_total.inc(misses - _cache_misses_last if hasattr(update_cache_stats, "_last_misses") else misses)
    if total > 0:
        cache_hit_ratio.set(hits / total)
    update_cache_stats._last_hits = hits
    update_cache_stats._last_misses = misses


def record_chromadb_query(operation: str, duration_s: float):
    """记录 ChromaDB 查询延迟。"""
    chromadb_query_duration_seconds.labels(operation=operation).observe(duration_s)


@contextmanager
def track_tool_call(tool_name: str):
    """上下文管理器：自动记录工具调用的耗时和成功/失败。

    用法:
        with track_tool_call("hybrid_search"):
            result = do_search()
    """
    start = time.monotonic()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start
        record_tool_call(tool_name, status, duration)


@contextmanager
def track_request(mode: str):
    """上下文管理器：自动记录请求耗时和状态。

    用法:
        with track_request(mode) as ctx:
            ctx["status"] = "ok"
            do_work()
    """
    start = time.monotonic()
    concurrent_requests.inc()
    status = "ok"
    try:
        ctx = {"status": "ok"}
        yield ctx
        status = ctx.get("status", "ok")
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start
        request_duration_seconds.labels(mode=mode).observe(duration)
        request_total.labels(mode=mode, status=status).inc()
        concurrent_requests.dec()


# ── Metrics Endpoint Helper ───────────────────────────

def get_metrics_response():
    """返回 Prometheus 格式的 metrics 文本。"""
    if not PROMETHEUS_AVAILABLE:
        return "# prometheus_client not installed. Run: pip install prometheus_client\n"
    return generate_latest(REGISTRY)


def init_app_info():
    """在 startup 时设置 app info。"""
    app_info.info({
        "version": "2.0.0",
        "python": os.environ.get("PYTHON_VERSION", "3.11"),
        "deepseek_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    })


# ── Sync metrics from ConcurrencyManager ──────────────

def sync_concurrency_metrics(manager):
    """从 ConcurrencyManager 同步指标到 Prometheus。

    建议在后台定时调用（如每 5 秒），或通过 FastAPI background task。
    """
    try:
        health = manager.get_health()

        # Circuit breaker states
        for name, cb_health in health.get("circuits", {}).items():
            set_circuit_breaker_state(name, cb_health.get("state", "unknown"))

        # Queue depth
        set_queue_depth(health.get("queue_depth", 0))

        # Degradation level
        level_map = {"full": 0, "throttled": 1, "degraded": 2, "minimal": 3}
        degradation_level.set(level_map.get(health.get("degradation_level"), -1))

    except Exception:
        pass  # metrics 不应影响主流程
