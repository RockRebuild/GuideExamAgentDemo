# 生产就绪改进方案：压力测试 · 可观测性 · 健壮性

> 基于对项目全面代码审查后的改进实施文档，涵盖原理、方案、代码变更和运维 runbook。

---

## 目录

1. [改进总览](#1-改进总览)
2. [结构化日志 + Trace ID](#2-结构化日志--trace-id)
3. [Prometheus Metrics](#3-prometheus-metrics)
4. [全局线程池 + asyncio 桥接](#4-全局线程池--asyncio-桥接)
5. [请求超时 + Graceful Shutdown](#5-请求超时--graceful-shutdown)
6. [ChromaDB 熔断器接入](#6-chromadb-熔断器接入)
7. [BM25 线程安全](#7-bm25-线程安全)
8. [评估日志轮转](#8-评估日志轮转)
9. [Deep Health Check](#9-deep-health-check)
10. [压力测试套件](#10-压力测试套件)
11. [端到端集成测试](#11-端到端集成测试)
12. [运维 Runbook](#12-运维-runbook)

---

## 1. 改进总览

### 1.1 改进前状态评估

| 维度 | 已有基础 | 关键缺口 |
|------|---------|---------|
| 并发控制 | 4 层防护（限流→排队→熔断→执行） | ChromaDB 熔断器定义了但未接入 guard |
| 日志 | 标准 logging 模块 | 无结构化格式、无请求级 trace_id |
| 指标 | 无 | 无 /metrics 端点，无法接入 Prometheus |
| 健康检查 | `/health` 检查 reranker + concurrency | 无 Redis/ChromaDB/API 实际连通性检查 |
| 线程管理 | 每个请求新建 ThreadPoolExecutor | 线程泄漏风险，无全局复用 |
| 超时控制 | 排队有 `queue_timeout_seconds` | Agent 执行本身无超时保护 |
| 优雅关闭 | 仅关闭 Redis 连接 | 无等待进行中请求、无线程池关闭 |
| 测试 | 单组件单元测试 | 无端到端集成测试、无 HTTP 压力测试 |
| 日志轮转 | 无 | eval_log.jsonl 无限增长 |

### 1.2 改进后架构

```
                         ┌─────────────────────────┐
                         │     Prometheus          │
                         │     / Grafana           │
                         └──────────┬──────────────┘
                                    │ scrape /metrics (every 15s)
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI (main.py)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  RequestIdMiddleware → X-Request-Id (trace_id 传播)      │ │
│  │  /metrics → Prometheus 格式                              │ │
│  │  /health  → Deep Check (Redis + ChromaDB + API + 系统)   │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  chat_guard (middleware/concurrency.py)                  │ │
│  │    Layer 1: 语义缓存 (提前返回)                           │ │
│  │    Layer 2: ChromaDB 熔断器 → 缓存兜底 / 503              │ │
│  │    Layer 3: DeepSeek 熔断器 → 缓存兜底 / 503              │ │
│  │    Layer 4: 三层限流 (global + user + ip)                 │ │
│  │    Layer 5: 优先级排队 → 429 + queue_token                │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  agent_service.stream_chat()                              │ │
│  │    请求级 120s 硬超时                                      │ │
│  │    全局共享 ThreadPoolExecutor (复用，不泄漏)              │ │
│  │    SSE 流式输出 (event: tool/agent/done/hitl/error)       │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  lifespan shutdown (Graceful)                             │ │
│  │    1. 停止接收新请求                                       │ │
│  │    2. 等待现有 SSE 连接完成 (最多 30s)                     │ │
│  │    3. shutdown_executor(wait=True, timeout=30)             │ │
│  │    4. Redis.close()                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐  ┌──────────┐  ┌──────────────┐
              │ 日志输出  │  │ Metrics  │  │ eval_log 轮转 │
              │ JSON Lines│  │ Counter  │  │ 按天归档      │
              │ stdout    │  │ Histogram│  │ 保留 30 天    │
              │ + 文件    │  │ Gauge    │  │               │
              └──────────┘  └──────────┘  └──────────────┘
```

---

## 2. 结构化日志 + Trace ID

### 2.1 问题

改进前的日志：

```
⚠️ 上下文超限: 105000 > 96000 tokens, 开始裁剪 (48 条消息)
```

问题：无法知道是哪个请求、哪个用户触发的。在并发环境下，日志行交错在一起，完全无法追踪。

### 2.2 原理

使用 Python `contextvars` 实现请求级的 Trace ID 自动传播：

```
contextvars 的特性:
  - 类似线程局部存储 (thread-local)，但支持 asyncio Task 继承
  - 当 async Task A spawns Task B 时，B 自动继承 A 的 context
  - 在 ThreadPoolExecutor 中也会自动 propagate
  → 不需要在函数签名中显式传递 trace_id
```

### 2.3 实现

**新文件**: `server/core/structured_logger.py`

```
核心组件:
  _trace_id: ContextVar[str]      ← 请求级 trace_id
  _request_context: ContextVar    ← 附加上下文 (mode, user_id)
  JsonFormatter                   ← JSON Lines 输出
  StructuredLogger                ← 支持 extra_fields 注入
  RequestIdMiddleware             ← FastAPI 中间件，自动生成/传播 trace_id
```

**日志输出格式** (JSON Lines):

```json
{"timestamp":"2026-07-27T10:30:01.123Z","level":"INFO","trace_id":"a1b2c3d4","logger":"server.services.agent_service","message":"stream start","extra":{"mode":"📖 教材知识问答","thread_id":"guide_exam_knowledge"}}
```

**集成方式** (`main.py`):

```python
from server.core.structured_logger import setup_structured_logging, RequestIdMiddleware
setup_structured_logging()  # 在 app 创建前调用
app.add_middleware(RequestIdMiddleware)  # 每个请求自动生成 trace_id
```

### 2.4 日志聚合对接

| 平台 | 配置 |
|------|------|
| ELK (Filebeat) | Filebeat 直接读取 stdout JSON，自动解析 |
| Loki + Grafana | Promtail 配置 `json` pipeline stage |
| ClickHouse | `INSERT INTO logs FORMAT JSONEachRow` |
| 本地开发 | 用 `jq` 格式化：`tail -f app.log \| jq '.'` |

---

## 3. Prometheus Metrics

### 3.1 问题

没有指标，无法回答以下运维问题：
- 当前 QPS 是多少？P95 延迟是多少？
- 哪种工具调用最慢？
- 缓存命中率是上升还是下降？
- 熔断器什么时候打开过？

### 3.2 原理

Prometheus 是 CNCF 的监控标准，核心数据模型：

```
指标名{标签1="值",标签2="值"} 数值 时间戳

# 示例
guide_exam_request_total{mode="📖 教材知识问答",status="ok"} 12345 1753624201000
guide_exam_request_duration_seconds_bucket{mode="📖 教材知识问答",le="5.0"} 10000
```

四种指标类型选择：

| 类型 | 用途 | 本项目示例 |
|------|------|-----------|
| Counter | 只增不减的累计值 | `request_total`, `tool_call_total`, `token_usage_total` |
| Histogram | 分布统计（自动生成 bucket） | `request_duration_seconds`, `tool_call_duration_seconds` |
| Gauge | 瞬时值 | `circuit_breaker_state`, `queue_depth`, `cache_hit_ratio` |
| Info | 不变元数据 | `app_info` (version, model) |

### 3.3 实现

**新文件**: `server/core/metrics.py`

**暴露的指标** (14 个):

| 指标 | 类型 | 标签 | 用途 |
|------|------|------|------|
| `request_total` | Counter | mode, status | 计算 QPS、错误率 |
| `request_duration_seconds` | Histogram | mode | P50/P95/P99 延迟 |
| `stream_first_token_seconds` | Histogram | mode | 首 Token 延迟 |
| `tool_call_total` | Counter | tool_name, status | 工具调用频率 |
| `tool_call_duration_seconds` | Histogram | tool_name | 工具性能 |
| `llm_api_errors_total` | Counter | error_type | API 错误分类 |
| `llm_retry_total` | Counter | attempt | 重试分布 |
| `token_usage_total` | Counter | model, type | Token 用量 |
| `circuit_breaker_state` | Gauge | circuit_name | 熔断器状态 |
| `queue_depth` | Gauge | — | 排队深度 |
| `rate_limited_total` | Counter | blocked_by | 限流统计 |
| `concurrent_requests` | Gauge | — | 当前并发 |
| `cache_hits_total` / `cache_misses_total` | Counter | — | 缓存命中率 |
| `chromadb_query_duration_seconds` | Histogram | operation | ChromaDB 性能 |

**Grafana Dashboard 核心面板**:

```promql
# QPS
rate(guide_exam_request_total[1m])

# P95 延迟
histogram_quantile(0.95, rate(guide_exam_request_duration_seconds_bucket[5m]))

# 错误率
sum(rate(guide_exam_request_total{status="error"}[5m]))
/ sum(rate(guide_exam_request_total[5m]))

# 缓存命中率
guide_exam_cache_hit_ratio

# 熔断器状态（1=OPEN 需告警）
guide_exam_circuit_breaker_state
```

### 3.4 告警规则 (PrometheusRule)

```yaml
groups:
  - name: guide_exam
    rules:
      - alert: HighErrorRate
        expr: rate(guide_exam_request_total{status="error"}[5m]) / rate(guide_exam_request_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "错误率超过 10%"

      - alert: HighP95Latency
        expr: histogram_quantile(0.95, rate(guide_exam_request_duration_seconds_bucket[5m])) > 15
        for: 5m
        annotations:
          summary: "P95 延迟超过 15 秒"

      - alert: CircuitBreakerOpen
        expr: guide_exam_circuit_breaker_state == 1
        for: 1m
        annotations:
          summary: "熔断器打开 — DeepSeek API 可能故障"

      - alert: HighQueueDepth
        expr: guide_exam_queue_depth > 50
        for: 5m
        annotations:
          summary: "排队深度超过 50 — 可能需要扩容"

      - alert: LowCacheHitRate
        expr: guide_exam_cache_hit_ratio < 0.2
        for: 30m
        annotations:
          summary: "缓存命中率低于 20%"
```

---

## 4. 全局线程池 + asyncio 桥接

### 4.1 问题 1: 每请求创建新线程池

改进前 `agent_service.py`:
```python
# 每个 SSE 请求创建一个新线程池！
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
executor.submit(_run_sync)
# ... 
finally:
    executor.shutdown(wait=False)  # 异常路径下可能被跳过
```

问题：
- 线程泄漏：客户端断开连接时 `finally` 可能不执行
- 线程膨胀：10 个并发请求 = 10 个线程池 = 10+ 线程
- `wait=False` 意味着任务可能被丢弃

### 4.2 问题 2: `asyncio.run()` 嵌套

改进前 `tools.py`:
```python
def sync_call(**kwargs):
    async def _call(): ...
    return asyncio.run(_call())  # ← 在事件循环内调用会崩溃！
```

LangGraph 的 async 路径可能在已有事件循环中调用工具函数 → `RuntimeError: asyncio.run() cannot be called from a running event loop`。

### 4.3 解决方案

**新文件**: `server/core/executor.py`

```
核心组件:
  get_executor()            ← 全局单例 ThreadPoolExecutor (线程安全懒加载)
  run_in_executor(fn)       ← async 上下文中执行同步阻塞函数
  run_async_in_sync(coro)   ← 同步代码中安全执行 async 协程
    自动检测: 不在事件循环内 → asyncio.run()
             在事件循环内   → 独立线程 + 独立循环
  shutdown_executor()       ← 优雅关闭 (lifespan shutdown 中调用)
```

**线程池大小**: `max_workers = local_max_concurrency + 4`（默认 9）

---

## 5. 请求超时 + Graceful Shutdown

### 5.1 请求超时

**问题**: Agent 调用可能永久挂起（API 无响应、死循环等），SSE 连接一直不关闭。

**修复** (`agent_service.py`):

```python
REQUEST_TIMEOUT_S = float(os.environ.get("AGENT_REQUEST_TIMEOUT_S", "120"))
deadline = asyncio.get_event_loop().time() + REQUEST_TIMEOUT_S

# 在 SSE 事件循环中:
remaining = deadline - asyncio.get_event_loop().time()
if remaining <= 0:
    yield sse_event("error", {"message": "请求超时..."})
    return
item = await asyncio.wait_for(queue.get(), timeout=min(remaining, 30))
```

**配置**: 环境变量 `AGENT_REQUEST_TIMEOUT_S`（默认 120s）

### 5.2 Graceful Shutdown

**问题**: 服务关闭时，正在进行的 SSE 流被粗暴断开 → 用户看到连接错误。

**修复** (`main.py` lifespan shutdown):

```python
# 1. Uvicorn 收到 SIGTERM → 停止接收新连接
# 2. 等待现有请求完成（uvicorn 默认 graceful_timeout=30）
# 3. 关闭全局线程池
shutdown_executor(wait=True, timeout=30)
# 4. 关闭 Redis
app.state.redis.close()
```

**Kubernetes 配合**:

```yaml
lifecycle:
  preStop:
    exec:
      command: ["sleep", "5"]  # 给负载均衡器 5s 更新时间
terminationGracePeriodSeconds: 40  # > preStop + shutdown_timeout
```

---

## 6. ChromaDB 熔断器接入

### 6.1 问题

`create_circuits()` 创建了 `deepseek` 和 `chromadb` 两个熔断器，但在 `ConcurrencyManager.guard()` 中只检查了 `deepseek` 熔断器。ChromaDB 的熔断器形同虚设。

### 6.2 修复

在 `guard()` 中增加 ChromaDB 熔断检查（优先级在 DeepSeek 之前）：

```python
# Layer 2: 熔断检查
chroma_cb = self.circuits.get("chromadb")
if chroma_cb and chroma_cb.is_open:
    # ChromaDB 故障 → 立即拒绝（不走排队）
    return GuardResult(action="circuit_open", circuit_name="chromadb", ...)

deepseek_cb = self.circuits.get("deepseek")
if deepseek_cb and deepseek_cb.is_open:
    # DeepSeek 故障 → 尝试语义缓存兜底
    return GuardResult(action="circuit_open", circuit_name="deepseek", ...)
```

**触发条件**: ChromaDB 查询连续失败 `cb_failure_threshold` 次（默认 5 次）

**恢复机制**: `cb_timeout_seconds` 后进入 HALF_OPEN → 探测成功 → CLOSED

**当前限制**: ChromaDB 的失败目前需要在工具函数中显式调用 `manager.circuits["chromadb"].on_failure()`。已在 `tools.py` 保留接入点，下一步可统一封装。

---

## 7. BM25 线程安全

### 7.1 问题

`_init_bm25()` 是模块级懒加载函数，使用 `if _bm25_index is not None: return` 做快速退出。但在并发初始化场景下，多个线程同时看到 `None`，都会进入构建路径 → 重复加载、CPU 飙升、GIL 竞争。

### 7.2 修复

使用 `threading.Lock()` 保护初始化路径（double-check pattern）：

```python
_bm25_init_lock = threading.Lock()

def _init_bm25():
    if _bm25_index is not None:
        return
    with _bm25_init_lock:
        if _bm25_index is not None:  # Double-check
            return
        # ... 实际构建 ...
```

---

## 8. 评估日志轮转

### 8.1 问题

`eval_log.jsonl` 每次评估追加一行，从不清除。在高 QPS 场景下文件无限增长，最终撑满磁盘。

### 8.2 修复

`eval_logger.py` 增加：

- **按天轮转**: 每天凌晨首条日志触发 `_rotate_if_needed()`，将昨天的 `eval_log.jsonl` 重命名为 `eval_log_2026-07-26.jsonl`
- **定期清理**: 自动删除超过 `EVAL_LOG_MAX_AGE_DAYS`（默认 30 天）的日志文件
- **追加保护**: 如果目标文件已存在，追加内容而非覆盖

```python
EVAL_LOG_MAX_AGE_DAYS = int(os.environ.get("EVAL_LOG_MAX_AGE_DAYS", "30"))
```

**配置**: 环境变量 `EVAL_LOG_ROTATION_ENABLED`（默认 true）、`EVAL_LOG_MAX_AGE_DAYS`（默认 30）

---

## 9. Deep Health Check

### 9.1 问题

改进前 `/health` 只检查 reranker 和 concurrency 的状态字符串，没有实际连通性测试。Kubernetes 的 readiness/liveness 探针无法感知真实的依赖故障。

### 9.2 修复

`/health` 现在执行深度检查：

```json
{
  "status": "ok",       // ok | degraded (综合判定)
  "checks": {
    "reranker": "enabled",
    "redis": {
      "status": "ok",
      "latency_ms": 2.3       // 实际 PING 延迟
    },
    "chromadb": {
      "status": "ok",
      "guide_child": 12341,
      "guide_parent": 1791,
      "total_docs": 14460
    },
    "deepseek_api": {
      "status": "ok",
      "latency_ms": 156.2      // 实际 API 调用延迟
    },
    "concurrency": {
      "degradation_level": "full",
      "circuits": {"deepseek": {"state": "closed"}},
      "stats": {"total_passed": 1234}
    },
    "system": {
      "memory_total_mb": 4096,
      "memory_available_mb": 1024,
      "memory_percent": 75.0,
      "cpu_percent": 12.5
    }
  }
}
```

**综合判定逻辑**:
- Redis 连接失败 → `degraded`
- ChromaDB 连接失败 → `degraded`
- DeepSeek API 不可达 → `degraded`
- 全部异常 → `degraded`（不是 `unhealthy`，因为缓存可能仍可服务）

**Kubernetes 探针配置**:

```yaml
livenessProbe:    # 进程存活
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 30

readinessProbe:   # 是否接收流量
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3
```

---

## 10. 压力测试套件

### 10.1 Locust HTTP 压力测试

**新文件**: `tests/locustfile.py`

**设计原理**:

```
模拟真实用户行为分布:
  - 80% 📖 教材知识问答 (高频简单查询)
  - 10% 📝 智能出卷 (中等复杂度)
  - 5%  📊 阅卷批改
  - 5%  🤖 多Agent协作 (高复杂度)

用户行为模型:
  - wait_time = between(3, 10) 秒 ← 模拟真实用户的思考间隔
  - 每个用户独立维持 SSE 长连接
  - 429 限流返回视为正常（验证限流生效）
```

**运行方式**:

```bash
# 安装
pip install locust

# Web UI 模式 (推荐开发/调试):
locust -f tests/locustfile.py --host http://localhost:8080
# 打开 http://localhost:8089 配置并发数和启动速率

# 无 UI 模式 (CI/CD 集成):
locust -f tests/locustfile.py --host http://localhost:8080 \
  --headless \
  --users 50 \           # 模拟 50 个并发用户
  --spawn-rate 5 \       # 每秒启动 5 个用户
  --run-time 5m \        # 运行 5 分钟
  --html report.html \   # HTML 报告
  --csv results          # CSV 数据 (用于 Grafana 可视化)
```

**关键场景**:

| 场景 | 并发数 | 目的 |
|------|--------|------|
| 基准测试 | 5 | 建立单用户性能基线 |
| 正常负载 | 20 | 验证限流配置是否合理 |
| 压力测试 | 50 | 找到系统容量拐点 |
| 尖峰测试 | 100 (spawn-rate=20) | 验证排队和熔断行为 |
| 长稳测试 | 20 × 30min | 检测内存泄漏 |

### 10.2 ChromaDB 基准测试

**新文件**: `tests/bench_chromadb.py`

**原理**: 在不同并发级别下对 `hybrid_search` 做基准测试，测量延迟分布和 QPS。

```bash
# 运行
BENCH_MAX_CONCURRENCY=10 python tests/bench_chromadb.py

# 输出 JSON 供自动化分析
BENCH_JSON_OUTPUT=bench_results.json python tests/bench_chromadb.py
```

**输出解读**:

```
并发=1   │ avg= 45ms  p95= 68ms  QPS= 22.2   ← 单线程基线
并发=5   │ avg= 82ms  p95=145ms  QPS= 61.0   ← 线性增长
并发=10  │ avg=156ms  p95=289ms  QPS= 64.1   ← QPS 达到峰值（拐点）
并发=15  │ avg=250ms  p95=450ms  QPS= 60.0   ← QPS 开始下降（过载）
并发=20  │ avg=320ms  p95=610ms  QPS= 62.5   ← 性能退化明显
```

---

## 11. 端到端集成测试

**新文件**: `tests/test_integration.py`

**覆盖范围**:

| 测试类 | 用例 | 验证内容 |
|--------|------|---------|
| TestHealthCheck | health/json, metrics, modes | 基础端点可用 |
| TestSSEStream | sse_completes, has_tool_calls, greeting_no_tool | SSE 流完整性和正确性 |
| TestInputValidation | blocked_keywords, empty, max_length | 输入过滤 |
| TestConcurrencyGuard | rate_limit_json | 限流生效 |
| TestRAGASEvaluation | eval_flow | RAGAS 评估管道 |
| TestFeedback | submit_feedback | 反馈提交 |
| TestHITL | exam_interrupt | HITL 中断恢复 |

**运行**:

```bash
# 完整运行
pytest tests/test_integration.py -v -s

# 只跑 SSE 相关
pytest tests/test_integration.py -v -k "sse"

# 跳过需要 API 调用的测试
pytest tests/test_integration.py -v -k "not (test_sse or test_eval or test_hitl)"
```

---

## 12. 运维 Runbook

### 12.1 日常巡检

```bash
# 1. 检查服务健康
curl -s http://localhost:8080/health | jq '.status, .checks.redis.status, .checks.chromadb.status'

# 2. 查看排队深度
curl -s http://localhost:8080/metrics | grep guide_exam_queue_depth

# 3. 查看缓存命中率
curl -s http://localhost:8080/metrics | grep guide_exam_cache_hit_ratio

# 4. 查看最近错误日志
tail -100 chat_logs/*.log | jq 'select(.level=="ERROR") | {trace_id, message}'

# 5. 检查磁盘（eval_log 轮转）
ls -lh eval_log*.jsonl | tail -10
du -sh chroma_db/
```

### 12.2 故障排查

| 症状 | 检查方法 | 可能原因 |
|------|---------|---------|
| 全部返回 "服务不可用" | `/health` → checks.deepseek_api.status | DeepSeek API 故障 → 查看熔断器状态 |
| 检索结果为空 | `/health` → checks.chromadb.status | ChromaDB 数据损坏 → `ls -la chroma_db/` |
| 延迟突然升高 | `/metrics` → P95 延迟 | API 限速、ChromaDB 过载、内存不足 |
| 频繁 429 | `/metrics` → rate_limited_total | 调整 `CONCURRENCY_GLOBAL_RPM` |
| 内存持续增长 | 长稳测试 → locust 30min | Python 内存泄漏、eval_log 未轮转 |
| 线程数持续增长 | `ps aux \| grep python` → thread count | 检查 ThreadPoolExecutor 是否正常复用 |

### 12.3 性能调优

```bash
# 提高全局限流（如果 ChromaDB Benchmark 显示有余量）
export CONCURRENCY_GLOBAL_RPM=100

# 提高本地并发
export CONCURRENCY_LOCAL_MAX_CONCURRENCY=8

# 收紧熔断阈值（快速失败，配合缓存兜底）
export CONCURRENCY_CB_FAILURE_THRESHOLD=3
export CONCURRENCY_CB_TIMEOUT_SECONDS=15

# 缩短排队超时（不满意的用户早点重试）
export CONCURRENCY_QUEUE_TIMEOUT_SECONDS=60

# Agent 执行超时
export AGENT_REQUEST_TIMEOUT_S=90
```

### 12.4 容量规划表

基于 ChromaDB Benchmark 的参考数据（需用实际环境测量，以下为示例）：

| 并发数 | 平均延迟 | P95 延迟 | P99 延迟 | QPS | 单请求 Token | 日费用估算 |
|--------|---------|---------|---------|------|------------|-----------|
| 5 | 82ms | 145ms | 210ms | 61 | ~5K | ¥3.05/天 |
| 10 | 156ms | 289ms | 420ms | 64 | ~5K | ¥3.20/天 |
| 20 | 320ms | 610ms | 890ms | 63 | ~5K | ¥3.15/天 |

建议：**控制在并发 10 以内**，超过时加队列缓冲而非提高并发。

---

## 附录: 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `server/core/structured_logger.py` | 新增 | 结构化日志 + Trace ID |
| `server/core/metrics.py` | 新增 | Prometheus 指标定义 |
| `server/core/executor.py` | 新增 | 全局线程池 + asyncio 桥接 |
| `tests/locustfile.py` | 新增 | HTTP 压力测试 |
| `tests/bench_chromadb.py` | 新增 | ChromaDB 性能基准 |
| `tests/test_integration.py` | 新增 | 端到端集成测试 |
| `server/main.py` | 修改 | 日志初始化、中间件、/metrics、Deep /health、Graceful Shutdown |
| `server/core/tools.py` | 修改 | BM25 线程安全锁、asyncio.run() 修复 |
| `server/services/agent_service.py` | 修改 | 全局线程池复用、请求超时 |
| `server/core/concurrency/__init__.py` | 修改 | ChromaDB 熔断器接入 guard |
| `server/core/eval_logger.py` | 修改 | 按天轮转 + 自动清理 |
| `pyproject.toml` | 修改 | 新增 prometheus-client, psutil 依赖 |
