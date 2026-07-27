# 1000 并发用户：LLM 限流、排队、降级、缓存架构

## 背景

当前系统基于 FastAPI + LangGraph + DeepSeek 的导游考试 AI 问答系统。面对 1000 并发用户，DeepSeek API 有 RPM 限制，原有系统仅有一个未生效的 `asyncio.Semaphore(3)`（死代码），**没有分布式限流、请求排队、熔断降级机制**。

**设计原则**：全部基于已有依赖（redis、tenacity、asyncio），不引入 Celery / Kafka 等新组件。Redis 不可用时自动降级为本地模式。

---

## 四层防护架构

```
用户请求
    │
    ▼
┌──────────────────────────────────────────────┐
│  Layer 0: 浏览器端（JS 防抖 300ms + 重复检测）    │  ← 拦截 80% 的无效请求
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Layer 1: 语义缓存（已有，命中则绕过一切）         │  ← ChromaDB 余弦相似度，响应 < 50ms
└──────────────┬───────────────────────────────┘
               │ Miss
               ▼
┌──────────────────────────────────────────────┐
│  Layer 2: 分布式限流（新增 Redis Token Bucket）   │  ← 全局 RPM + 每用户 RPM + 每 IP RPM
└──────────────┬───────────────────────────────┘
               │ 限流未通过 → 返回 429 → 前端排队UI
               │ 通过 / Redis不可用
               ▼
┌──────────────────────────────────────────────┐
│  Layer 3: 熔断器（新增，滑动窗口错误率）            │  ← OPEN → 快速失败 + 兜底缓存
└──────────────┬───────────────────────────────┘
               │ CLOSED / HALF_OPEN
               ▼
┌──────────────────────────────────────────────┐
│  Layer 4: Agent 执行（已有，加全局 Semaphore）     │  ← 最多 N 个并发 Agent（真正打 LLM 的）
└──────────────────────────────────────────────┘
```

---

## 文件结构

### 新增文件 (7)

| 文件 | 说明 |
|---|---|
| `server/core/concurrency/__init__.py` | **ConcurrencyManager** 编排器，串联限流→排队→熔断 |
| `server/core/concurrency/config.py` | 配置 dataclass + 全环境变量覆盖 |
| `server/core/concurrency/redis_scripts.py` | GCRA + 三层联合检查 Lua 脚本（Redis 原子执行） |
| `server/core/concurrency/rate_limiter.py` | Redis Token Bucket + 本地 asyncio.Semaphore 降级 |
| `server/core/concurrency/circuit_breaker.py` | 滑动窗口熔断器 (CLOSED→OPEN→HALF_OPEN) |
| `server/core/concurrency/request_queue.py` | Redis Sorted Set 优先级排队 + 本地 asyncio.Queue 降级 |
| `server/middleware/concurrency.py` | 路由 guard 函数（缓存→限流→排队→熔断） |

### 修改文件 (4)

| 文件 | 改动 |
|---|---|
| `server/main.py` | lifespan 中初始化 ConcurrencyManager；新增 `/api/queue/status` SSE 端点；health 增加 concurrency 状态 |
| `server/routes/chat.py` | `/api/chat/stream` 接入 guard 链；流结束后 release |
| `.env` | 新增 20 个并发控制配置项 |
| `static/index.html` | 防抖 300ms + 429 排队/限流处理 + queue SSE 事件 + 排队进度 UI |

---

## 架构数据流

```
用户点击发送
  → JS 防抖 300ms（重复点击跳过）
  → POST /api/chat/stream
  → chat_guard()
    → Layer 1: 语义缓存（命中 → 直接返回）
    → Layer 2: 熔断检查（OPEN → 兜底缓存/503）
    → Layer 3: 限流（GCRA Token Bucket, global+user+ip 三层）
      → 通过：acquire_local() 槽位 → 放行
      → 不通过 → 入队 / 返回 429 + Retry-After
    → Layer 4: Agent 执行（已有流程不变）
      完成后 → release_local() + record_result() 更新熔断器
```

---

## 各层详细设计

### Layer 0: 浏览器端防抖

```javascript
// 300ms 内相同问题不重复发送
const now = Date.now();
if (pendingRequest.value && pendingRequest.value.prompt === prompt &&
    now - pendingRequest.value.ts < 300) {
  return;
}
```

### Layer 1: 语义缓存（已有）

`server/core/semantic_cache.py`：ChromaDB 余弦相似度 ≥0.95 命中直接返回，无需任何改动。

### Layer 2: 分布式限流

**算法：GCRA（Generic Cell Rate Algorithm）**

- Redis Lua 脚本保证原子性
- 三层限流：全局 RPM (50)、每用户 RPM (5)、每 IP RPM (10)
- 支持突发容忍（burst）
- Redis 不可用 → 降级为本地 `asyncio.Semaphore(5)`

```
Redis Key: rl:global:deepseek, rl:user:{id}, rl:ip:{hash}
Value: TAT (Theoretical Arrival Time)
```

**为什么不用固定窗口计数器？** 计数器在窗口边界会尖刺。GCRA 只需存一个 TAT，O(1) 内存，边界平滑。

### Layer 3: 熔断器

**状态机：CLOSED → OPEN → HALF_OPEN → CLOSED**

```
CLOSED（正常）                     OPEN（熔断）                     HALF_OPEN（探测）
┌──────────┐   错误数≥阈值        ┌──────────┐   timeout后         ┌──────────┐
│ 请求正常  │ ────────────────→   │ 快速拒绝  │ ────────────────→   │ 放行1个   │
│ 通过     │                     │ 返回兜底  │                    │ 请求     │
└──────────┘                     └──────────┘                    └──────────┘
      ↑                                ↑                              │
      │                                │                成功 → CLOSED │
      └────────────────────────────────┘                失败 → OPEN   │
              错误率恢复正常                                                │
```

配置：
- 滑动窗口 60 秒，窗口内 5 次错误 → OPEN
- OPEN 30 秒后 → HALF_OPEN（允许 1 个探测请求）
- 探测成功 → CLOSED，失败 → 回到 OPEN

**熔断打开时的降级**：不直接 503，而是先尝试缓存兜底。

### Layer 4: 优先级排队

**Redis Sorted Set**：score = priority × 10^12 + timestamp（越小越优先）

- 高优先级 (0)：付费用户 / HITL 恢复
- 普通优先级 (5)：默认用户
- 低优先级 (10)：爬虫/高频用户

**入队流程**：
```
POST /api/chat/stream → 429 + {"error": "queued", "queue_position": 15, "queue_token": "xxx"}
前端展示"前方 15 人，预计 30 秒"
GET /api/queue/status?token=xxx → SSE 推送位置变化
```

---

## 配置项

```bash
# .env 新增
# ── 限流 ──
CONCURRENCY_RATE_LIMIT_ENABLED=true
CONCURRENCY_GLOBAL_RPM=50
CONCURRENCY_GLOBAL_BURST=10
CONCURRENCY_PER_USER_RPM=5
CONCURRENCY_PER_USER_BURST=2
CONCURRENCY_PER_IP_RPM=10
CONCURRENCY_PER_IP_BURST=3

# ── 排队 ──
CONCURRENCY_QUEUE_ENABLED=true
CONCURRENCY_QUEUE_MAX_SIZE=200
CONCURRENCY_QUEUE_TIMEOUT_SECONDS=120
CONCURRENCY_QUEUE_POLL_INTERVAL=1.5

# ── 熔断 ──
CONCURRENCY_CB_ENABLED=true
CONCURRENCY_CB_FAILURE_THRESHOLD=5
CONCURRENCY_CB_WINDOW_SECONDS=60
CONCURRENCY_CB_TIMEOUT_SECONDS=30
CONCURRENCY_CB_HALF_OPEN_MAX=1

# ── 本地降级（Redis 不可用时）──
CONCURRENCY_LOCAL_MAX_CONCURRENCY=5
```

---

## Redis 不可用的全链路降级

所有组件遵循同一模式：Redis 可用 → 分布式精度；Redis 不可用 → 本地内存模式 + WARNING 日志。

| 组件 | Redis 模式 | 本地降级 |
|---|---|---|
| RateLimiter | GCRA Lua 脚本（多进程安全） | `asyncio.Semaphore(local_max_concurrency)` |
| CircuitBreaker | Redis Hash 同步状态 | 实例变量（单进程） |
| RequestQueue | Sorted Set + Lua 原子出队 | `asyncio.Queue`（单进程） |

---

## 测试结果

```
[PASS] Config — 默认值和环境变量加载正确
[PASS] RateLimiter — 本地 Semaphore 模式：槽位用完正确拒绝
[PASS] CircuitBreaker — CLOSED → OPEN 转换：5 次错误触发熔断
[PASS] RequestQueue — 入队 / 位置查询 / 出队正常
[PASS] ConcurrencyManager guard — 通过 + 限流 + 熔断三种路径正确
[PASS] Stress 100 并发 — passed=5, limited=95（local_max_concurrency=5）
```

100 并发测试中，5 个 semaphore 槽位正确限制，其余 95 个被限流（排队或直接拒绝），验证限流机制有效。Docker 部署时 Redis 模式将提供分布式精度。

---

## 降级级别

| 级别 | 触发器 | 行为 |
|---|---|---|
| **FULL** | 正常运行 | 全功能 |
| **THROTTLED** | 队列深度 > 100 | 限流收紧 |
| **DEGRADED** | 熔断打开 | 仅缓存，非缓存查询返回"服务繁忙" |
| **MINIMAL** | Redis + LLM 连续失败 | 仅健康检查 + 静态文件 |

---

## 前端交互

1. **防抖**：300ms 内相同问题不重复发请求
2. **429 限流**：展示"请求频率过高，N 秒后可重试"
3. **排队模式**：展示排队进度条 + 预估等待时间
4. **熔断降级**：展示"AI 服务暂时繁忙，以下为缓存结果"
5. **缓存命中**：展示"⚡ 来自缓存"
