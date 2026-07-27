# 多轮对话上下文窗口管理

## 问题背景

LangGraph `create_react_agent` + `RedisSaver` checkpointer 的默认行为：**将所有历史消息一字不落地拼进 LLM 上下文**。ReAct Agent 每轮 tool_calls 往返平均产生 3000+ tokens 的工具返回内容（教材段落检索结果），实际 **15~20 轮对话就可能撑爆 DeepSeek V4 的 128K 上下文窗口**。

当前系统在以下三个点上没有任何处理：

| # | 问题 | 原始状态 | 后果 |
|---|---|---|---|
| 1 | Redis checkpoint TTL | `RedisSaver(ttl=None)`，永不过期 | 废弃 checkpoint 永久占用 Redis 内存 |
| 2 | 超出上下文窗口 | 无检测、无裁剪、无降级 | DeepSeek 返回 400，前端显示"Agent 调用失败"，对话彻底断掉 |
| 3 | 无摘要压缩 | 不存在滑动窗口或 LLM 摘要 | 旧对话永远堆积在上下文中 |

---

## 修复方案

### 新增文件

`server/core/context_manager.py` — 上下文窗口管理模块

### 修改文件

| 文件 | 改动 |
|---|---|
| `server/core/agent.py` | RedisSaver 加 TTL；新增 `BadRequestError` 上下文超限检测 + 自动裁剪重试 |
| `server/services/agent_service.py` | 在 `agent.stream()` 之前主动检查并裁剪 checkpoint |
| `.env` | 新增 7 个配置项 |

---

## 实现细节

### 1. Checkpoint TTL

```python
# agent.py:55-58 — RedisSaver 加 7 天 TTL
_memory = RedisSaver(redis_url=REDIS_URL, ttl=CHECKPOINT_TTL_SECONDS)
# CHECKPOINT_TTL_SECONDS = 604800 (7天)
```

7 天后未活跃的会话 checkpoint 自动被 Redis 清理，防止废弃数据无限堆积。

### 2. 上下文窗口管理总流程

```
agent_service.stream_chat()
  │
  ├─ 读取 checkpoint state
  │     ↓
  ├─ estimate_messages_tokens() 估算总 token 数
  │     ↓
  ├─ < SOFT_LIMIT ? → 放行，正常 stream
  │     ↓
  ├─ > SOFT_LIMIT ? → manage_context() 裁剪
  │     │
  │     ├─ 保留 SystemMessage（必选）
  │     ├─ 从后往前保留最近 ~20 条消息
  │     ├─ 过长的 ToolMessage 截断（>3000 字符）
  │     ├─ 被丢弃的消息 → LLM 压缩为 ~300 字摘要
  │     │    └─ 摘要失败 → 降级为纯文本截断
  │     └─ agent.update_state() 写回 Redis
  │           │
  └─ agent.stream() ← LangGraph 加载裁剪后的状态
```

### 3. Token 估算

```python
def estimate_tokens(text: str) -> int:
    """中文 ~1.5 字符/token, 英文 ~3.5 字符/token (BPE)"""
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    other = len(text) - chinese
    return max(1, int(chinese / 1.5 + other / 3.5))
```

### 4. 滑动窗口裁剪

```python
# 从后往前取消息，直到接近 soft_limit 的 80%
available = soft_limit - system_tokens - margin  # margin=20000
for msg in reversed(normal_msgs):
    if kept_tokens + mt > min(available, soft_limit * 0.8):
        if len(kept) >= MIN_RECENT_KEEP:  # 至少保留 20 条
            break
    kept.insert(0, msg)
    kept_tokens += mt
```

### 5. 工具返回截断

ToolMessage 是上下文膨胀的主要原因（单条可达 5000+ 字符）。超出 3000 字符时保留头部 60% + 尾部 40%，中间省略。

```python
truncated = content[:1800] + "\n\n... [省略] ...\n\n" + content[-1200:]
```

### 6. LLM 摘要压缩

被滑动窗口丢弃的消息不直接删除，而是调用 LLM 压缩为一段 ≤300 字的摘要，以 `SystemMessage` 形式注入到保留消息之前。LLM 调用失败时降级为纯文本截断。

### 7. 运行时超限兜底

即使 `agent_service.py` 的主动裁剪没触发，如果 DeepSeek 仍然返回上下文超限错误（400），`stream_agent_with_retry` 也会检测错误消息中的关键词，自动裁剪 checkpoint 后重试：

```python
# agent.py:207-228 — stream_agent_with_retry 的上下文超限检测
CONTEXT_OVERFLOW_KEYWORDS = [
    "context length", "too long", "maximum context",
    "context_length_exceeded", "maximum token", "token limit",
]

if is_context_overflow and attempt < 2:
    metadata = trim_checkpoint_state(agent, config)
    # 裁剪后重试
    continue
```

---

## 配置项

```bash
# .env
# ── 上下文窗口管理 ──
CONTEXT_MAX_TOKENS=128000          # 模型最大输入 token 数
CONTEXT_SAFE_RATIO=0.75            # 安全比例（75% → 96K 软上限）
CONTEXT_MIN_RECENT_KEEP=20         # 滑动窗口最少保留的近期消息数
CONTEXT_SUMMARIZE_ENABLED=true     # 超限时是否调用 LLM 生成摘要
CONTEXT_MAX_TOOL_CHARS=3000        # 单条工具返回最大字符数
CONTEXT_MAX_SUMMARY_CHARS=300      # 摘要最大字数
CHECKPOINT_TTL_SECONDS=604800      # Redis checkpoint 过期时间（秒）
```

---

## 测试结果

```
[Test 1] Token estimation
  Chinese (21 chars): 12 tokens
  English (43 chars): 12 tokens
  Mixed (64 chars):  25 tokens

[Test 2] Message token estimation
  4 messages: 268 tokens, soft_limit=96000

[Test 3] Tool message truncation
  Truncated: 2000 chars → under limit (no action)

[Test 4] manage_context — under limit
  Trimmed=False, tokens=24

[Test 5] manage_context — over limit (simulate)
  Before: 601 messages, 42808 tokens
  After:  22 messages, 2030 tokens       ← 裁剪率 96%
  Trimmed=True, Summarized=True, Discarded=580

[Test 6] Checkpoint TTL
  TTL = 604800s = 7 days
```

---

## 行为矩阵

| 场景 | 触发条件 | 行为 |
|---|---|---|
| 正常对话 | total_tokens < 96K | 不做任何处理，正常 stream |
| 接近上限 | total_tokens ≥ 96K | 裁剪旧消息 + 写回 checkpoint，然后正常 stream |
| 工具返回过长 | 单条 ToolMessage > 3000 字符 | 保留头尾，中间截断 |
| 已超限但未触发 | DeepSeek 返回 400 错误 | stream_agent_with_retry 检测关键词，裁剪后自动重试 |
| LLM 摘要失败 | ChatOpenAI 调用异常 | 降级为纯文本截断 |
| 会话闲置 | 7 天无活动 | Redis checkpoint 自动过期清理 |
