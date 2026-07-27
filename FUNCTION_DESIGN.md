# AI 导游考试 Agent — 新功能设计文档

> **分支**: `feature/multi-agent-upgrade`
> **实际实现状态**: Multi-Agent、HITL、语义缓存、Langfuse 深度追踪已实现。Agent 行为评估已移除。

---

## 1. 功能一：多 Agent 协作模式

### 1.1 实现方案

**架构**: 单 ReAct Agent 装载全部工具 + Supervisor LLM 预判路由可视化。非 StateGraph 编排。

**为什么不用 StateGraph**: StateGraph stream 输出格式为 `{node_name: state_update}`，与 ReAct Agent 的 `{"tools": [...], "agent": [...]}` 格式不兼容，前端工具调用可视化无法复用。统一为 ReAct 模式后，多 Agent 和单 Agent 共享同一条 stream 处理路径。

### 1.2 Supervisor 预判 (`classify_intent`)

**文件**: `server/core/multi_agent.py` (63 行)

```
用户输入 → Supervisor LLM (独立调用) → JSON 路由决策 → 虚拟 tool 事件
```

独立 LLM 调用分析意图，输出:
```json
{"reasoning": "出卷请求", "worker": "exam_worker", "task_instructions": "..."}
```

决策作为第一个 `event: tool` 发送给前端——"🤖 Supervisor" 工具调用，内容为路由分析摘要。前端折叠面板可展开查看。

### 1.3 Agent 装配

**文件**: `server/core/agent.py` — `get_agent_for_mode("🤖 多Agent协作")`

创建单 ReAct Agent，装载全部工具: 5 检索 + search_questions + grade_answer + 天气 MCP。

System Prompt 追加多 Agent 协作提示: "你现在可以同时使用教材检索、智能出卷、阅卷批改三种能力。"

### 1.4 与单 Agent 模式的区别

| | 单 Agent | 多 Agent |
|---|---|---|
| 工具集 | 按模式限制 (1-5 个) | 全部工具 (8+ 个) |
| Supervisor | 无 | LLM 预判 + 虚拟 tool 展示 |
| HITL | 无 | 有 (出卷后弹窗确认) |
| 适用场景 | 单一任务 | 混合任务 (出卷+解释知识点) |

---

## 2. 功能二：Human-in-the-Loop 出卷确认

### 2.1 实现方式

**纯客户端检测**。不依赖 LangGraph `interrupt()` API。

### 2.2 触发流程

```
sendMessage() → fetch /api/chat/stream → SSE 事件流
  → event: tool "search_questions" → toolRecords 累积
  → event: agent → streamingAnswer 累积
  → event: done (含 full_answer + toolRecords)
      ↓
  toolRecords.some(r => r.name === 'search_questions')
      ↓ 命中
  暂停 stream，不创建永久消息
  弹出 HITL 确认弹窗
      ↓
  ✅ 确认 → 创建永久消息 + saveChatLog
  ✏️ 修改 → 预填 "修改题目数量为N道，重新出卷" 到输入框
  ❌ 取消 → 丢弃所有流式内容
```

### 2.3 为什么用客户端检测

服务端 SSE 流中 `full_answer` 只包含 LLM 生成的文字，不包含 `search_questions` 工具返回的 `ID:单选_1` 等题目信息（这些在 `event: tool` 里）。`toolRecords` 在 `done` 事件中完整传递给前端，判断最可靠。

### 2.4 关键文件

- `static/index.html` — HITL 检测逻辑 (~15 行)、HITL 弹窗 HTML (~12 行)、confirm/modify/cancel 处理函数 (~30 行)
- `server/services/agent_service.py` — 无特殊逻辑，统一走单 Agent stream 路径

---

## 3. 功能三：Langfuse 全链路深度追踪

### 3.1 实现

**文件**: `server/core/agent.py` + `server/core/tools.py`

### 3.2 追踪层级

```
Trace (User Request)
├── @observe: stream_agent_with_retry
├── @observe: hybrid_search / search_textbook / search_questions / grade_answer
├── report_ragas_to_langfuse() → Score: ragas_faithfulness=0.92
└── report_tool_call_to_langfuse() → Score: tool_search_questions=1.0
```

### 3.3 关键改动

- `agent.py`: 新增 `get_langfuse()` 懒加载客户端、`report_ragas_to_langfuse()` 和 `report_tool_call_to_langfuse()` 两个上报函数
- `tools.py`: 4 个 `@tool` 函数加 `@observe(name="xxx")`，兼容 langfuse 新旧版本导入路径
- `llm_service.py`: 已有 `@observe()` + `_record_usage()` 成本追踪

---

## 4. 功能四：语义缓存

### 4.1 架构

详见 [SEMANTIC_CACHE.md](SEMANTIC_CACHE.md)。

### 4.2 核心要点

- ChromaDB 独立集合 `semantic_cache`，零额外依赖
- 余弦相似度 ≥0.95 → 直接返回，跳过检索+精排
- 集成在 `retrieve_with_rerank()` 内，5 种检索策略全部受益
- TTL 24h + FIFO (500 上限) + 覆盖更新
- 所有异常静默降级

### 4.3 监控

`GET /api/agent-eval/cache-stats` → `{hits, misses, hit_rate}`

---

## 5. 文件变更清单

### 新增 (4 个)

| 文件 | 行数 | 功能 |
|------|------|------|
| `server/core/multi_agent.py` | 63 | Supervisor 预判 + HITL 缓存 |
| `server/core/semantic_cache.py` | 201 | 语义缓存层 |
| `prompts/supervisor_prompt.md` | 90 | Supervisor 路由提示词 |
| `FUNCTION_DESIGN.md` | — | 本文档 |

### 修改 (6 个)

| 文件 | 变更 |
|------|------|
| `server/core/agent.py` | +106 行: 多 Agent 模式、Langfuse 客户端、RAGAS 分数上报 |
| `server/core/tools.py` | +48 行: 4 工具 @observe、语义缓存集成到 retrieve_with_rerank |
| `server/services/agent_service.py` | +40 行: 多 Agent Supervisor 预判虚拟 tool、统一 stream 路径 |
| `server/services/eval_service.py` | +5 行: RAGAS 每指标 45s 超时 |
| `server/main.py` | +12 行: 🤖 多Agent协作 模式注册 |
| `static/index.html` | +90 行: 新模式+示例、HITL 弹窗+处理函数、Hitl 确认 |

### 未修改

`server/routes/` 所有路由、`server/core/wrong_book.py`、`server/core/retrieval_utils.py`、`weather_server.py`、`Dockerfile`、`docker-compose.yml` — 全部未动。

---

## 6. 已移除的功能

**Agent 行为评估** (`server/core/agent_eval.py` / `server/routes/agent_eval.py`):
- 10 个标注用例 + 工具选择评分算法 + 趋势追踪 — 开发完成但评估指标数据不稳定
- 前端入口已移除，后端文件保留不加载
