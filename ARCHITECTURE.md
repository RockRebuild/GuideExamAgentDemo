# AI 导游考试 Agent — 系统架构文档

> 基于 LangGraph + ChromaDB + BGE-Reranker 的 Multi-Agent RAG 系统。支持教材检索、智能出卷、阅卷批改、错题本管理、天气查询（MCP），Human-in-the-Loop 出卷确认，语义缓存加速。

---

## 1. 系统总览

### 1.1 功能与痛点

| 痛点 | 方案 | 对应功能 |
|---|---|---|
| 教材内容庞杂，查找困难 | 5 种检索策略 + BGE 精排 | 📖 教材知识问答 |
| 刷题需求大，出卷耗时 | 7000+ 题库按章节/题型过滤 | 📝 智能出卷 |
| 自主练习无法判断对错 | 自动批改 + 解析 + 知识点推荐 | 📊 阅卷批改 |
| 错题分散，复习无针对性 | 自动记录 + 按科目/次数统计 | 📒 错题本 |
| 高频问题重复检索 | 语义缓存，相似度 ≥0.95 直接复用 | 💾 语义缓存 |
| 出卷结果不可控 | 出卷后弹窗确认，用户可审核/修改 | 🛑 HITL |

### 1.2 四种模式

| 模式 | 工具集 | Agent 架构 |
|------|--------|-----------|
| 📖 教材知识问答 | 5 种检索 + 天气 MCP | 单 ReAct Agent |
| 📝 智能出卷 | search_questions + 天气 | 单 ReAct Agent |
| 📊 阅卷批改 | grade_answer + search_textbook + 天气 | 单 ReAct Agent |
| 🤖 多Agent协作 | 全部工具（检索+出卷+批改+天气） | 单 ReAct Agent + Supervisor 预判 |

### 1.3 全局请求流

```
用户输入（Vue3 SPA）
  → FastAPI SSE (/api/chat/stream)
  → 输入净化 + 孤儿 tool_calls 检测
  → [多Agent模式] Supervisor LLM 意图分类 → 虚拟 tool 事件展示路由决策
  → LangGraph ReAct Agent (DeepSeek V4 Flash)
  → 工具调用（检索/出卷/批改/天气）→ BGE 精排 → 语义缓存
  → SSE 流式输出 (event: tool / agent / done / hitl)
  → [出卷完成] HITL 暂停 → 前端确认弹窗 → 用户确认/取消
```

---

## 2. 架构设计

### 2.1 分层架构

```
表示层     static/index.html — Vue3 SPA, SSE streaming, dark theme
接入层     server/main.py — FastAPI, lifespan, 8 路由模块
服务层     agent_service.py (流式编排) / eval_service.py (RAGAS) / qb_service.py (题库)
领域层     agent.py (Agent 创建) / tools.py (7 工具) / retrieval_utils.py (精排)
           wrong_book.py (错题本) / llm_service.py (成本) / multi_agent.py (路由)
           semantic_cache.py (缓存) / eval_logger.py (日志)
基础设施   ChromaDB (4 集合 + 缓存集合) / Redis (checkpoint/反馈/费用/错题本)
           DeepSeek V4 Flash / DashScope Embedding / BGE-Reranker / BM25+jieba
```

### 2.2 Docker 三服务

```
Redis (6379) ←→ guide-agent (8080→8501) ←→ weather-mcp (8000)
  2.08GB 镜像    Python + LangGraph + ChromaDB     ~200MB FastAPI
```

---

## 3. 核心模块详解

### 3.1 Agent 引擎

**文件**: `server/core/agent.py`

- **按模式动态创建 Agent**: 四种模式加载不同工具集，避免单一 Agent 工具膨胀。实例缓存复用
- **Redis Checkpoint**: 通过 `RedisSaver` 持久化对话状态。四种模式独立 `thread_id` 互不干扰
- **孤儿 tool_calls 修复**: 检测 checkpoint 中残留的未完成工具调用 → 自动创建新 thread_id 绕过
- **天气 MCP 重试**: 依次尝试 Docker 容器名 → localhost → 127.0.0.1
- **重试策略**: tenacity 指数退避 (RateLimit/APIConnection/InternalServer/APITimeout，最多 3 次)

### 3.2 检索系统

**文件**: `server/core/tools.py`

**向量库**: `guide_child`(段落) / `guide_summary`(摘要) / `guide_sentence`(句子) / `guide_parent`(父切片)

| 策略 | 工具 | 适用场景 | 方式 |
|------|------|---------|------|
| 语义搜索 | `search_textbook` | 一般事实查询 | ChromaDB 余弦相似度 + 分数过滤 |
| 混合检索 | `hybrid_search` | 专有名词、法律条文 | 语义 + BM25(jieba 分词) → 合并去重 |
| 多粒度并行 | `multi_search` | 章节概览 | 摘要+段落+句子三层并行 |
| 父子切片 | `parent_child_search` | 需要完整上下文 | 子切片匹配 → 父切片返回 |
| 改写检索 | `rewritten_search` | 模糊口语化问题 | LLM 改写 3 变体 + 原问题 → 并行检索 |

**BM25**: 用 guide_parent (800 字符) 构建索引，jieba 分词，与语义检索互补。

### 3.3 BGE-Reranker 精排

**文件**: `server/core/retrieval_utils.py`

策略: 拆句 → BGE 逐句打分 → 段落分 = max(句分) → 过滤低分 → 排序去重 → Top-K

- **为什么拆句**: 一个段落 80% 无关内容会拖低整体分；拆句后每句独立打分，取最高分避免稀释
- **动态 Top-K**: 列举类 15、对比类 12、流程类 10、短事实 3、默认 5
- **OOM 三层防护**: 启动前内存检查 → 推理时 sub_batch=4 + 张量释放 → OOM 降级为返回原始上下文

### 3.4 工具系统

**文件**: `server/core/tools.py`

7 个 LangChain `@tool` + 1 个 MCP 动态工具:

```
search_textbook / hybrid_search / multi_search / parent_child_search / rewritten_search
search_questions (题库过滤) / grade_answer (批改 + 错题自动记录) / get_weather (MCP)
```

MCP 工具通过 HTTP 发现: `POST /mcp tools/list` → `StructuredTool.from_function()` + JSON Schema → Pydantic args_schema。

### 3.5 MCP 天气服务

**文件**: `weather_server.py` + `Dockerfile.mcp`

独立 FastAPI 服务 (8000 端口)，实现 MCP 协议的 `tools/list` 和 `tools/call`，调用 wttr.in API。仅需 4 个依赖 (~200MB 镜像)。

### 3.6-3.8 三种单 Agent 模式

知识问答/智能出卷/阅卷批改的工作流、System Prompt 约束、防编造机制、错题自动记录，详见 `prompts/system_prompt.md` 和 3.1 节。

### 3.9 错题本

**文件**: `server/core/wrong_book.py` + `server/routes/wrong_book.py`

- Redis Hash 持久化 (`wrongbook:items`)
- 双重检测: 工具调用检测 (精确) + LLM 文本解析 (兜底)
- 同一 question_id 重错 → wrong_count 累加，按错误次数降序排列

### 3.10 流式通信

**文件**: `server/services/agent_service.py`

桥接方案: `asyncio.Queue` + `ThreadPoolExecutor`。同步 LangGraph stream 在独立线程运行 → `call_soon_threadsafe` 投递到队列 → FastAPI async handler 消费。

SSE 事件: `tool` / `agent` / `done` / `error` / `hitl`

每个模式独立 `thread_id`，对话历史隔离。

### 3.11 质量评估 (RAGAS)

**文件**: `server/services/eval_service.py`

四维指标: Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall。
DeepSeek V4 Pro 做 Judge LLM。异步任务模式: POST 启动 → 轮询 GET。结果写入 `eval_log.jsonl`。

### 3.12 成本监控

**文件**: `server/core/llm_service.py`

按日+模型聚合 token 用量 → Redis Hash (48h TTL)。5 日趋势 + 预算告警 (>70%)。支持 DeepSeek V4 Flash/Pro + DashScope Embedding 三种模型独立计费。

### 3.13 多 Agent 协作

**文件**: `server/core/multi_agent.py` + `server/core/agent.py`

**设计**: 非 StateGraph 编排，而是单 ReAct Agent 加载全部工具（5检索+出卷+批改+天气），配合 Supervisor LLM 预判路由可视化和 HITL 出卷确认。

**Supervisor 预判** (`classify_intent`):
- 独立 LLM 调用分析用户意图，输出 `{worker, reasoning, task_instructions}` JSON
- 路由结果作为虚拟 tool 事件 `🤖 Supervisor` 发送给前端
- 前端工具调用折叠面板展示: "🔀 Supervisor 路由决策 → Worker: exam_worker"
- 预判失败不影响主流程，fallback 到 retrieval_worker

**对比单 Agent 模式**: 多 Agent 模式装载全部工具，Agent 可在一次对话中自行决定检索/出卷/批改，无需切换模式。工具膨胀通过 System Prompt 的优先级规则缓解。

### 3.14 Human-in-the-Loop

**实现方式**: 纯客户端检测，不依赖 LangGraph `interrupt()`。

```
done 事件到达 (前端)
  → 检查 toolRecords 是否包含 search_questions 调用
  → 是 → 暂停 stream，弹出确认弹窗
  → ✅ 确认 → 固化消息，保存 chat log
  → ✏️ 修改 → 预填修改指令到输入框
  → ❌ 取消 → 丢弃流式内容
```

**设计考量**: 服务端 SSE 流中的 `full_answer` 只包含 LLM 生成的文字，不包含工具返回的题目 ID 信息。`toolRecords` 在 `done` 事件中完整可用，客户端判断最可靠。

### 3.15 语义缓存

**文件**: `server/core/semantic_cache.py`

详见 [SEMANTIC_CACHE.md](SEMANTIC_CACHE.md)。

- 复用 ChromaDB 五个集合之一 (`semantic_cache`)，零额外依赖
- 余弦相似度 ≥0.95 → Cache HIT，跳过检索 + 精排整条链路
- TTL 24h + FIFO 淘汰 (500 条上限) + 覆盖更新
- 所有异常静默降级，缓存失败不影响主流程
- 集成在 `retrieve_with_rerank()` 内，5 种检索策略全部受益
- 监控: `GET /api/agent-eval/cache-stats` 返回命中率

---

## 4. 数据层设计

### ChromaDB
```
chroma_db/
├── guide_child/       # 段落 (200 字符)
├── guide_summary/     # 摘要
├── guide_sentence/    # 句子
├── guide_parent/      # 父切片 (800 字符)
└── semantic_cache/    # 语义缓存
```

### Redis
```
Checkpoint (LangGraph 内部) / token_usage:{date}:{model} (Hash, 48h TTL)
feedback:list (List) / wrongbook:items (Hash)
```

### 题库
`question_bank.json` — 7000+ 题 JSON 文件 (无需数据库)

---

## 5. 前端架构

**文件**: `static/index.html` (~1450 行 Vue3 SPA)

- 暗色主题，模仿 Streamlit 风格
- 4 种模式单选侧边栏 + 示例问题按钮 + 题库浏览器 (阅卷模式)
- SSE 流式渲染 + 工具调用折叠面板
- HITL 确认弹窗 (出卷后自动弹出)
- 错题本面板 (按科目筛选 + 删除)
- RAGAS 评估面板 (四项指标卡片)
- API 费用面板 (按模型明细 + 5 日趋势)
- 响应式适配 (768px / 480px 断点)

---

## 6. 部署架构

Docker Compose 三服务 + bridge 网络:

| 服务 | 端口 | 镜像 |
|------|------|------|
| Redis | 6379 | redis-stack-server |
| Agent | 8501→8080 | 2.08GB (Python + LangGraph + ChromaDB + BGE) |
| Weather-MCP | 8000 | ~200MB (FastAPI + httpx) |

**镜像优化**: 12.6GB → 7.8GB → 2.08GB (去 CUDA torch → pip CPU 版)

ECS 配置: 2C4G, 40G SSD

---

## 7. 关键技术决策

| 决策 | 原因 |
|------|------|
| LangGraph 而非 LangChain AgentExecutor | Checkpoint 持久化 + 孤儿修复 |
| 5 种检索策略而非单一 | 弥补纯语义搜索在精确匹配/长下文/口语化场景的盲区 |
| BGE-Reranker CPU 推理 | ECS 无 GPU，CPU 推理 + sub_batch=4 可控 |
| 四种模式独立 thread_id | 工具集不同，checkpoint 混用会导致工具冲突 |
| JSON 而非数据库存题库 | 7000 题几 MB，查询仅按字段过滤 |
| 语义缓存用 ChromaDB 而非 Redis | 需要语义级匹配（同义查询），精确 key 匹配做不到 |
| HITL 用客户端检测 | 服务端 full_answer 不含工具返回内容，toolRecords 只在 done 事件中可用 |
| 多 Agent 用单 ReAct + 预判 | StateGraph stream 格式与单 Agent 不兼容，统一为 ReAct 简化维护 |

---

## 8. 安全与容错

| 场景 | 策略 |
|------|------|
| Redis 不可用 | 反馈/费用/错题本降级，知识问答正常 |
| 天气 MCP 不可用 | 从 prompt 移除天气描述 |
| BGE-Reranker OOM | 三层防护 → 降级为原始上下文 |
| LLM API 限流 | tenacity 指数退避重试 |
| 孤儿 tool_calls | 自动检测 + 新 thread_id |
| 输入注入 | 500 字符截断 + 敏感词过滤 |
| 题库泄露 | API 不返回 answer 字段 |
| 缓存故障 | 所有异常静默降级，不影响检索 |

---

## 文件清单

```
server/
├── main.py                    FastAPI 入口 + 路由注册
├── state.py                   ContextVar 请求级状态
├── core/
│   ├── agent.py               Agent 创建 + 重试 + 孤儿修复 + Langfuse
│   ├── tools.py               7 工具 + MCP 加载 + 语义缓存集成
│   ├── retrieval_utils.py     BGE-Reranker 精排 + OOM 防护
│   ├── multi_agent.py         Supervisor 预判 + HITL 缓存
│   ├── semantic_cache.py      语义缓存层
│   ├── llm_service.py         LLM 调用 + 成本核算
│   ├── wrong_book.py          错题本 Redis CRUD + LLM 文本检测
│   └── eval_logger.py         JSONL 评估日志
├── routes/
│   ├── chat.py                SSE 流式聊天
│   ├── wrong_book.py          错题本 API
│   ├── evaluation.py          RAGAS 评估
│   ├── question_bank.py       题库查询
│   ├── feedback.py            用户反馈
│   ├── cost.py                费用查询
│   └── chat_log.py            对话日志
├── services/
│   ├── agent_service.py       流式编排 + HITL + 多 Agent
│   ├── eval_service.py        异步 RAGAS
│   └── qb_service.py          题库分页
└── models/schemas.py          Pydantic 模型
static/index.html              Vue3 SPA
weather_server.py              MCP 天气服务
prompts/system_prompt.md       Agent 系统提示词
prompts/supervisor_prompt.md   Supervisor 路由提示词
```
