# AI 导游考试 Agent - RAG 智能问答系统

基于 LangGraph + ChromaDB + BGE-Reranker 的 Multi-Agent RAG 系统，支持教材知识检索、智能出卷、阅卷批改、错题本管理。独立完成从架构设计到云端部署的全流程。

---

## 功能概览

### 📖 教材知识问答
- 从导游考试教材中检索知识点，**5 种检索策略**自适应选择
- **BGE-Reranker 精排**，过滤无关上下文，减少 LLM 幻觉
- 每条回答标注教材来源（章节、页码）

### 📝 智能出卷
- 按科目、章节、题型从题库抽题，支持单选 / 多选 / 判断
- 自动批改 + 详细解析 + 知识点推荐
- **错题自动记录到错题本**，支持按科目筛选、手动移除
- **人在回路确认**：出卷后弹窗预览，用户可确认/修改/取消

### 📊 阅卷批改
- 按题号或章节批量批改
- 每题判断对错 + 解析，答错自动推荐相关知识点复习

### 🤖 多 Agent 协作
- **Supervisor LLM** 预判用户意图，路由到最适合的 Worker（检索/出卷/批改）
- 路由决策以虚拟工具调用形式可视化展示在前端
- 单 ReAct Agent 装载全部工具，4 种模式共享统一的 stream 路径

### 🌤 天气查询
- 基于 **MCP 协议**的自定义天气服务，支持当天及未来 3 天预报
- 展示 Agent 工具调用 + 外部服务集成的能力

### ⚡ 语义缓存
- 基于 ChromaDB 余弦相似度的查询缓存（相似度 ≥ 0.95 命中）
- 24h TTL + FIFO 淘汰（上限 500 条），写入时覆盖更新
- 零额外依赖，5 条检索管道全部透明接入
- 详见 [SEMANTIC_CACHE.md](SEMANTIC_CACHE.md)

---

## 系统架构

```
                   Web 前端 (Vue3)  ←─ SSE 流式 ─→  FastAPI 后端
                                                        │
         ┌──────────────────────────────────────────────┼──────────────────────────┐
         │          LangGraph ReAct Agent               │     Supervisor LLM       │
         │              (统一 stream)                    │    (意图分类 + 路由预判)    │
         │                      │                       │                          │
         │     DeepSeek V4 Flash (LLM)                  │   模式:                   │
         │                      │                       │   📖 知识 / 📝 出卷        │
         │        ┌─────────────┴─────────────┐        │   📊 批改 / 🤖 多Agent    │
         │        │       Tool Calling         │        │                          │
         │        └─────────────┬─────────────┘        │   人在回路:                │
         │   ┌──────────────────┼──────────────────┐   │   客户端检测 + 弹窗确认      │
         │   ▼                  ▼                   ▼   │                          │
         │ 5种检索策略     出卷/批改/天气         MCP Server                       │
         └──────────────────┬──────────────────────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────────┐
    │                       │                           │
  ChromaDB 向量库    BGE-Reranker 精排    Redis (会话/反馈/错题本/费用)
(DashScope Embedding)  (CPU, sub-batch)    + Semantic Cache (ChromaDB)
```

---

## 人在回路 (Human-in-the-Loop)

出卷后弹窗确认，用户可在三个选项中操作：

| 操作 | 行为 |
|------|------|
| ✅ 确认出卷 | 将试卷写入聊天记录，保存日志 |
| ✏️ 修改 | 丢弃结果，预填修改指令供用户编辑 |
| ❌ 取消 | 丢弃所有流式内容 |

实现方式：客户端 SSE stream 完成后检测 `search_questions` 工具调用 → 阻止消息入库 → 弹窗 → 用户选择后决定是否保存。无需 LangGraph `interrupt()` API，4 种模式共享同一套 stream 处理路径。

**设计决策**：StateGraph 的 `interrupt()` 需要多节点图编排，其 stream 格式 `{node_name: state_update}` 与 ReAct Agent 的 `{"tools": [...], "agent": [...]}` 不兼容，会要求前端维护两套渲染逻辑。客户端检测方案 10 行代码实现同等体验，零后端改动。

---

## 技术栈

| 分类 | 技术 | 说明 |
|---|---|---|
| Agent 框架 | LangGraph | ReAct Agent + Redis checkpoint 多轮记忆 |
| Multi-Agent | Supervisor LLM 预判 | 独立 LLM 调用做意图分类 + 路由可视化 |
| LLM | DeepSeek V4 Flash | 通过 OpenAI 兼容 API 调用 |
| 向量数据库 | ChromaDB | 阿里云 DashScope text-embedding-v4 |
| 语义缓存 | ChromaDB 余弦相似度 | 查询级缓存，相似度 ≥0.95 命中 |
| 精排模型 | BGE-Reranker-base | CPU 推理，子批量处理防 OOM |
| 关键词检索 | BM25 (rank-bm25 + jieba) | 与语义检索互补 |
| 后端 | FastAPI + SSE | 流式响应，异步非阻塞 |
| 前端 | Vue3 单页应用 | 实时展示工具调用过程 |
| 评估 | RAGAS 四指标 + Agent 行为评估 | 检索质量 + 工具选择准确率双重评估 |
| 监控 | Langfuse | Token 用量 + 全链路追踪 + 成本预警 |
| 部署 | Docker Compose | 三服务：agent / redis / weather-mcp |

---

## 5 种检索策略

| 策略 | 适用场景 | 核心技术 |
|---|---|---|
| `search_textbook` | 一般事实查询 | 语义搜索（余弦相似度） |
| `hybrid_search` | 专有名词、法律条文 | 语义 + BM25 关键词混合 |
| `parent_child_search` | 需要完整段落上下文 | 子切片匹配 → 父切片返回 |
| `rewritten_search` | 模糊、口语化问题 | LLM 改写多路查询 → 合并去重 |
| `multi_search` | 章节概览 | 摘要 + 段落 + 句子三层并行 |

检索后统一经过 **BGE-Reranker 精排**：拆句 → 逐句打分 → 段落级聚合 → 过滤低分噪音 → 取 Top-K，确保 LLM 只接收高质量上下文。

---

## 评估体系

### RAGAS 检索质量评估
- 4 项指标：Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall
- DeepSeek V4 Pro 作为判定 LLM
- 异步作业模型：POST 触发 → 轮询 GET → 结果写入 `eval_log.jsonl`

### Agent 行为评估
- **10 个标注测试用例**，覆盖知识问答、出卷、批改、越狱防护
- 评估工具选择准确率（A/B/C/F 四级）+ 端到端成功率
- 支持按分类和难度聚合，历史报告趋势分析
- API：`POST /api/agent-eval/run` 触发全量评估，`GET /api/agent-eval/reports` 查看报告

---

## 可观测性

- **Langfuse 全链路追踪**：Agent 决策 + 工具调用 + LLM 耗时，`@observe` 装饰器自动采集
- **Token 用量**：每次请求的 input/output tokens 通过 API `usage` 字段提取
- **成本监控**：按天 + 按模型聚合到 Redis，5 天趋势 + 70% 预算预警
- **语义缓存命中率**：实时 hit/miss 统计，API 可查询

---

## 关键设计决策

| 决策 | 原因 |
|------|------|
| LangGraph 而非 LangChain AgentExecutor | Checkpoint 持久化 + 孤儿 tool_calls 修复 |
| 5 种检索策略而非单一 | 弥补纯语义搜索在精确匹配/长下文/口语化场景的盲区 |
| BGE-Reranker CPU 推理 | ECS 无 GPU，CPU 推理 + sub_batch=4 可控 |
| 四种模式独立 thread_id | 工具集不同，checkpoint 混用会导致工具冲突 |
| 语义缓存用 ChromaDB 而非 Redis | 需要语义级匹配（同义查询），精确 key 匹配做不到 |
| HITL 用客户端检测 | StateGraph stream 格式不兼容，统一用 ReAct 简化维护 |
| 多 Agent 用单 ReAct + 预判 | 同上，4 种模式共享同一套 stream 处理路径 |

---

## 快速启动

### 环境要求

- Docker Engine 20.10+，4GB+ 可用内存
- DeepSeek API Key + 阿里云 DashScope API Key

### 1. 克隆项目

```bash
git clone https://github.com/RockRebuild/GuideExamAgentDemo.git
cd GuideExamAgentDemo
```

### 2. 配置 .env

```bash
# .env 文件填入以下密钥
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
# 可选
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
```

### 3. 准备数据

```bash
# chroma_db/     — 向量库数据（预生成，需自行准备）
# question_bank.json — 题库文件（7000+ 题）
# bge_reranker_cache/ — BGE 模型（首次启动自动从 hf-mirror 下载）
```

### 4. 启动

```bash
docker compose build
docker compose up -d
```

首次启动自动下载 BGE-Reranker 模型并缓存（~30 秒），后续直接秒启。

### 5. 访问

```
http://localhost:8501
```

---

## Docker 镜像优化

| 阶段 | 体积 | 手段 |
|---|---|---|
| 初始 | 12.6GB | PDM install 含 CUDA torch + nvidia/triton 全家桶（23 个 GPU 包） |
| 第一轮 | 7.8GB | 移除 nodejs/npm(-945MB)、去除模型 COPY(-1.1GB)、完善 .dockerignore |
| 最终 | **2.08GB** | torch 从 pyproject.toml 移除 → PDM 不装 CUDA → 单独 pip 装 CPU 版 |

分析每个 Docker layer 的体积，发现 **5.6GB 的 CUDA/nvidia/triton** 在 CPU 推理场景下完全无意义。移除后包数从 180 → 157，build 时间 25min → 7min。ECS 40G 系统盘仅占 5%。

---

## 冒烟测试

```bash
# 5 条典型用例：知识问答、出卷、批改、问候、越狱防护
python tests/test_smoke.py
```

10 个标注用例的完整 Agent 行为评估通过 API 触发：
```bash
curl -X POST http://localhost:8501/api/agent-eval/run
```

---

## 项目结构

```
├── server/
│   ├── main.py                 # FastAPI 入口 + 路由注册
│   ├── core/                   # 领域层
│   │   ├── agent.py            # Agent 创建 + 重试 + 孤儿 tool_calls 修复
│   │   ├── multi_agent.py      # Supervisor 路由预判
│   │   ├── tools.py            # 检索/出卷/批改/天气 工具定义
│   │   ├── retrieval_utils.py  # BGE-Reranker 精排（分批+OOM防护）
│   │   ├── semantic_cache.py   # ChromaDB 语义缓存
│   │   ├── agent_eval.py       # Agent 行为评估框架（10 标注用例）
│   │   ├── llm_service.py      # LLM 调用 + Token 计数 + 成本
│   │   ├── wrong_book.py       # 错题本 Redis CRUD + 自动检测
│   │   └── eval_logger.py      # 评估日志
│   ├── routes/                 # API 路由（9 个模块）
│   ├── services/               # Agent 流式编排 + RAGAS + 题库
│   └── models/                 # Pydantic schemas
├── static/index.html           # Vue3 SPA 前端
├── tests/                      # 测试脚本 + 冒烟测试
├── prompts/
│   ├── system_prompt.md        # Agent 系统提示词
│   └── supervisor_prompt.md    # Supervisor 路由提示词
├── weather_server.py           # MCP 天气服务
├── Dockerfile                  # Agent 镜像 (2.08GB)
├── Dockerfile.mcp              # MCP 天气镜像 (~200MB)
├── docker-compose.yml          # 三服务编排
├── chroma_db/                  # 向量库（需自行准备）
├── question_bank.json          # 题库 7000+ 题
└── bge_reranker_cache/         # BGE 模型缓存
```

---

## ECS 性能数据

| 指标 | 数值 |
|---|---|
| 题库规模 | 7,000+ 题 × 4 科目 |
| 教材片段 | 2,000+ 段落 + 摘要 + 句子 |
| 检索延迟 | < 2s（含 BGE 精排） |
| LLM 首 Token 延迟 | ~2s (DeepSeek API) |
| 语义缓存命中率 | > 30%（稳定运行后） |
| 镜像大小 | agent 2.08GB + MCP 200MB + Redis 520MB |
| ECS 配置 | 2C4G，40G 磁盘 |

---

## License

MIT
