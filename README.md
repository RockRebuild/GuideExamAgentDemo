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

### 📊 阅卷批改
- 按题号或章节批量批改
- 每题判断对错 + 解析，答错自动推荐相关知识点复习

### 🌤 天气查询
- 基于 **MCP 协议**的自定义天气服务，支持当天及未来 3 天预报
- 展示 Agent 工具调用 + 外部服务集成的能力

---

## 系统架构

```
                   Web 前端 (Vue3)  ←─ SSE 流式 ─→  FastAPI 后端
                                                        │
                   ┌────────────────────────────────────┼──────────────────────┐
                   │              LangGraph ReAct Agent                       │
                   │                      │                                   │
                   │     DeepSeek V4 Flash (LLM)                             │
                   │                      │                                   │
                   │        ┌─────────────┴─────────────┐                    │
                   │        │       Tool Calling         │                    │
                   │        └─────────────┬─────────────┘                    │
                   │   ┌──────────────────┼──────────────────┐               │
                   │   ▼                  ▼                   ▼               │
                   │ 5种检索策略     出卷/批改/天气         MCP Server         │
                   └──────────────────┬──────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
    ChromaDB 向量库           BGE-Reranker 精排              Redis
  (DashScope Embedding)       (CPU, sub-batch)       (会话/反馈/错题本)
```

---

## 技术栈

| 分类 | 技术 | 说明 |
|---|---|---|
| Agent 框架 | LangGraph | ReAct Agent + Redis checkpoint 多轮记忆 |
| LLM | DeepSeek V4 Flash | 通过 OpenAI 兼容 API 调用 |
| 向量数据库 | ChromaDB | 阿里云 DashScope text-embedding-v4 |
| 精排模型 | BGE-Reranker-base | CPU 推理，子批量处理防 OOM |
| 关键词检索 | BM25 (rank-bm25 + jieba) | 与语义检索互补 |
| 后端 | FastAPI + SSE | 流式响应，异步非阻塞 |
| 前端 | Vue3 单页应用 | 实时展示工具调用过程 |
| 评估 | RAGAS 四指标 | Faithfulness / Relevance / Precision / Recall |
| 监控 | Langfuse | Token 用量 + 链路追踪 |
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

## 项目结构

```
├── server/
│   ├── main.py              # FastAPI 入口 + 路由注册
│   ├── routes/              # API 路由
│   │   ├── chat.py          # SSE 流式聊天
│   │   ├── wrong_book.py    # 错题本 CRUD
│   │   ├── evaluation.py    # RAGAS 评估
│   │   ├── question_bank.py # 题库浏览
│   │   ├── feedback.py      # 用户反馈
│   │   ├── cost.py          # API 费用
│   │   └── chat_log.py      # 对话日志
│   ├── services/
│   │   ├── agent_service.py # Agent 流式调用 + 孤儿 tool_calls 修复
│   │   └── eval_service.py  # RAGAS 异步评估
│   └── models/schemas.py    # Pydantic 模型
├── static/index.html        # Vue3 前端（错题本/题库/反馈/评估）
├── agent.py                 # Agent 配置 + 重试 + 孤儿检测
├── tools.py                 # 检索/出卷/批改/天气 工具定义
├── retrieval_utils.py       # BGE-Reranker 精排（分批+OOM防护）
├── wrong_book.py            # 错题本 Redis CRUD + LLM 文本自动检测
├── weather_server.py        # MCP 天气服务（当天 + 未来3天预报）
├── prompts/system_prompt.md # Agent 系统提示词
├── Dockerfile               # Agent 镜像 (2.08GB)
├── Dockerfile.mcp            # MCP 天气镜像 (~200MB)
├── docker-compose.yml       # 三服务编排
├── chroma_db/               # 向量库（需自行准备）
├── question_bank.json       # 题库 7000+ 题
└── bge_reranker_cache/      # BGE 模型缓存
```

---

## ECS 性能数据

| 指标 | 数值 |
|---|---|
| 题库规模 | 7,000+ 题 × 4 科目 |
| 教材片段 | 2,000+ 段落 + 摘要 + 句子 |
| 检索延迟 | < 2s（含 BGE 精排） |
| LLM 首 Token 延迟 | ~2s (DeepSeek API) |
| 镜像大小 | agent 2.08GB + MCP 200MB + Redis 520MB |
| ECS 配置 | 2C4G，40G 磁盘 |

---

## License

MIT
