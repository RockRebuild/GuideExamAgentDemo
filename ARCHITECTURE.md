# AI 导游考试 Agent-RAG 智能问答系统 — 架构设计文档

> **项目定位**：基于 LangGraph + ChromaDB + BGE-Reranker 的 Multi-Agent RAG 系统，为导游资格考试提供教材知识检索、智能出卷、阅卷批改、错题本管理等一站式备考服务。

---

## 目录

1. [系统总览](#1-系统总览)
2. [架构设计](#2-架构设计)
3. [核心模块详解](#3-核心模块详解)
   - [3.1 Agent 引擎](#31-agent-引擎)
   - [3.2 检索系统](#32-检索系统)
   - [3.3 BGE-Reranker 精排](#33-bge-reranker-精排)
   - [3.4 工具系统](#34-工具系统)
   - [3.5 MCP 天气服务](#35-mcp-天气服务)
   - [3.6 知识问答模式](#36-知识问答模式)
   - [3.7 智能出卷模式](#37-智能出卷模式)
   - [3.8 阅卷批改模式](#38-阅卷批改模式)
   - [3.9 错题本系统](#39-错题本系统)
   - [3.10 流式通信](#310-流式通信)
   - [3.11 质量评估](#311-质量评估)
   - [3.12 成本监控](#312-成本监控)
4. [数据层设计](#4-数据层设计)
5. [前端架构](#5-前端架构)
6. [部署架构](#6-部署架构)
7. [关键技术决策](#7-关键技术决策)
8. [安全与容错](#8-安全与容错)

---

## 1. 系统总览

### 1.1 项目要解决的问题

导游资格考试备考的四个核心痛点：

| 痛点 | 解决方案 | 对应功能 |
|---|---|---|
| **教材内容庞杂，查找困难** | 5 种检索策略 + BGE 精排，从 2000+ 教材段落中精准定位 | 📖 教材知识问答 |
| **刷题需求大，出卷耗时** | 按科目/章节/题型自动抽题，7000+ 题库实时检索 | 📝 智能出卷 |
| **自主练习无法判断对错** | 自动批改 + 详细解析 + 知识点推荐 | 📊 阅卷批改 |
| **错题分散，复习无针对性** | 错题自动记录，按科目/错误次数统计，支持手动移除 | 📒 错题本 |

### 1.2 全局请求流

```
用户输入（Vue3 SPA）
    │
    ▼
FastAPI SSE 端点 (/api/chat/stream)
    │
    ├─ 输入净化（长度截断、敏感词过滤）
    │
    ▼
agent_service.stream_chat()
    │
    ├─ 孤儿 tool_calls 检测 → 自动切换 thread_id
    ├─ System Prompt 注入
    │
    ▼
LangGraph ReAct Agent（DeepSeek V4 Flash）
    │
    ├─ LLM 推理 → 决定调用哪个工具
    ├─ 工具执行（检索/出卷/批改/天气）
    ├─ 工具结果经 BGE-Reranker 精排
    │
    ▼
SSE 事件流 → 前端实时渲染
    │
    ├─ event: tool   → 工具调用过程（前端可折叠查看）
    ├─ event: agent  → LLM 逐字输出
    ├─ event: done   → 完整回答 + token 统计
    └─ event: error  → 错误信息
```

---

## 2. 架构设计

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                      表示层 (Presentation)                    │
│  static/index.html — Vue3 SPA, SSE streaming, dark theme    │
├─────────────────────────────────────────────────────────────┤
│                      接入层 (Gateway)                        │
│  server/main.py — FastAPI, lifespan, CORS, static files     │
│  server/routes/ — RESTful + SSE endpoints                   │
├─────────────────────────────────────────────────────────────┤
│                      服务层 (Service)                        │
│  server/services/agent_service.py — 流式编排, 错题检测       │
│  server/services/eval_service.py — RAGAS 异步评估            │
│  server/services/qb_service.py — 题库查询/分页               │
├─────────────────────────────────────────────────────────────┤
│                      领域层 (Domain)                         │
│  server/core/agent.py — LangGraph ReAct Agent, 动态工具加载   │
│  server/core/tools.py — 5 种检索 + 出卷 + 批改工具定义        │
│  server/core/retrieval_utils.py — BGE-Reranker 精排引擎      │
│  server/core/wrong_book.py — 错题本业务逻辑                   │
│  server/core/llm_service.py — LLM 调用封装 + 成本核算         │
├─────────────────────────────────────────────────────────────┤
│                      基础设施层 (Infrastructure)              │
│  ChromaDB — 4 个向量集合（段落/摘要/句子/父切片）              │
│  Redis — checkpoint 记忆 / 反馈 / 费用 / 错题本               │
│  DashScope Embedding — text-embedding-v4                     │
│  BM25 + jieba — 关键词检索                                   │
│  BGE-Reranker-base — 本地 CPU 精排                           │
│  DeepSeek V4 Flash — LLM 推理（OpenAI 兼容 API）             │
│  DeepSeek V4 Pro — RAGAS Judge 模型                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 三服务 Docker 编排

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Redis      │    │  guide-agent     │    │ weather-mcp  │
│  (6379)      │◄──►│  (8080→8501)     │◄──►│  (8000)      │
│              │    │                  │    │              │
│ Redis Stack  │    │ Python 3.11      │    │ Python 3.11  │
│  checkpoint  │    │ LangGraph Agent  │    │ FastAPI      │
│  错题本      │    │ ChromaDB 客户端   │    │ wttr.in API  │
│  反馈/费用   │    │ BGE-Reranker     │    │ MCP 协议     │
│              │    │ 2.08GB 镜像      │    │ ~200MB 镜像  │
└──────────────┘    └──────────────────┘    └──────────────┘
        ▲                    ▲                      ▲
        └────────────────────┴──────────────────────┘
                        guide-net (bridge)
```

---

## 3. 核心模块详解

### 3.1 Agent 引擎

**文件**：`server/core/agent.py`

**解决的问题**：如何让 LLM 拥有工具调用能力和多轮对话记忆，同时在不同功能模式间灵活切换。

**设计思路**：

```
                        ┌─────────────────────────┐
                        │   LangGraph ReAct Agent  │
                        │                          │
   System Prompt ──────►│  LLM (DeepSeek V4 Flash) │
   (Markdown 文件)      │  temperature = 0         │
                        │  streaming = True        │
                        │  thinking = disabled     │
                        │                          │
                        │  ┌────────────────────┐  │
                        │  │  RedisSaver         │  │
                        │  │  (checkpoint 持久化) │  │
                        │  └────────────────────┘  │
                        └──────────┬───────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   动态工具加载（按模式）       │
                    │                             │
                    │ 📖 知识问答: 5 种检索策略     │
                    │ 📝 智能出卷: search_questions │
                    │ 📊 阅卷批改: search + grade   │
                    │ 🌤 所有模式: get_weather      │
                    └─────────────────────────────┘
```

**关键设计点**：

1. **按模式动态创建 Agent**：不同模式加载不同工具集，避免工具列表膨胀（LLM 选择工具的准确率随工具数量增加而下降）。Agent 实例缓存复用，避免重复创建。

2. **Redis Checkpoint 多轮记忆**：通过 `RedisSaver` 持久化 LangGraph 的 checkpoint，实现跨请求的对话状态保存。每个模式独立 `thread_id`，三个模式的对话互不干扰。

3. **孤儿 tool_calls 检测与修复**：当用户刷新页面或连接中断时，checkpoint 中可能残留未完成的工具调用（AIMessage 有 tool_calls 但没有对应的 ToolMessage）。下次请求会触发 LangGraph 状态校验失败。解决方案：检测到孤儿状态后自动创建新 `thread_id`（`{old_id}_{timestamp}`），绕过损坏的 checkpoint。

4. **天气 MCP 重试机制**：依次尝试 `weather-mcp`（Docker 容器名）→ `localhost` → `127.0.0.1`。加载失败时从 System Prompt 中移除天气相关描述，避免 LLM 尝试调用一个不存在的工具。

5. **重试策略**：使用 `tenacity` 库，对 `RateLimitError`、`APIConnectionError`、`InternalServerError`、`APITimeoutError` 进行指数退避重试（最多 3 次）。

---

### 3.2 检索系统

**文件**：`server/core/tools.py`

**解决的问题**：用户口语化、模糊的提问如何精准命中教材中的相关内容？不同信息需求（事实查询 vs. 概念对比 vs. 章节概览）需要不同的检索策略。

#### 3.2.1 向量库设计

```
ChromaDB 持久化目录: chroma_db/

┌──────────────────┬─────────────────┬──────────────────────┐
│ 集合名称          │ 切片粒度         │ 用途                  │
├──────────────────┼─────────────────┼──────────────────────┤
│ guide_child      │ 200 字符段落     │ 语义/混合检索的基础库  │
│ guide_summary    │ 章节摘要         │ 概览/目录级查询        │
│ guide_sentence   │ 句子级           │ multi_search 三层之一  │
│ guide_parent     │ 800 字符父切片   │ parent_child 的大块返回 │
└──────────────────┴─────────────────┴──────────────────────┘
```

所有集合使用阿里云 DashScope `text-embedding-v4` 生成向量嵌入。

#### 3.2.2 五种检索策略对比

| 策略 | 工具名 | 适用场景 | 检索方式 | 为什么这样设计 |
|---|---|---|---|---|
| **语义搜索** | `search_textbook` | 一般事实查询 | ChromaDB 余弦相似度 + 分数过滤(>0.5) | 最基础的向量检索，适合语义相近的查询 |
| **混合检索** | `hybrid_search` | 专有名词、法律条文 | 语义(BGE向量) + BM25(jieba分词) 结果合并去重 | 专有名词在语义向量中容易被稀释，BM25 精确匹配能弥补 |
| **多粒度并行** | `multi_search` | 章节概览、宽泛了解 | 三层并行（摘要+段落+句子）→ 合并去重 → 精排 | 不同粒度的切片提供不同视角的信息密度 |
| **父子切片** | `parent_child_search` | 需要完整上下文 | 子切片匹配 → 取 parent_id → 返回大块父切片 | 解决"匹配准但上下文断裂"的问题：用细粒度索引找位置，用粗粒度返回完整信息 |
| **改写检索** | `rewritten_search` | 模糊口语化问题 | LLM 生成 3 个改写变体 + 原问题 → 并行检索 → 合并去重 | 用户说"带团要注意啥"，改写为"导游带团注意事项"+"地陪服务规范"+"全陪导游职责"三个方向 |

#### 3.2.3 BM25 关键词索引

**为什么需要 BM25**：纯向量语义检索在匹配专业术语、法律编号（如"第35条"）时效果差。BM25 基于词频-逆文档频率，对精确词汇匹配有天然优势。

**实现**：
- 使用 `guide_parent`（800 字符父切片）构建索引，信息量更大
- jieba 中文分词 → rank-bm25 打分
- 语义和关键词各取 top-k，合并后按文本相似度去重（`SequenceMatcher > 0.8`）

#### 3.2.4 统一精排入口

所有 5 种检索策略的结果在返回前都经过 **`retrieve_with_rerank()`** 统一处理：

```python
# server/core/tools.py:48
def retrieve_with_rerank(query: str, raw_contexts: List[str]) -> str:
    top_k = determine_top_k(query)                    # 动态决定保留几条
    refined = refine_and_rerank(raw_contexts, query, top_k=top_k)  # BGE 精排
    return CONTEXT_SEPARATOR.join(refined)            # 用分隔符拼接
```

---

### 3.3 BGE-Reranker 精排

**文件**：`server/core/retrieval_utils.py`

**解决的问题**：向量检索返回的 top-K 段落中可能包含大量与问题无关的噪音。直接喂给 LLM 会：
1. 浪费 token（成本）
2. 引入噪音导致幻觉（质量）
3. 挤占有价值的上下文（召回）

**精排策略 — 拆句打分 + 段落聚合**：

```
原始段落（去重后 N 段）
    │
    ▼
拆句（RecursiveCharacterTextSplitter，按中文标点拆分）
    │
    ├─ 列举类问题：保留所有句子
    ├─ 非列举类：预筛（含关键词 or 长度 > 50 字）
    │
    ▼
BGE-Reranker 逐句打分
    │
    ▼
段落分 = max(该段落内所有句子的分数)
    │
    ▼
过滤：段落分 < min_score(0.5) → 丢弃
    │
    ▼
排序 + 相似度去重 → Top-K 完整段落
```

**为什么是"拆句打分 + 段落聚合"而不是直接对段落打分**：
- BGE-Reranker 的输入长度限制 512 tokens，长段落会被截断
- 一个段落中可能 80% 与问题无关，只有 1-2 句关键信息。直接打分会拉低整体分数
- 拆句后每句独立打分，取最高分作为段落分，不会因"稀释效应"丢失有价值段落

**OOM 多层防护**：

```
第 1 层：启动前检查
    ├─ 持久化标记文件 /tmp/bge_reranker_disabled（上次进程被 OOM kill 后留下）
    └─ 可用内存检查（< 1.5GB 不加载模型）

第 2 层：推理时防护
    ├─ OMP_NUM_THREADS=1, MKL_NUM_THREADS=1（限制 CPU 并行度）
    ├─ sub_batch=4（每次只处理 4 句话，减少峰值内存）
    └─ 每批处理完主动 del + gc.collect() 释放张量

第 3 层：运行时降级
    └─ OOM 异常 → 标记禁用 → 降级为返回去重后的原始上下文
       知识问答功能不受影响（只是少了一道精排优化）
```

**动态 Top-K**：`determine_top_k()` 根据问题类型动态调整保留段落数：

| 问题类型 | Top-K | 原因 |
|---|---|---|
| 列举/枚举 | 15 | 需要覆盖多个独立知识点 |
| 对比/比较 | 12 | 需要多方信息 |
| 流程/步骤 | 10 | 需要完整步骤链 |
| 概述/介绍 | 12 | 需要多维度信息 |
| 短事实查询 | 3 | 精确匹配，少即是多 |
| 默认 | 5 | 平衡覆盖与成本 |

---

### 3.4 工具系统

**文件**：`server/core/tools.py`

**解决的问题**：如何让 LLM 以结构化方式与外部系统交互。

**工具列表**：

```
search_textbook      → ChromaDB 语义搜索 → BGE 精排
hybrid_search        → 语义 + BM25 混合 → BGE 精排
multi_search         → 三层并行 → 去重 → BGE 精排
parent_child_search  → 子切片匹配 → 父切片返回 → BGE 精排
rewritten_search     → LLM 改写 → 并行检索 → 去重 → BGE 精排
search_questions     → 题库 JSON 按章节/题型过滤
grade_answer         → 比对答案 + 返回解析 + 错题自动记录
get_weather          → MCP 协议 → wttr.in API
```

**MCP 工具动态加载**：

```python
# server/core/tools.py:406
def _json_schema_to_pydantic(schema: dict, name: str):
    """将 JSON Schema 转为 Pydantic BaseModel"""
    # 遍历 properties，映射 type → Python type
    # 必填字段 → ...，可选字段 → None
    # 返回 pydantic.create_model() 动态类
```

这样 LLM 能完整看到 MCP 工具的参数定义（通过 LangChain 的 `args_schema`），准确生成工具调用参数。

**事件循环兼容**：`load_mcp_tools_http()` 同时支持两种场景：
- 不在 asyncio 上下文中（脚本/Streamlit）→ `asyncio.run()`
- 在 asyncio 上下文中（FastAPI handler）→ `ThreadPoolExecutor` 独立线程运行 `asyncio.run()`

---

### 3.5 MCP 天气服务

**文件**：`weather_server.py` + `Dockerfile.mcp`

**解决的问题**：
1. 展示 Agent 调用外部服务的能力（工具调用的完整链路）
2. 体现 MCP（Model Context Protocol）协议的实际应用

**协议实现**：

```
Agent (LangGraph)                    weather-mcp (FastAPI:8000)
    │                                       │
    ├── POST /mcp {"method":"tools/list"} ──► 返回工具定义（name, inputSchema）
    │                                       │
    ├── POST /mcp {"method":"tools/call",  ──► GET wttr.in/{city}?format=...
    │    "params":{"name":"get_weather",         │
    │    "arguments":{"city":"杭州","days":3}}}   ├─ days=1: 简洁格式 "%C %t"
    │                                           └─ days>1: JSON 格式，解析最高/最低温
    │◄── {"content":[{"type":"text",           + 天气描述
    │     "text":"杭州 当前天气：晴 +25°C"}]}
```

**Dockerfile.mcp 镜像优化**：仅 4 个依赖（fastapi, uvicorn, httpx, starlette），镜像约 200MB，远小于 agent 镜像的 2.08GB。用 slim Python 基础镜像 + 阿里云 pip 源。

---

### 3.6 知识问答模式

**模式标识**：`📖 教材知识问答`

**加载工具**：5 种检索策略 + 天气

**工作流程**：

```
用户: "《旅游法》第35条是什么？"
    │
    ▼
LLM 判断工具选择优先级（System Prompt 规则 3）:
    "精确匹配专有名词/法律条文 → hybrid_search"
    │
    ▼
hybrid_search("旅游法 第35条")
    ├─ 语义搜索: ChromaDB 余弦相似度
    ├─ BM25关键词: jieba 分词 + BM25Okapi 打分
    ├─ 合并去重: SequenceMatcher > 0.8
    └─ BGE-Reranker 精排 → Top-K 段落
    │
    ▼
LLM 基于检索结果生成回答:
    "根据《政策与法律法规统编教材》第X章：
     '旅行社不得以不合理的低价组织旅游活动……'
     [来源：第X章 旅游法，第XX页]"
```

**System Prompt 的核心约束**：
- **零外部知识原则**（规则 0）：所有回答必须且只能来自工具返回的教材原文，禁止添加训练知识
- **逐句有出处**（规则 1）：每条事实性陈述必须在教材原文中可找到
- **推理透明**（规则 9）：跨知识点推理时必须标注"以下为分析"并列出依据原文

---

### 3.7 智能出卷模式

**模式标识**：`📝 智能出卷`

**加载工具**：`search_questions`

**工作流程**：

```
用户: "导游业务 团队导游服务规范 出3道单选题"
    │
    ▼
LLM 调用 search_questions(chapter="团队导游服务规范", qtype="单选", count=3)
    │
    ▼
题库过滤: 7000+ 题 → 按 chapter + type 过滤 → 取前 N 道
    │
    ▼
返回格式化题目:
    ID:单选_1 1. 题目内容... A.xxx B.xxx C.xxx D.xxx
    ID:单选_2 2. ...
    ID:单选_3 3. ...
    📝 请按格式回答："第X题，我的答案是 X"
```

**防编造机制**（System Prompt 规则 2）：
- search_questions 返回结果标注"共 X 道"
- 如果 X < 用户要求数量 → 如实告知"该章节该题型只有 X 道"，绝不编造
- 禁止从其他题型"改编"题目凑数
- 编造的题目 ID 不存在于题库，学员无法用 grade_answer 批改

**答题后自动批改**：
```
用户: "第一题，我的答案是 B"
    │
    ▼
LLM 调用 grade_answer(index=1, qtype="单选", chapter="团队导游服务规范", student_answer="B")
    │
    ├─ ✅ 正确 → 显示解析
    └─ ❌ 错误 → 显示正确答案 + 解析 + 知识点推荐 + 自动记入错题本
```

---

### 3.8 阅卷批改模式

**模式标识**：`📊 阅卷批改`

**加载工具**：`search_textbook` + `grade_answer`

**工作流程**：

```
用户: "请批改题目 科目一 第一章 单选 第一题，我的答案是 B"
    │
    ▼
LLM 调用 grade_answer(subject="科目一", chapter="第一章", qtype="单选", index=1, student_answer="B")
    │
    ▼
题库匹配: 按 subject + chapter + type 过滤 → 取 index 位置题目 → 比对学生答案
    │
    ▼
返回批改结果:
    ✅ 回答正确！/ ❌ 回答错误。
    题目: ...
    正确答案: ...
    解析: ...
    💡 建议：复习「XXX」的相关知识点
```

**与智能出卷的区别**：批改模式下多加载了 `search_textbook` 工具，允许学员在查看批改结果后直接追问教材知识点。

---

### 3.9 错题本系统

**文件**：`server/core/wrong_book.py` + `server/routes/wrong_book.py`

**解决的问题**：学员练习中答错的题目分散在多轮对话中，难以统一管理和针对性复习。

**数据模型**（Redis Hash：`wrongbook:items`）：

```json
{
  "question_id": "单选_123",
  "question": "题目原文",
  "user_answer": "B",
  "correct_answer": "A",
  "subject": "导游业务",
  "chapter": "团队导游服务规范",
  "type": "单选",
  "explanation": "解析内容",
  "wrong_count": 3,
  "first_wrong_at": "2026-07-01T10:00:00",
  "last_wrong_at": "2026-07-13T15:30:00"
}
```

**双重自动记录机制**：

```
路径 1: 工具调用检测（精确）
    grade_answer 返回 "❌ 回答错误"
        → detect_and_record() 解析工具返回文本
        → 正则提取题目/答案/解析
        → 查题库补全 subject/chapter
        → 写入 Redis

路径 2: LLM 文本检测（兜底）
    LLM 在生成文本中直接批改（未调用 grade_answer 工具）
        → detect_from_agent_text() 解析 "❌ 回答错误" 后的内容
        → 正则提取 + 题库模糊匹配
        → 写入 Redis
```

**为什么需要路径 2（文本检测兜底）**：智能出卷模式下，LLM 有时会在同一轮对话中出卷+批改，此时它可能在文本回复中直接判断对错而不调用 grade_answer 工具。路径 2 通过解析 LLM 的文本回复来捕获这种情况。

**错题去重**：同一 question_id 再次错 → `wrong_count += 1`，更新 `last_wrong_at` 和 `user_answer`。按错误次数降序排列，高频错题优先展示。

**API 设计**：
- `GET /api/wrong-book/list?subject=全部` → 按科目筛选
- `GET /api/wrong-book/stats` → 按科目/章节分布统计
- `DELETE /api/wrong-book/item/{id}` → 手动移除（已掌握）
- `DELETE /api/wrong-book/clear` → 清空

---

### 3.10 流式通信

**文件**：`server/services/agent_service.py`

**解决的问题**：LangGraph Agent 的 stream 是同步阻塞的，但 FastAPI 是异步的。需要桥接两者，同时保持 SSE 事件流的实时性。

**桥接方案 — asyncio.Queue + 线程池**：

```
FastAPI async handler (事件循环线程)
    │
    ├─ 创建 asyncio.Queue
    ├─ 提交 _run_sync() 到 ThreadPoolExecutor
    │
    ▼
_run_sync() 在独立线程中运行
    │
    ├─ stream_agent_with_retry(agent, messages, config)
    ├─ 每个 chunk → loop.call_soon_threadsafe(queue.put_nowait, chunk)
    └─ 结束 → queue.put_nowait(None)  # 哨兵
    │
    ▼
FastAPI handler 异步消费队列
    │
    ├─ chunk = await queue.get()
    ├─ "tools" in chunk → yield event:tool (工具调用过程)
    ├─ "agent" in chunk → yield event:agent (LLM 逐字输出)
    └─ 哨兵 → yield event:done (最终结果 + token 统计)
```

**SSE 事件类型**：

| 事件 | 触发时机 | 前端处理 |
|---|---|---|
| `event: tool` | 每次工具调用完成 | 追加到 toolRecords，折叠面板显示 |
| `event: agent` | LLM 每次输出 token | 逐字追加到 streamingAnswer |
| `event: done` | 流结束 | 组装完整消息，渲染 Markdown，触发错题检测 |
| `event: error` | 异常 | 显示错误消息 |

**多模式会话隔离**：每个模式独立的 `thread_id`：

```python
THREAD_IDS = {
    "📖 教材知识问答": "guide_exam_knowledge",
    "📝 智能出卷": "guide_exam_testgen",
    "📊 阅卷批改": "guide_exam_grading",
}
```

切换模式时对话历史独立，不会混淆上下文。

---

### 3.11 质量评估

**文件**：`server/services/eval_service.py` + `server/routes/evaluation.py`

**解决的问题**：如何量化评估 RAG 系统的回答质量，持续监测检索和生成环节的效果。

**RAGAS 四维评估**：

| 指标 | 衡量什么 | Judge LLM | 依赖数据 |
|---|---|---|---|
| **Faithfulness（忠实度）** | 回答是否完全基于检索上下文，有无编造 | DeepSeek V4 Pro | question + answer + contexts |
| **Answer Relevancy（答案相关性）** | 回答是否与问题相关 | DeepSeek V4 Pro + embedding | question + answer |
| **Context Precision（上下文精度）** | 检索到的上下文中，相关与无关的比例 | DeepSeek V4 Pro | question + answer + contexts |
| **Context Recall（上下文召回）** | 回答所需信息在上下文中的覆盖率 | DeepSeek V4 Pro | question + answer + contexts |

**异步任务模式**：

```
POST /api/eval/start → 创建 eval task → 返回 task_id
    │
    ▼
asyncio.create_task() 后台运行
    │
    ├─ ThreadPoolExecutor(4) 并行计算 4 项指标
    ├─ 每项间隔 3s 启动（避免同时冲击 API 速率限制）
    │
    ▼
GET /api/eval/status/{task_id} → 前端 2s 轮询
    │
    ├─ status: "running" → 继续轮询
    ├─ status: "done" → scores: {faithfulness: 0.92, ...}
    └─ status: "error" → error: "..."
```

**评估日志**（`eval_log.jsonl`）：每次评估结果追加到 JSONL 文件，包含完整的时间戳、问题、回答、上下文和四项分数。可用于后续构建评估数据集和趋势分析。

---

### 3.12 成本监控

**文件**：`server/core/llm_service.py` + `server/routes/cost.py`

**解决的问题**：API 调用有成本，需要实时了解费用消耗、按模型拆分明细、5 日趋势。

**费用计算模型**：

```python
# DeepSeek 定价（¥ / 百万 tokens）
PRICES = {
    "deepseek-v4-flash": {"input": 1.0, "output": 2.0},   # Agent 主模型
    "deepseek-v4-pro":   {"input": 2.0, "output": 8.0},   # RAGAS Judge
}
# DashScope Embedding: ¥0.7 / 百万 tokens
```

**数据流**：

```
每次 Agent 调用
    │
    ├─ 提取 usage_metadata（input_tokens, output_tokens）
    ├─ 写入 Redis: token_usage:{date}:{model} → {input, output, total}
    └─ 48 小时 TTL 自动清理
    │
    ▼
GET /api/cost/daily
    ├─ 按模型聚合当日用量
    ├─ 计算费用 = input/1M * price.input + output/1M * price.output
    ├─ 5 日趋势（遍历最近 5 天的 Redis key）
    └─ 预算告警（> 70% 高亮提醒）
```

**并发控制**：`LLMService` 使用 `asyncio.Semaphore(3)` 限制同时进行的 LLM 调用，防止突发流量打爆 API 配额。

---

## 4. 数据层设计

### 4.1 ChromaDB — 向量知识库

```
chroma_db/
├── guide_child/       # 段落级（200 字符），语义搜索主库
├── guide_summary/     # 摘要级，章节目录/概览
├── guide_sentence/    # 句子级，细粒度匹配
└── guide_parent/      # 父切片（800 字符），BM25 索引源 + parent_child 返回
```

**数据来源**：导游考试教材 PDF → PyPDFLoader 加载 → RecursiveCharacterTextSplitter 切片 → DashScope text-embedding-v4 向量化。

**元数据**：每个向量块携带 `source`（来源文件）、`chapter`（章节）、`page`（页码），回答时可标注出处。

**初始化工具**：`initialize_vectorstores()` 自动检测空库并导入 PDF。支持增量更新（已有数据不重复导入）。

### 4.2 Redis — 多用途缓存与持久化

```
Redis 数据结构设计：

1. LangGraph Checkpoint（会话记忆）
   Key: 由 LangGraph 内部管理
   用途: 多轮对话状态持久化

2. token_usage:{date}:{model}（Hash）
   Fields: input, output, total
   用途: API 费用按日+模型聚合
   TTL: 48 小时

3. feedback:list（List）
   用途: 用户反馈时间序列
   用途: 好评率统计

4. wrongbook:items（Hash）
   Key: question_id
   Value: JSON 字符串
   用途: 错题本持久化
```

### 4.3 题库 — JSON 文件

```
question_bank.json（7000+ 题）

每条题目结构：
{
  "id": "单选_1234",
  "type": "单选 | 多选 | 判断",
  "subject": "导游业务 | 政策与法律法规 | ...",
  "chapter": "团队导游服务规范",
  "question": "题目正文",
  "options": ["A.xxx", "B.xxx", "C.xxx", "D.xxx"],
  "answer": "A",
  "explanation": "解析内容"
}
```

**为什么用 JSON 而非数据库**：
- 题库数据量 7000+ 题，JSON 文件完全够用（加载到内存约几 MB）
- 无需额外数据库依赖，简化部署
- 查询场景简单（按 subject/chapter/type 过滤），不需要 SQL

---

## 5. 前端架构

**文件**：`static/index.html`（单文件 Vue3 SPA，1368 行）

**设计理念**：模仿 Streamlit 的暗色主题视觉风格，但用纯前端 Vue3 实现，不依赖 Streamlit 服务端渲染。

### 5.1 组件树

```
App
├── Sidebar（左侧栏）
│   ├── ModeSelector（三种模式单选）
│   ├── FeedbackStats（反馈统计面板）
│   ├── CostPanel（费用明细 + 5 日趋势）
│   └── (移动端) Hamburger 菜单 + Overlay
│
├── Main（主内容区）
│   ├── SampleQuestions（4 个示例问题按钮，按模式切换）
│   ├── QuestionBankBrowser（阅卷模式专属，筛选 + 分页）
│   ├── ChatArea
│   │   ├── MessageList
│   │   │   ├── UserMessage（蓝色气泡，右对齐）
│   │   │   ├── AssistantMessage（卡片样式，Markdown 渲染）
│   │   │   │   ├── ToolExpander（折叠面板，展示工具调用过程）
│   │   │   │   ├── FeedbackRow（👍/👎 + 质量评估按钮）
│   │   │   │   └── RagasScores（四项指标卡片）
│   │   │   └── StreamingMessage（闪烁光标 + 工具调用计数）
│   │   └── ChatInput（textarea + 发送按钮）
│   │
│   └── WrongBookButton（错题本入口，智能出卷模式可见）
│
├── EvalModal（评估进度模态框）
└── WrongBookPanel（错题本全屏面板）
    ├── 科目筛选下拉框
    ├── 错题卡片列表（科目/章节/题型标签，对错答案对比）
    └── 清空/关闭按钮
```

### 5.2 核心交互流程

**消息发送**：
```
sendMessage()
    ├─ 用户消息追加到当前模式的 messagesByMode[mode]
    ├─ fetch POST /api/chat/stream (ReadableStream)
    ├─ 逐行解析 SSE 事件（按 \n\n 分割 → event:/data: 拆分）
    ├─ event:agent → streamingAnswer 逐字累积（Markdown 实时渲染）
    ├─ event:tool  → toolRecords 追加（侧边指示器计数递增）
    ├─ event:done  → 组装完整消息对象，追加到 messagesByMode
    └─ event:error → 显示错误消息
```

**多模式消息隔离**：`messagesByMode` 是按模式分组的 `reactive({})` 对象，切换模式时 `currentMessages` 计算属性自动切换到对应模式的消息列表，无需重新加载。

**评分可视化**：前端接收到 RAGAS 四维分数后，用颜色编码展示：
- ≥ 0.9 → 绿色（优秀）
- ≥ 0.7 → 橙色（一般）
- < 0.7 → 红色（差）

### 5.3 移动端适配

通过 CSS `@media` 查询实现三级响应式断点：
- **768px**：侧边栏变为滑入式覆盖层，示例问题单列，题库筛选双列
- **480px**：侧边栏全宽，题库筛选单列，评分卡片改为水平布局
- 侧边栏关闭按钮和汉堡菜单仅在移动端显示

---

## 6. 部署架构

### 6.1 Docker Compose 三服务编排

```yaml
services:
  redis:           # Redis Stack Server（checkpoint + 错题本 + 费用）
  agent:           # 主应用（FastAPI + LangGraph + ChromaDB + BGE-Reranker）
  weather-mcp:     # 独立的 MCP 天气微服务（FastAPI，端口 8000）
```

**网络拓扑**：所有服务通过 `guide-net` bridge 网络通信。`redis` 和 `weather-mcp` 不暴露外部端口（仅 `127.0.0.1:6379`），agent 通过容器名（`redis`、`weather-mcp`）访问另外两个服务。

### 6.2 Docker 镜像优化历程

| 阶段 | 体积 | 优化手段 |
|---|---|---|
| 初始 | 12.6GB | PDM install 包含 CUDA torch + nvidia/triton 全家桶（23 个 GPU 包） |
| 第一轮 | 7.8GB | 移除 nodejs/npm(-945MB)、去除模型 COPY(-1.1GB)、完善 .dockerignore |
| 最终 | **2.08GB** | torch 从 pyproject.toml 移除 → PDM 不装 CUDA → 单独 pip 装 CPU 版 |

**关键发现**：5.6GB 的 CUDA/nvidia/triton 在 CPU 推理场景下完全无意义。通过分层安装策略（PDM 不装 torch，事后 pip --target 到 PDM 的包目录），实现 6 倍体积缩减。

### 6.3 ECS 性能指标

| 指标 | 数值 |
|---|---|
| ECS 配置 | 2C4G，40G SSD |
| 题库规模 | 7,000+ 题 × 4 科目 |
| 教材片段 | 2,000+ 段落 + 摘要 + 句子 |
| 检索延迟 | < 2s（含 BGE 精排） |
| LLM 首 Token 延迟 | ~2s (DeepSeek API) |
| 镜像总大小 | agent 2.08GB + MCP 200MB + Redis 520MB ≈ 2.8GB |

### 6.4 启动流程

```
entrypoint.sh 启动
    ├─ 检查 BGE-Reranker 模型是否存在
    │   ├─ 不存在 → huggingface_hub.snapshot_download() 从 hf-mirror 下载
    │   └─ 存在 → 跳过
    └─ exec uvicorn server.main:app --host 0.0.0.0 --port 8080
        ├─ 编码设置（locale + PYTHONIOENCODING）
        ├─ Langfuse 初始化（可选）
        ├─ lifespan startup:
        │   ├─ Redis 连接（依次尝试 redis → localhost → 127.0.0.1）
        │   └─ 预热 BM25 索引 + DashScope Embeddings
        ├─ 注册 7 个路由模块
        └─ 挂载静态文件
```

---

## 7. 关键技术决策

### 7.1 为什么是 LangGraph 而不是 LangChain Agent？

| 对比维度 | LangChain AgentExecutor | LangGraph ReAct Agent |
|---|---|---|
| 状态管理 | 简单 in-memory | Checkpoint 持久化（Redis） |
| 多轮对话 | 需手动管理 | 自动恢复上次状态 |
| 错误恢复 | Agent 崩溃则上下文丢失 | 孤儿 tool_calls 可检测并修复 |
| 自定义流程 | 固定的 ReAct 循环 | 可自定义图结构 |
| 工具选择 | 一次性加载所有工具 | 支持按模式动态切换 |

LangGraph 的 checkpoint 机制是实现多轮会话持久化和孤儿修复的关键。

### 7.2 为什么是 5 种检索策略而不是 1 种？

单一语义搜索存在明显的盲区：
- **精确匹配失效**："第35条" 在语义向量中和 "第34条" 高度相似，但用户要的是精确的法律条文编号 → 需要 BM25
- **上下文断裂**：200 字的段落片段无法覆盖"地陪导游服务全流程"这种跨段落知识 → 需要父子切片
- **口语化查询不匹配**："带团要注意啥" 和教材中的"导游服务规范"语义距离较远 → 需要 LLM 改写
- **章节概览需求**：混合了不同信息密度的查询 → 需要多粒度并行

5 种策略覆盖了从"精确查找"到"广泛浏览"的信息需求光谱。

### 7.3 为什么 BGE-Reranker 用 CPU 而不是 GPU？

1. **ECS 实例无 GPU**：2C4G 配置的 ECS 没有 GPU 资源
2. **BGE-Reranker-base 对 CPU 友好**：模型约 1.1GB，CPU 推理单句打分在毫秒级
3. **子批量处理**：sub_batch=4 将内存峰值控制在可接受范围
4. **降级策略**：即使 Reranker OOM，知识问答功能不受影响（返回原始上下文）

### 7.4 为什么三种模式独立 thread_id？

- 知识问答、出卷、批改是独立的对话场景
- 混用同一个 thread_id 会导致：
  - 出卷模式的 System Prompt 和历史消息污染知识问答
  - 不同模式的工具列表不同，checkpoint 中的 tool_calls 可能引用不存在的工具
- 独立 thread_id 隔离三种模式，切换时自动切换对话历史

---

## 8. 安全与容错

### 8.1 输入安全

```python
# server/routes/chat.py
MAX_INPUT_LENGTH = 500  # 长度截断
BLOCKED_KEYWORDS = [    # 敏感词过滤
    "system", "忽略", "ignore", "忘记", "重新开始", "越狱",
    "你是一个", "你的prompt", "你的system", "把你的指令给我"
]
```

### 8.2 容错设计

| 故障场景 | 处理策略 |
|---|---|
| **Redis 不可用** | 反馈/费用/错题本返回空数据，知识问答功能不受影响 |
| **天气 MCP 不可用** | 从 System Prompt 移除天气描述，不影响其他功能 |
| **BGE-Reranker OOM** | 标记禁用 → 降级为返回去重后的原始上下文，不影响检索 |
| **LLM API 限流** | tenacity 指数退避重试（1s → 2s → 4s，最多 3 次） |
| **孤儿 tool_calls** | 自动检测 + 创建新 thread_id 绕过损坏 checkpoint |
| **题库文件缺失** | 返回友好错误信息，不抛出 500 |
| **输入过长** | 自动截断至 500 字符 |
| **prompt 注入** | 敏感词检测拒绝 |

### 8.3 数据安全

- `.env` 含 API Key 不进镜像（docker-compose env_file 从宿主机注入）
- 题库查询 API 不返回答案字段（前端可见题目和选项，不可见答案）
- Dockerfile 中 `rm -rf /app/.env` + `.dockerignore` 双重保护

---

## 附录：项目文件清单

```
.
├── server/
│   ├── main.py                   # FastAPI 入口：lifespan、路由、静态文件
│   ├── state.py                  # ContextVar 请求级状态
│   ├── core/
│   │   ├── agent.py              # LangGraph ReAct Agent 配置
│   │   ├── tools.py              # 7 个工具定义 + MCP 加载器
│   │   ├── retrieval_utils.py    # BGE-Reranker 精排引擎
│   │   ├── llm_service.py        # LLM 调用封装 + 成本核算
│   │   ├── wrong_book.py         # 错题本 Redis CRUD + LLM 文本检测
│   │   └── eval_logger.py        # JSONL 评估日志
│   ├── routes/
│   │   ├── chat.py               # SSE 流式聊天
│   │   ├── wrong_book.py         # 错题本 API
│   │   ├── evaluation.py         # RAGAS 评估 API
│   │   ├── question_bank.py      # 题库查询 API
│   │   ├── feedback.py           # 用户反馈 API
│   │   ├── cost.py               # 费用查询 API
│   │   └── chat_log.py           # 对话日志 API
│   ├── services/
│   │   ├── agent_service.py      # 流式编排 + 错题检测
│   │   ├── eval_service.py       # RAGAS 异步评估
│   │   └── qb_service.py         # 题库过滤分页
│   └── models/
│       └── schemas.py            # Pydantic 数据模型
├── static/
│   └── index.html                # Vue3 SPA 前端（1368 行）
├── prompts/
│   └── system_prompt.md          # Agent System Prompt
├── weather_server.py             # MCP 天气服务
├── entrypoint.sh                 # 容器启动脚本
├── Dockerfile                    # Agent 镜像（2.08GB）
├── Dockerfile.mcp                # MCP 天气镜像（~200MB）
├── docker-compose.yml            # 三服务编排
├── deploy-setup.sh               # ECS 部署前置检查
├── pyproject.toml                # PDM 依赖管理
├── question_bank.json            # 题库（7000+ 题）
├── chroma_db/                    # 向量库（4 个集合）
├── bge_reranker_cache/           # BGE 模型缓存
└── README.md                     # 项目说明
```
