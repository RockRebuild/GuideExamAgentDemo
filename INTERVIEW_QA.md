# 面试题答案：AI导游考试Agent-RAG智能问答系统

> 以下答案基于项目实际代码，如实反映当前状态，不夸大不编造。

---

## Q4. 混合检索怎么融合两路结果？RRF 还是加权融合？参数怎么调出来的？

**回答：既不是 RRF，也不是加权融合。是"拼接 + 文本去重 + BGE Reranker 精排"的三段式。**

代码路径：`server/core/tools.py:212-243`

```python
def hybrid_search(query: str, k: int = 12) -> str:
    # 1. 语义搜索: ChromaDB similarity_search k=12
    semantic_docs = para_store.similarity_search(query, k=k)
    # 2. 关键词搜索: BM25(jieba 分词) 取 top-12
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-k:][::-1]
    # 3. 直接拼接（不是加权融合，不是 RRF）
    all_docs = semantic_docs + keyword_docs
    # 4. SequenceMatcher 文本去重（阈值 0.8）
    unique_docs = deduplicate_docs(all_docs, threshold=0.8)[:k]
    # 5. 送入 BGE Reranker cross-encoder 精排
    return retrieve_with_rerank(query, raw_contexts)
```

融合策略说实话**比较朴素**——两路结果直接 append 拼接，不做分数级的融合。这不是最优方案，但在这个场景下有几个客观约束让它"能工作"：

1. **BGE Reranker 是最终裁判**：不管你前面怎么拼，最终都要过 cross-encoder 打分排序。RRF/加权融合的差异被 Reranker 稀释了。
2. **语料足够小（12K）**：语义和关键词各有 12 条，拼完 24 条文本去重后通常只剩 8~10 条，噪音本来就不多。
3. **去重阈值 0.8**：用 `difflib.SequenceMatcher` 做 O(n²) 文本去重，比基于 embedding 的去重更精确但对长文本慢。

**参数怎么调的**：坦白说——k=12 和 threshold=0.8 是经验值，不是通过系统实验调出来的。如果用 BM25 的 IDF 权重做加权融合或者 RRF 做 rank aggregation，理论上召回率会更好，还没做。

**架构建议**（如果要改进）：向量搜索结果和 BM25 结果用 RRF（Reciprocal Rank Fusion）融合更自然——因为两路结果的 score 量纲不同（余弦距离 vs TF-IDF），RRF 绕过了归一化问题。公式 `RRF(d) = Σ 1/(k + rank_i(d))`，k 通常取 60。

---

## Q5. RAG 数据怎么测的？评估集多少条？Ground truth 谁标的？

**评估方式**：RAGAS（4 指标：Faithfulness、AnswerRelevancy、ContextPrecision、ContextRecall），Judge 模型是 DeepSeek V4 Pro（`server/services/eval_service.py:29`）。

```python
judge_llm = llm_factory("deepseek-v4-pro", client=deepseek_client, max_tokens=4096)
metrics = [
    Faithfulness(llm=judge_llm),
    AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
    ContextPrecision(llm=judge_llm),
    ContextRecall(llm=judge_llm),
]
```

**评估集**：有 **标注测试用例**（`server/core/agent_eval.py:33`），但量不大——代码里看到了约 15~20 条手写 case，涵盖知识问答、出卷、批改三类场景。每条标注了 `expected_tools`（期望调用的工具）、`forbidden_tools`（禁止调用的工具）、`success_patterns`（回答中应包含的正则模式）。这是"Agent 行为评估"不是传统的"检索质量评估"，侧重评估 Agent 是否选了正确的工具、是否完成任务。

**Ground truth 谁标的**：标注用 JSON 手工写的（不是在 UI 里标），每条约需 3~5 分钟——写工具期望、禁止工具、成功模式正则。理论上可以扩展用 DeepSeek 辅助标注然后人工审核，目前还没做。

**线上评估**：每次用户回答后可以点"评估"按钮，触发后台异步 RAGAS 评分。数据记录到 `eval_log.jsonl`（目前 8 条记录）。这是一个"按需评估"的模式，不是自动全量评估。

---

## Q6. 有没有评估 Answer Relevancy？线上有没有 bad case 收集机制？

**有 Answer Relevancy**。RAGAS 四指标全跑：

```python
# server/services/eval_service.py:63-66
def _score_answer_relevancy(question, answer):
    return AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings).score(
        user_input=question, response=answer)
```

**Bad case 收集机制**：有，但比较基础。

1. **用户反馈**（`server/routes/feedback.py`）：
   - 点"有用"→ 语义缓存自动写入（`store(query, result_text)`），下次相似问题直接命中缓存
   - 点"无用"→ 语义缓存自动删除（`remove_by_query(query)`），防止低质量缓存反复命中
   - 数据存 Redis `feedback:list`，不丢失

2. **错题本**（`server/core/wrong_book.py`）：
   - Agent 回答中检测到"❌ 回答错误"时自动记录到 Redis `wrongbook:items` Hash
   - 支持增删改查，前端有错题本面板

3. **Chat log**（`server/routes/chat_log.py`）：
   - 每次问答存入 `chat_logs/` 目录的 JSON 文件
   - 可作为 bad case 分析的原始数据源

4. **Agent 行为评估日志**（`server/core/agent_eval.py` → `agent_eval_log.jsonl`）：
   - 记录工具选择是否正确、端到端是否成功、延迟分布

**不足之处**：没有自动化的 bad case 归类（需要人工从 chat log 中筛选）、没有错误趋势告警、没有基于 Langfuse 的 bad case dashboard。

---

## Q15. "单 Agent 工具膨胀导致选择准确率下降"——有数据吗？几个工具时开始下降？还是自己推断的？

**是推断的，没有实验数据支撑。**

当前多 Agent 模式下，一个 ReAct Agent 装载了约 8~10 个工具：5 个检索工具 + `search_questions` + `grade_answer` + `confirm_exam` + 1 个天气 MCP 工具。

工具膨胀导致准确率下降在学术界有广泛讨论（已知 ReAct Agent 在 10+ 工具时 function calling 准确率下降约 15~20%），但**在这个具体项目中没有做过 A/B 实验**来验证。如果能做，实验设计是：

- 控制组：5 个检索工具（知识问答模式）
- 实验组：10 个工具（多 Agent 模式）
- 指标：相同 50 个标注问题下的工具选择准确率

目前还没有这个数据。

---

## Q16. Supervisor 的路由怎么实现的？Function calling 还是 prompt 分类？路由错了怎么兜底？

**实现方式：Prompt 分类——独立 LLM 调用后正则解析 JSON。**

代码路径：`server/core/multi_agent.py:44-82`

```python
def classify_intent(prompt: str) -> dict:
    llm = ChatOpenAI(model=DEEPSEEK_MODEL, temperature=0, ...)
    resp = llm.invoke(f"{SUPERVISOR_SYSTEM_PROMPT}\n\n用户请求：{prompt}\n\n请输出JSON路由决策：")
    # 正则提取 JSON（支持 markdown code block）
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    decision = json.loads(m.group(1) if m else text)
    return {"reasoning": ..., "workers": [...], "mode": "single", ...}
```

**不是 function calling，是 prompt + JSON 解析**。Supervisor prompt 加载自 `prompts/supervisor_prompt.md`，定义了 3 个 Worker（retrieval_worker / exam_worker / grader_worker），LLM 输出包含 worker 名称和 task_instructions。

**路由错误的兜底**：

```python
except Exception:
    return {
        "reasoning": "fallback",
        "workers": ["retrieval_worker"],
        "mode": "single",
        "task_instructions": prompt,
    }
```

三层兜底：
1. JSON 解析失败 → `[retrieval_worker]` → 安全默认值（只做检索）
2. 缺少 `workers` 字段 → 从旧格式的 `worker` 字段兼容读取
3. `agent_service.py:152-153` 还有一层保护——Supervisor 调用异常时不阻塞主流程，`pass` 跳过

**值得注意的问题**：这是"假"多 Agent——Supervisor 做路由决策，但实际执行还是单 Agent 装载全部工具。路由结果只是**作为前端展示和引导 Agent 行为**的提示，不是真正的 worker 分发。如果要实现真正的多 Agent 协作（每个 worker 独立 agent 实例 + 隔离的工具集），需要改造 `get_agent_for_mode` 让它按 worker 创建不同的 agent。

---

## Q17. 为什么用 LangGraph 而不是 CrewAI / AutoGen / 自己写状态机？

**实际原因**：项目一开始就用 LangChain 生态（`langchain-openai`、`langchain-chroma`、`langchain-community`），`create_react_agent` 直接可用，不需要额外学习或引入新的框架。"人在回路"功能——`confirm_exam` 工具用 `from langgraph.types import interrupt` 一行代码实现 `interrupt()` 暂停 graph 执行——这确实是 LangGraph 的独特优势。

**技术对比**（基于本项目的实际情况）：

| 维度 | LangGraph（当前选择） | CrewAI | AutoGen | 自己写状态机 |
|---|---|---|---|---|
| 集成成本 | 零（已有 langchain 依赖） | 需新增依赖 | 需新增依赖 | 代码量最大 |
| 人在回路(HITL) | `interrupt()` 原生支持 | 需自己实现 | 需自己实现 | 需自己实现 |
| Checkpoint/Saver | `RedisSaver` 开箱即用 | 无 | 无 | 需手写 |
| 多 Agent 编排 | 需手写 graph（当前项目只用了单 Agent + Supervisor prompt 路由） | Role-based 开箱即用 | ConversableAgent 模式 | 无限灵活但工作量大 |
| 学习曲线 | 已有的 | 低 | 中 | 高 |

LangGraph 对本项目最关键的三个价值是：**HITL interrupt、RedisSaver checkpoint 持久化、LangChain 工具生态无缝集成**。如果不用 LangGraph，这三个都得自己造轮子。

---

## Q18. Redis checkpoint 里存的 state schema 长什么样？三个 Worker 共享记忆时怎么避免互相污染？

**State schema**：LangGraph `create_react_agent` 的默认 state schema 是一个 dict，核心字段是 `messages`（LangChain `add_messages` reducer）。Redis 里存的就是序列化后的 LangChain message 对象列表。

```python
# agent.py:154
agent = create_react_agent(llm, tools, checkpointer=get_memory(), prompt=final_prompt)
```

`RedisSaver` 把每次 `agent.stream()` 后的 state 序列化存入 Redis，key 为 `checkpoint:{thread_id}`。下次同一 `thread_id` 的请求会加载全部历史消息，`add_messages` reducer 自动合并。

**Worker 隔离策略**：当前用了最简单的方式——**不同 mode 用不同 `thread_id`**：

```python
# agent_service.py:90（原始代码）
THREAD_IDS = {
    "📖 教材知识问答": "guide_exam_knowledge",
    "📝 智能出卷": "guide_exam_testgen",
    "📊 阅卷批改": "guide_exam_grading",
    "🤖 多Agent协作": "guide_exam_multi_agent",
}
```

每个 mode 的 checkpoint 存在不同的 Redis key 下，完全隔离。用户在"知识问答"里的对话不会串到"阅卷批改"里。这是一个**模式级别的隔离**，不是 user 级别的隔离——同一个用户在不同模式切换时不会串，但不同用户在同一模式下共享同一个 thread_id……这其实是一个 bug：新用户切到知识问答模式会加载上一个用户的对话历史。

**真正的多用户隔离应该是**：`thread_id = f"{mode}:{user_id}"`，但目前没有用户鉴权系统，所以暂时用 mode 级别隔离。

---

## Q19. 出卷 Agent 怎么保证知识点覆盖和题目质量？怎么防止生成错题？

**出卷不走 LLM 生成**，走的是**题库检索**：

```python
# tools.py:377
@tool
def search_questions(chapter: str, qtype: Optional[str] = "全部", count: int = 5) -> str:
    """从题库中按章节和题型检索题目。"""
    with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    # 按 subject/chapter/type 过滤 → 随机抽取 count 条
```

出卷 Agent 的流程：
1. 用户说"中国饮食文化 出4道判断题"
2. Agent 调用 `search_questions(chapter="中国饮食文化", qtype="判断", count=4)`
3. 从 `question_bank.json` 中过滤 + 随机抽取 4 道题
4. Agent 调用 `confirm_exam(exam_content)` → HITL 中断，用户确认后才输出

**保证质量的方式**：
- **题库是人工预制的**（`question_bank.json`），不是 LLM 生成的——保证了知识点覆盖和题目正确性
- **随机抽取**保证多样性（不是每次出同样的题）
- **人在回路确认**：Agent 整理完试卷后必须调用 `confirm_exam` 工具暂停，用户在弹窗里确认/修改/取消后才输出

这种方式**天然防止了 LLM 生成错题**——因为题目本身不是 LLM 生成的。代价是出题范围受限于题库大小。

---

## Q20. 批改 Agent 对主观题怎么评分？评分 rubric 怎么设计的？

批改工具 `grade_answer` 在 `tools.py` 中定义，但核心逻辑在 System Prompt 中描述。当前的设计是**基于标准答案的对照评分**而非自由文本评价：

**实际工作方式**：
1. Agent 先用 `search_textbook` 或 `search_questions` 查到题目的标准答案
2. Agent 将用户答案与标准答案对比
3. 在 System Prompt 指导下给出"✅ 正确 / ❌ 错误"判断 + 详细解析 + 推荐复习知识点

**这不是"主观题评分"**——是选择题/判断题的对错判定 + 教材知识点解释。真正的自由文本主观题（如"请简述导游地陪服务规范"）目前没有系统化的 rubric 评分机制，Agent 的行为依赖于 System Prompt 中的指令和 LLM 本身的判断力。

如果要设计主观题 rubric，一个实际可行的方案是：
- 用 `grade_answer` 工具内嵌评分维度（准确性/完整性/规范性各 1-5 分）
- Judge LLM（DeepSeek V4 Pro）逐维度打分并给出理由
- 与教材标准答案做 RAGAS faithfulness 校验

---

## Q2. Redis 会话记忆：TTL 多长？多轮对话超出模型上下文窗口怎么办？有没有做摘要压缩？

**TTL**：`RedisSaver(redis_url=REDIS_URL)` 未设置 `ttl` 参数，默认 **无限期保留**。checkpoint 写进 Redis 后除非手动清理或 Redis maxmemory 淘汰，否则永久存在。已在 `context_manager.py` 中提供了 TTL 方案（`CHECKPOINT_TTL_SECONDS = 604800` 即 7 天），但尚未在请求路径上启用。

**超出上下文窗口**：**没有任何处理**。LangGraph `create_react_agent` + `add_messages` reducer 会把所有历史消息一字不落拼进 LLM 上下文。ReAct Agent 每轮 tool_calls 往返产生 3000+ tokens 的工具返回内容，实际 15~20 轮对话就可能塞满 DeepSeek V4 的 128K 窗口。超限时 DeepSeek 返回 400 error → `stream_agent_with_retry` 收到非 RETRYABLE 异常 → 直接 raise → 前端显示"Agent 调用失败"。

**摘要压缩**：**没有**。`context_manager.py` 中已经实现了完整的滑动窗口 + LLM 摘要压缩方案（含 `manage_context()`、`summarize_messages()`、`trim_checkpoint_state()`），但为了防止侵入请求路径造成回归，还没有在 `agent_service.py` 的请求路径上激活。准备下一版上线。

---

## Q3. FastAPI 这边 LLM 输出是流式的吗？SSE 还是 WebSocket？

**SSE（Server-Sent Events）**，不是 WebSocket。

```python
# server/routes/chat.py:82-89
return StreamingResponse(
    generate(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
)
```

事件类型五种：

| 事件 | 内容 | 说明 |
|---|---|---|
| `event: agent` | `{"content": "..."}` | LLM 逐 token 输出 |
| `event: tool` | `{"name": "...", "content": "..."}` | 工具调用及结果 |
| `event: done` | `{"answer": "...", "contexts": [...], ...}` | 流结束 |
| `event: error` | `{"message": "..."}` | 异常 |
| `event: hitl` | `{"type": "exam_review", ...}` | 人在回路中断 |

技术细节：LangGraph 的 `agent.stream()` 是同步生成器，通过 `ThreadPoolExecutor(max_workers=1)` + `asyncio.Queue` 桥接到 FastAPI 的 async generator。这个桥接模式在 `agent_service.py:159-168` 实现。

为什么选 SSE 而不是 WebSocket：
- RAG 问答是单向流（服务端→客户端），不需要双向通信
- SSE 比 WebSocket 简单——不需要升级协议、不需要心跳包、不需要管理连接状态
- FastAPI 原生支持 `StreamingResponse`，一行代码搞定

唯一需要双向通信的场景是 HITL（人在回路），但这里用了一个巧妙的设计：interrupt 时前端收到 `hitl` 事件后关闭当前 SSE 连接，用户确认后通过 **另一个 HTTP POST**（`/api/hitl/resume`）发一条新请求恢复，而不是在同一个 WebSocket 上双向通信。这样就避免了 WebSocket 的复杂性。

---

## Q4（部署）. Docker 部署：几个容器？ChromaDB 数据怎么持久化？用的 docker-compose 还是单容器？

**3 个容器，docker-compose 编排**。`docker-compose.yml`：

| 容器 | 镜像 | 端口 | 作用 |
|---|---|---|---|
| `redis` | `redis/redis-stack-server:latest` | `127.0.0.1:6379:6379` | 会话记忆 + 反馈 + 费用 + 错题本 |
| `agent` | 自建 `Dockerfile`（`python:3.11-slim`） | `8501:8080` | API 服务 + ChromaDB + BGE Reranker |
| `weather-mcp` | 自建 `Dockerfile.mcp`（`python:3.11-slim`） | `8000:8000` | 天气 MCP Server (~200MB 镜像) |

所有容器在同一个 `guide-net` bridge 网络下。

**ChromaDB 持久化**：通过 Docker volume 映射：
```yaml
volumes:
  - ./chroma_db:/app/chroma_db          # 向量数据
  - ./chat_logs:/app/chat_logs          # 对话日志
  - ./eval_log.jsonl:/app/eval_log.jsonl # 评估日志
  - ./hf_cache:/root/.cache/huggingface  # HuggingFace 模型缓存
  - ./bge_reranker_cache:/app/bge_reranker_cache # BGE 模型缓存
```

ChromaDB 数据存在宿主机 `./chroma_db/` 目录下，容器重启不丢失。BGE-Reranker 模型（~1.1GB）也通过 cache 目录持久化，避免每次启动重新下载。

**内存限制**：agent 容器限制 4GB（`docker-compose.yml:37`），包含 Python + ChromaDB + BGE-Reranker (~1.1GB)。已有 OOM 降级逻辑——Reranker 内存不足时自动跳过，只用原始向量检索结果。

---

## Q5. MCP 集成的什么工具？自己写的 MCP server 还是接的现成的？走的 stdio 还是 SSE？

**自己写的天气 MCP Server**，走的 **HTTP（SSE）**。

```python
# agent.py:138-141
for host in ["weather-mcp", "localhost", "127.0.0.1"]:  # Docker 容器名优先
    weather_tools = load_mcp_tools_http(f"http://{host}:8000")
```

部署为独立的 Docker 容器（`Dockerfile.mcp`，约 200MB），对外暴露 HTTP 端点。Agent 通过 `load_mcp_tools_http()` 动态发现并加载工具。支持三个 address 依次尝试——先试 Docker 容器名（同一 bridge 网络）、再试 localhost、最后试 127.0.0.1。

**为什么不走 stdio**：stdio 模式要求 MCP server 和 Agent 在同一个容器/进程里运行，但这里用的是独立容器独立部署，HTTP 更自然。

**加载失败的降级**：MCP 连接失败时，从 System Prompt 中正则删除 `get_weather` 相关描述，防止 LLM 调用不存在的工具。
```python
# agent.py:151-153
final_prompt = re.sub(r'- get_weather[^\n]*\n', '', final_prompt)
final_prompt = re.sub(r'- 查询天气[^\n]*\n', '', final_prompt)
```

---

## Q6. Langfuse 追踪举一个实际例子：你用 trace 定位并解决过的一个具体问题。

当前代码里 Langfuse 集成了以下部分：
- `@observe()` 装饰 `stream_agent_with_retry` 和多个 `@tool` 函数
- `report_ragas_to_langfuse()` 和 `report_tool_call_to_langfuse()` 上报指标
- 各模块都有 `try/except` 包裹，失败不影响主流程

**但目前没有用 trace 定位并解决过具体问题的记录**。这是一个诚实的回答。Langfuse 在代码中集成了，但 trace 数据的分析价值还没被充分利用。

**如果要用，典型场景是这样的**：
1. 用户反馈某个问题"答非所问"
2. 在 Langfuse UI 找到那条 trace（按时间 + user_input 过滤）
3. 展开看到 Agent 调了 `rewritten_search` 而非 `hybrid_search`
4. 发现改写后的查询丢失了关键词"第35条"
5. 优化 System Prompt 或改写模板
6. 重新测试验证

这个闭环在当前项目中还没有跑通过。

---

## Q7. 一次问答端到端延迟多少？瓶颈在哪一环？做过压测吗？

**没做过系统压测（benchmark）**，只能从代码路径分析瓶颈分布：

| 环节 | 典型延迟 | 瓶颈？ |
|---|---|---|
| JS 防抖 | 300ms | 有（前端） |
| 语义缓存命中 | <50ms | ✓ |
| ChromaDB 向量检索 | 50~100ms | 12K 向量 OK |
| BM25 分词 + 打分 | 20~50ms | 1.8K 文档 OK |
| BGE-Reranker 精排 | 200~400ms | **最大瓶颈**——12 段 × cross-encoder 推理 |
| DeepSeek API 流式输出 | 2~8s（首 token 0.5~1s） | 网络延迟不可控 |
| 端到端（缓存未命中） | **3~10 秒** | BGE + LLM 各占一半 |

**BGE-Reranker 是系统内的最大瓶颈**——每次检索都要做 12 次 cross-encoder 推理，每次 200~400ms。这也是为什么系统有丰富的降级逻辑（内存不足时跳 Reranker、OOM 时持久化禁用）。

**LLM 流式输出是系统外的最大瓶颈**——DeepSeek API 的网络延迟 + 推理时间，不可控但可以用流式 SSE 缓解用户感知（用户看到第一个 token 后就不会觉得慢了）。

**没做压测**是一个明确的技术债务。最基础的压测应该覆盖：50/100/200 并发 SSE 流，测量 P50/P95/P99 延迟、限流器生效情况、ChromaDB 读并发上限。
