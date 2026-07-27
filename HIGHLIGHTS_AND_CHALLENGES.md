# 项目亮点与难点

> 面试用。不涉及 Docker/部署，专注架构设计和工程深度。

---

## 亮点

### 1. 人在回路的工程化实现：Graph 级别的暂停/恢复

不是"前端弹个 confirm 框"那么简单，是整个 LangGraph 执行流的中断、持久化和恢复。

```
Agent 调用 confirm_exam(exam_content)
    → LangGraph interrupt() 暂停 graph 执行
    → checkpoint 完整写入 Redis（保留推理状态、工具调用历史）
    → SSE 流检测到 __interrupt__ chunk → 关闭连接，thread_id 传给前端
    → 用户确认后 → POST /api/hitl/resume → Command(resume=...) 从断点恢复
```

**为什么是亮点**：

- **不是 UI 层的中断，是 graph 状态机级别的暂停**。Agent 的推理链、工具调用栈、中间结果全部保存，恢复时从精确断点继续，不是"重新开始再拼接"
- **跨设备可恢复**：thread_id 存在前端，用户刷新页面、关浏览器、甚至换设备，只要 thread_id 没丢就能恢复
- **断点后支持修改**：用户不只是"确认/取消"，可以提交 `modifications` 参数（"把这题改成多选题"），Agent 从断点恢复时带着修改指令继续出卷

代码路径：`server/core/hitl_tool.py` → `server/services/agent_service.py:188-199` → `server/routes/hitl.py`

---

### 2. 五策略检索 + BGE Cross-Encoder 精排

不是"一份教材分块塞向量库然后查"的标准 RAG，而是 **5 种检索策略让 Agent 根据问题类型自动选择**：

| 策略 | 适用场景 | 做法 |
|---|---|---|
| `search_textbook` | 事实性问答 | 纯语义搜索 |
| `hybrid_search` | 精确条文（"第35条"） | 语义 + BM25(jieba) 双路 → 去重 → 精排 |
| `parent_child_search` | 需要完整上下文 | 子切片(120字符)召回 → 取 parent_id → 父切片(800字符)返回 |
| `multi_search` | 章节概览/宽泛问题 | 摘要层 + 段落层 + 句子层 三层并行 → 合并去重 |
| `rewritten_search` | 口语化/模糊问题 | LLM 改写 3 个变体 → 各搜 k=12 → 合并去重 |

五路结果全部汇入统一精排入口 `retrieve_with_rerank()`：

```
第一阶段（粗召回）: 各策略查 12~48 条不等
    → 文本去重 (SequenceMatcher, threshold=0.8)
第二阶段（精排）: BGE-Reranker cross-encoder
    → 拆句 → 逐句打分 → 段落取最佳句分
    → 过滤 min_score < 0.5
    → 动态 top_k（默认5 / 对比12 / 列表15）
最终喂给 LLM 的上下文: ≤15 段高质量段落
```

为什么不是"固定 k=3 搜一下就行"：导游考试题目类型多样——法律条文需要精确匹配（BM25 更适合），章节概述需要多粒度覆盖（multi_search 更适合），模糊口语问法需要改写变体（rewritten_search 更适合）。不同问题需要不同策略。

代码路径：`server/core/tools.py:210-373`（5 个 tool 函数）→ `server/core/retrieval_utils.py:155-253`（精排引擎）

---

### 3. 语义缓存：复用 ChromaDB 做缓存

用 ChromaDB 的相似度搜索能力做查询结果缓存：

```
新查询 → embedding → 在 "semantic_cache" collection 中相似搜索
    → 余弦相似度 > 0.95 → Cache Hit（比精确 match 更灵活）
    → 否则 → 执行检索 + 精排 → 结果存入缓存
```

**亮点细节**：

- **零额外依赖**：Chromadb 本来就在做向量搜索，缓存也复用同一个引擎
- **语义匹配不是关键词匹配**："导游证有几种"和"导游证的分类有哪些"在关键词匹配下是两次查询，在语义缓存下命中同一条
- **用户反馈联动**：点"有用" → `store(query, result)`；点"无用" → `remove_by_query(query)` 语义搜索并删除相似缓存条目
- **自动驱逐**：FIFO + 容量上限(f"{MAX_ENTRIES}") + TTL(``24h) + 命中率统计
- **写入失败不影响主流程**：`store()` 内部 try/except 全部吞掉

代码路径：`server/core/semantic_cache.py`

---

### 4. 全链路降级哲学

项目里几乎每个关键组件都有独立的降级路径，**不是 try/except 包一下了事，而是"这个组件炸了系统还能做什么"**：

| 组件 | 降级方式 | 亮点 |
|---|---|---|
| BGE-Reranker OOM | 三层防御：启动内存检查 → 推理 OOM 捕获 → **持久化禁用标记文件**防重启后无限循环 | 生产环境 ECS 4GB OOM Killer 踩出来的 |
| 天气 MCP 不可用 | 正则从 prompt 删除 get_weather 描述 | 防止 LLM 调用不存在的工具（比 try/except 更根本） |
| Redis 不可用 | Checkpointer 返回 None（无记忆仍可对话）、反馈跳过、错题本不可用 | 核心对话功能完全不受影响 |
| Supervisor 路由失败 | 默认回落 `retrieval_worker` | JSON 解析失败/LLM 返回格式错误/网络超时全部兜底 |
| RAGAS 单指标失败 | `_safe_score` 隔离，返回 None | 不阻塞其他三个指标的计算 |
| 孤儿 tool_calls | 检测到自动换 thread_id 重试 | 对用户透明 |

**降级不是"功能弱了"**——Reranker 跳过时返回原始上下文，用户感知不到；缓存写入失败时正常走检索，用户完全无感。降级路径保证的是"核心对话功能永远不会因为非核心组件挂掉而不可用"。

---

### 5. Agent 行为评估框架（不只是 RAG 质量评估）

通常 RAG 项目只做 RAGAS（评估检索/生成质量），这个项目多做了一层——**评估 Agent 的决策行为**：

```python
# agent_eval.py — 手写标注测试用例
LABELED_TEST_CASES = [
    {
        "id": "knowledge_001",
        "prompt": "导游证的种类有哪些？",
        "expected_tools": {"search_textbook"},           # 必须调用
        "alternative_tools": {"hybrid_search", ...},     # 也算对
        "forbidden_tools": {"search_questions", ...},    # 调用了算负分
        "success_patterns": [r"导游证", r"种类|分类"],    # 回答必须包含
        "category": "事实查询",
        "difficulty": "easy",
    },
    # ... 手写了约 20 条，覆盖知识问答/精确条文/出卷/批改
]
```

评估维度：
1. **工具选择准确率**：Agent 调用了 expected_tools 吗？调用了 forbidden_tools 吗？
2. **端到端成功率**：回答中包含 success_patterns 吗？
3. **响应延迟分布**：按类别和难度分布
4. **工具调用链路分析**：Agent 是先搜摘要再搜段落，还是直接 hybrid_search？

这在面试中叫"我不仅关心答案对不对，还关心 Agent 有没有用正确的方式得到答案"。

代码路径：`server/core/agent_eval.py`

---

## 难点

### 1. BGE Reranker 在小内存实例上的 OOM 攻防战

**问题**：BGE-Reranker-base 模型 1.1GB，推理时临时张量峰值 1.5GB。ECS 实例只有 4GB 内存，还要同时跑 Python、ChromaDB、FastAPI。OOM 之后 ECS 的 OOM Killer 杀容器 → 重启 → 第一次请求又加载模型 → 又 OOM → 无限循环。

**解决方案：三层防御**：

```
第 1 层: 启动时读取 /proc/meminfo，可用内存 < 1.5GB → 跳过加载
第 2 层: 推理时 RuntimeError("out of memory") → 标记 _reranker_disabled = True
第 3 层: 写入 /tmp/bge_reranker_disabled 持久化标记
         → 重启后检测到此文件 → 永久跳过 → 不再尝试加载
```

第三层是关键——**OOM Kill 后容器重启，没有这个持久化标记文件的话会无限循环：加载模型→OOM→被kill→重启→加载模型→...**。有了它，重启后直接用原始向量检索结果，虽然少了精排但系统稳定运行。

代码路径：`server/core/retrieval_utils.py:37-93`

---

### 2. 同步 LangGraph Stream 到异步 FastAPI 的阻抗匹配

LangGraph 的 `agent.stream()` 是同步生成器，FastAPI `StreamingResponse` 需要 async generator。这不是简单的 `async for`：

```python
# 桥接架构
┌─ ThreadPoolExecutor(max_workers=1) ─────────────┐
│  _run_sync():                                    │
│    agent.stream() → 同步生成器                    │
│       ↓                                          │
│    call_soon_threadsafe(queue.put_nowait)        │
└──────────────────┬───────────────────────────────┘
                   │ asyncio.Queue
                   ▼
┌─ Event Loop ─────────────────────────────────────┐
│  await queue.get()                               │
│       ↓                                          │
│  yield SSE event → FastAPI StreamingResponse     │
└──────────────────────────────────────────────────┘
```

实际实现中要处理的四个边界条件：

1. **线程安全投递**：必须用 `loop.call_soon_threadsafe(queue.put_nowait)` 而不是直接 `put_nowait`——后者不在事件循环线程上调用会抛异常
2. **线程泄漏**：每个请求创建 `ThreadPoolExecutor(max_workers=1)`，必须在 `finally` 里 `executor.shutdown(wait=False)`——否则 1000 并发 = 1000 个泄漏的线程
3. **HITL 中断传播**：`__interrupt__` 出现在 stream chunk raw JSON 字符串里，不是抛异常——需要解析 chunk 内容来检测是否为中断信号
4. **孤儿 tool_calls**：上次请求在工具执行中途断开（用户刷新/网络断开），checkpoint 里留下没有 ToolMessage 对应的 AIMessage → LangGraph 校验失败 → 只能换 thread_id

第 4 点是生产环境遇到过的真实 bug——代码里有专门的处理逻辑（`agent.py:204-212`）。

---

### 3. Agent 工具选择的黑盒问题

5 个检索工具给 LLM 选，但 LLM 的工具选择不是训练过的——它依赖 System Prompt 的文字描述来判断。同一个问题可以有多个合理选择——"旅游法第35条"用 `search_textbook` 还是 `hybrid_search`？没有唯一正确答案。

**为什么是难点**：

- **不是"召回率不够"的问题**：加更多段落、换更好的 embedding 模型都解决不了
- **本质上是一个不可微的离散决策**：LLM 选哪个工具是不可微的，没有梯度信号来优化
- **调试困难**：用户说"答得不好"，怎么判断是因为选了错误的检索策略，还是检索策略对了但向量搜出来的内容不对？

**当前应对**：

- 手写标注测试用例，定义 expected_tools 和 success_patterns
- `agent_eval.py` 跑全量用例后生成按类别/难度分布的报告
- 但测试集目前只有约 20 条——覆盖不足是明确的技术债务

---

## 面试话术

**一句话概括**：一个 **5 策略检索路由 + BGE 精排 + LangGraph 人在回路 graph 中断/恢复 + 全链路降级** 的导游考试 AI 问答系统。

**如果只能讲一个亮点**：人在回路。不是前端弹窗，是 LangGraph `interrupt()` 挂起整个 graph 执行 → checkpoint 序列化到 Redis → 用户换设备也能从 `Command(resume=...)` 恢复。这解决了传统 confirm 弹窗"刷新即丢"的问题。

**如果只能讲一个难点**：BGE Reranker 在小内存 ECS 上的 OOM 攻防。模型 1.1GB、实例 4GB、OOM Kill 后重启→再加载→再 Kill 的无限循环——通过持久化禁用标记文件（`/tmp/bge_reranker_disabled`）在重启后阻断循环，同时保持原始向量检索降级可用。

**如果需要体现系统性思考**：5 种检索策略不是拍脑袋加的——每个对应一种问题类型（精确条文→hybrid、口语化→rewritten、宽泛→multi_search），Agent 根据问题自动选择，背后有 Agent 行为评估框架验证"Agent 选对工具了吗"。
