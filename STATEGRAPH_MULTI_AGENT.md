# StateGraph 多 Agent 编排：从假 Multi-Agent 到真 StateGraph

## 背景

最初的多 Agent 是"假"的：一个 ReAct Agent 装 10 个工具，Supervisor LLM 做路由判断，但路由结果只作为前端展示——实际执行还是那个 Agent 自己在 10 个工具里选。

改造目标：用 LangGraph StateGraph 搭建真正的 Supervisor → Workers，每个 Worker 有独立工具集，Supervisor 的路由决策真正参与调度。

这个过程踩了很多坑，本文档从最终工作版本出发，倒推讲清楚设计决策。

## 整体架构

```
用户请求
    │
    ▼
┌──────────────────────────────────────────────┐
│ StateGraph                                    │
│                                               │
│  ┌──────────┐                                 │
│  │supervisor │ ← classify_intent(LLM)          │
│  │   node   │   输出 {"workers":[...], ...}    │
│  └─────┬────┘                                 │
│        │ conditional edge                      │
│        │ route_after_supervisor()              │
│        │                                       │
│   ┌────┼────────────┐                         │
│   ▼    ▼             ▼                         │
│ ┌────┐┌──────────┐┌───────┐                   │
│ │ret ││  exam    ││grader │                    │
│ │rie-││ _worker  ││_worker│                    │
│ │val ││          ││       │                    │
│ └──┬─┘└────┬─────┘└───┬───┘                   │
│    │       │            │                       │
│    └───────┴────────────┘                       │
│            │                                    │
│            ▼                                    │
│          END                                    │
└──────────────────────────────────────────────┘
```

## State Schema

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list, _safe_add_messages]
    routing: Optional[dict]
    task_instructions: str
```

三个字段：
- **messages**：全量消息，用自定义 reducer `_safe_add_messages`。并行模式时 `Send` API 的分支输出是 `Command` 对象，原生 `add_messages` 不认，需先解包
- **routing**：Supervisor 的决策结果，在 conditional edge 中消费
- **task_instructions**：当前用户指令，从 supervisor 一路传到 worker

`_safe_add_messages` 是关键：

```python
def _safe_add_messages(left, right):
    if isinstance(right, Command):        # Send API 并行的返回
        inner = right.update
        if isinstance(inner, dict):
            return _safe_add_messages(left, inner)
        return left
    if isinstance(right, dict) and 'messages' in right:
        return add_messages(left, right['messages'])
    if isinstance(right, list):
        return add_messages(left, right)
    return left
```

## Supervisor 路由

### LLM 分类 + 关键词兜底

```python
def classify_intent(prompt: str) -> dict:
    try:
        # 调用 DeepSeek Flash，temperature=0，thinking=disabled
        # 正则提取 JSON（兼容 markdown code block 包裹）
        ...
        return {"reasoning": ..., "workers": [...], "mode": "single|parallel", ...}
    except Exception:
        # 关键词兜底——不走 LLM，直接从用户输入判断
        exam_kw = ["出题", "道单选", "道多选", "道判断", ...]
        if any(kw in prompt for kw in exam_kw):
            return {"workers": ["exam_worker"], ...}
```

两层防护：LLM 调用成功就用 LLM（零温度保证一致性），LLM 失败（网络超时、API 限流）就用正则关键词兜底——不会因为一个非核心判断让整个请求挂掉。

### conditional edge

```python
def route_after_supervisor(state: MultiAgentState):
    if mode == "parallel":
        return [Send(w, dict(state)) for w in workers]  # Send API 并行分发
    return workers[0]  # single/sequential → 第一个 worker
```

`Send` 是 LangGraph 的并行原语：`Send(node_name, state_branch)` 会在同一 tick 内启动多个节点。`add_messages` reducer 自动合并分支结果。

## 三个 Worker 的设计差异

这是设计中最关键的部分。三个 Worker 的复杂度不同，用了三种不同的实现策略。

### retrieval_worker：复用已有模式

检索 Worker 不需要 interrupt，不需要特殊编排。直接复用 `get_agent_for_mode("📖 教材知识问答")`——一个已经跑通的 ReAct Agent 带 5 个检索工具。

```python
def retrieval_worker_node(state, config=None):
    agent = get_agent_for_mode("📖 教材知识问答")  # 已有的单模式 Agent
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    return {"messages": result.get("messages", [])}
```

每个请求用 `configurable.thread_id = f"r_{id(state)}"` 保证独立 checkpoint，不堆历史。

### exam_worker：直接调 tool + StateGraph interrupt

出卷 Worker 是核心，有 interrupt 需求（confirm_exam）。**这里做了一个关键的设计决策：不嵌套 ReAct Agent。**

```python
def exam_worker_node(state, config=None):
    # Step 1: 正则提取参数（章节、题型、数量）
    chapter, qtype, count = _extract_exam_params(user_input)

    # Step 2: 直接调 tool 函数，不走 Agent
    sq_result = search_questions.invoke({"chapter": chapter, ...})

    # Step 3: 在 StateGraph 层 interrupt
    # 关键！— 这里的 interrupt() 是 StateGraph 原生的
    # 不是嵌套在一个 ReAct Agent 的工具函数里
    response = interrupt({"type": "exam_review", "content": confirm_text})

    # Step 4: 用户确认/取消后从这里继续执行
    if action == "cancel":
        return {"messages": [AIMessage(content="已取消出卷。")]}
    return {"messages": [AIMessage(content=answer)]}
```

**为什么从嵌套 Agent 改成直接调 tool：**

之前试过在 StateGraph 节点里套一个 ReAct Agent，Agent 内部用 `confirm_exam` 工具触发 `interrupt()`。踩了三个死结：

1. **嵌套 GraphInterrupt 不冒泡**：Agent 内部的 `interrupt()` 产生的 `GraphInterrupt` 异常被 LangGraph 内部消化了，外层 StateGraph 收不到 `__interrupt__` chunk。stream 正常结束，前端根本不知道有中断在等待
2. **checkpointer 不共享**：内层 Agent 的 interrupt 状态存在自己的 checkpointer 里，外层 StateGraph 查不到。resume 时外层的 `Command(resume=...)` 无法路由到内层的 interrupt 点
3. **上下文爆炸**：Agent 每次 `invoke()` 都从 checkpointer 加载历史 state，多轮后消息堆到 150 万 token，直接打爆 DeepSeek 的 128K 窗口

直接调 tool 就没这些问题——`interrupt()` 是 StateGraph 自己的，stream 正确产生 `__interrupt__` chunk，`Command(resume=...)` 直接从同一个 checkpointer 恢复。

**参数提取用正则不用 LLM tool calling：**

```python
def _extract_exam_params(user_input):
    # "导游业务 第三章 出 3道单选题" → ("导游业务 第三章", "单选题", 3)
    # 正则提取 90% 的 case，LLM 兜底 10%
```

不用 tool calling 提取参数的原因是：tool calling 本质上也是一个 LLM 调用 + 函数选择，对于"从自然语言里扒几个字段"这种简单任务来说太重了。正则够快、够准、不耗 token。

### grader_worker：复用已有模式

批改 Worker 跟检索 Worker 一样——不需要 interrupt，直接复用 `get_agent_for_mode("📊 阅卷批改")`。

### Graph 的编译和注入

```python
# agent.py
def get_agent_for_mode(mode):
    if mode == "🤖 多Agent协作":
        from server.core.multi_agent import build_supervisor_graph
        builder = build_supervisor_graph()
        agent = builder.compile(checkpointer=_ensure_checkpointer("多Agent协作"))
        ...
```

`_ensure_checkpointer` 确保 Redis 可用时用 Redis（多进程共享），不可用时用 MemorySaver（本地开发不依赖 Redis）。

## HITL 中断/恢复全链路

### Stream 阶段

```
StateGraph.stream(config)  ← checkpointer 确保相同 thread_id
    │
    ▼
supervisor_node → classify_intent → routing
    │
    ▼
exam_worker_node → search_questions → interrupt()
    │
    ▼
StateGraph 暂停 → 产生 __interrupt__ chunk
    │
    │  agent_service.py 检测 __interrupt__
    │  → 提取 interrupt value → emit hitl SSE 事件
    │  → 关闭 SSE 连接，thread_id 发给前端
```

### Resume 阶段

```
用户点确认 → POST /api/hitl/resume {thread_id, action}
    │
    ▼
resume_chat() → get_agent_for_mode(mode)  ← 同一个 agent 实例
    │
    ▼
agent.stream(Command(resume={"action": "confirm"}), config)
    │
    │  checkpointer 根据 thread_id 恢复 state
    │  找到 interrupt 点 → 从 interrupt() 下一行继续执行
    │
    ▼
exam_worker_node 恢复 → 处理 action → 返回 {"messages": [...]}
    │
    ▼
StateGraph → END → stream 正常结束 → emit done
```

关键细节：StateGraph 的 `agent.stream(Command(resume=...), config)` 写法——`Command` 不要包在 `{"messages": ...}` 里。包进去会让 StateGraph 当成新请求从头跑，不会从中断点恢复。LangGraph 1.x 用 `agent.stream(Command(...), config)`，2.x 改用 `agent.invoke(Command(...), config)`。

## SSE 事件流处理

`agent_service.py` 里对 StateGraph 的 chunk 做了特殊处理。StateGraph 的 stream chunk 的 key 是节点名（`"supervisor"`、`"exam_worker"` 等），不是单 Agent 的 `"agent"`/`"tools"`：

```python
# StateGraph chunk: {"exam_worker": {"messages": [AIMessage(...)]}}
# 单 Agent chunk:  {"agent": {"messages": [AIMessage(...)]}}

if mode == "🤖 多Agent协作":
    for node_name in ("exam_worker", "retrieval_worker", "grader_worker"):
        if node_name in chunk:
            worker_msgs = chunk[node_name].get("messages", [])
            for msg in worker_msgs:
                if type == 'ai': yield _sse_event("agent", ...)
                if type == 'tool': yield _sse_event("tool", ...)
```

Supervisor 节点也单独处理——把路由决策格式化为 `🔀 Supervisor` 虚拟工具事件展示在前端。

## 设计决策和代价

### 为什么 StateGraph 而不是 CrewAI/AutoGen

1. 已有 LangChain 依赖，零额外安装
2. `interrupt()` + `Command(resume)` 是 LangGraph 独有，CrewAI 没有对等的 HITL 原语
3. `Send` API 做并行分发，语法干净
4. StateGraph 的 stream 接口跟单 ReAct Agent 一致，`agent_service.py` 的桥接代码不用改

### 为什么 exam_worker 不用 ReAct Agent

前面说了，嵌套 ReAct Agent 的 interrupt 不传播。更深层的原因是：

**StateGraph 节点是给纯函数设计的**——输入 state，输出 state 片段。ReAct Agent 是一个有状态、有 checkpointer、有 tool calling 循环的执行器，跟 StateGraph 节点的无状态假设冲突。

当一个有 checkpointer 的 ReAct Agent 被嵌在有 checkpointer 的 StateGraph 节点里时，两个 checkpointer 各自独立运行，interrupt 状态存在内层、resume 从外层发——根本对接不上。

### 参数提取为什么不用 LLM tool calling

Tool calling 的本质是"LLM 选择一个函数并填参数"——对于 `search_questions(chapter="导游业务 第三章", qtype="单选题", count=3)` 来说确实合适。但：

1. 这个 tool calling 需要额外部署一个 `ChatOpenAI` 调用，+500ms 延迟
2. 用户输入 90% 是格式化的（"教材名 第X章 出N道XX题"），正则完全覆盖
3. tool calling 可能填错参数类型，需要额外的校验和重试逻辑

正则不够用时 LLM 兜底——两层防护，不依赖单一路径。

### `_safe_add_messages` 为什么不直接用 `operator.add` 或 `add_messages`

LangGraph 的 `Send` API 在并行分支完成后，分支的返回值是 `Command(update={"messages": [...]})`。`operator.add` 只做 `list + list`，碰到 `Command` 就报 `can only concatenate list (not "Command") to list`。原生 `add_messages` 同样不认识 `Command` 对象——`Command` 是 LangGraph 的运行时的内部类型，`add_messages` 是 LangChain 的消息层工具。

所以需要一个薄包装：先检查是不是 `Command`，是就解包 `.update`，再交给 `add_messages`。

## 面试时怎么说

**30 秒版**：用 LangGraph StateGraph 搭的 Supervisor → Workers 编排。Supervisor LLM 做意图分类，`conditional_edge` + `Send` API 做单路/并行分发。出卷 Worker 在 StateGraph 层直接调 tool + `interrupt()` 做 HITL 确认，不嵌套 Agent。恢复时用 `Command(resume=...)` 从同一个 checkpoint 恢复。

**深入版（如果面试官追问"为什么不用简单的函数调用"）**：讲三个东西——StateGraph 的 HITL 原语（`interrupt` + `Command resume` 是其他框架没有的）、Send API 并行分发的状态合并（`_safe_add_messages` 为什么要自己写）、嵌套 Agent 的 interrupt 传播问题（为什么最终改成直接调 tool）。

**踩坑版（如果面试官问"遇到过什么难点"）**：讲嵌套 Agent 的 interrupt 不传播那个坑。内层 ReAct Agent 的 `interrupt()` 产生的 `GraphInterrupt` 被 LangGraph 吞了，外层 StateGraph 不知道有中断。试了共享 checkpointer、手动 `except GraphInterrupt` 然后在 StateGraph 层重新 `interrupt()`、`copy.deepcopy(config)` 爆了 pickle lock——最后决定砍掉内层 Agent，直接在 StateGraph 节点里调 tool。这个决策过程能体现你对框架内部机制的理解深度。
