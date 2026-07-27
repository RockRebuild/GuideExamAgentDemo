# server/services/agent_service.py
# Async wrapper around LangGraph agent streaming

import asyncio
import re
import os
from typing import Optional

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from server.core.agent import get_agent_for_mode, stream_agent_with_retry, SYSTEM_PROMPT
from server.core.tools import extract_contexts_from_response, CONTEXT_SEPARATOR
from server.state import request_contexts, request_tool_records

RETRIEVAL_TOOLS = {"search_textbook", "hybrid_search", "multi_search",
                   "rewritten_search", "parent_child_search"}


def _has_orphaned_tool_calls(messages: list) -> bool:
    """Check if any AIMessage has tool_calls without a corresponding ToolMessage.

    This happens when a request is interrupted mid-tool-execution (e.g. user
    refreshes, connection breaks, or API retry after partial progress).
    """
    if not messages:
        return False
    # Collect all tool_call_ids that have a matching ToolMessage
    answered_ids = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            answered_ids.add(msg.tool_call_id)
    # Check if any AIMessage has a tool_call not in answered_ids
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("id") not in answered_ids:
                    return True
    return False


def _normalize_pdf_text(text: str) -> str:
    """合并 PDF 的硬换行：中文行末非句尾标点则拼接"""
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            result.append('')
            continue
        if result and result[-1]:
            prev = result[-1]
            # 上一行末尾是中文且不是句尾标点 → 直接拼接
            if re.search(r'[一-鿿]', prev[-1]) and not re.search(r'[。！？；：》）」』]$', prev):
                result[-1] = prev + stripped
                continue
        result.append(stripped)
    return '\n'.join(result)

# Each mode gets its own thread_id for separate conversation memory
THREAD_IDS = {
    "📖 教材知识问答": "guide_exam_knowledge",
    "📝 智能出卷": "guide_exam_testgen",
    "📊 阅卷批改": "guide_exam_grading",
    "🤖 多Agent协作": "guide_exam_multi_agent",
}


def deduplicate_contexts(contexts: list[str]) -> list[str]:
    """Deduplicate contexts by similarity (same logic as app.py)."""
    from difflib import SequenceMatcher
    if not contexts:
        return []
    deduped = []
    for ctx in contexts:
        is_dup = False
        for seen in deduped:
            if SequenceMatcher(None, ctx, seen).ratio() > 0.8:
                is_dup = True
                break
        if not is_dup:
            deduped.append(ctx)
    return deduped


async def stream_chat(mode: str, prompt: str):
    """
    Async generator yielding SSE-formatted strings.
    Each yield is a complete SSE event string (e.g. "data: {...}\n\n").
    """
    # 请求级超时：Agent 单次调用最长执行时间（环境变量可配）
    REQUEST_TIMEOUT_S = float(os.environ.get("AGENT_REQUEST_TIMEOUT_S", "120"))
    deadline = asyncio.get_event_loop().time() + REQUEST_TIMEOUT_S

    thread_id = THREAD_IDS.get(mode, "guide_exam_default")
    agent = get_agent_for_mode(mode)

    # 多 Agent 模式：每次请求用新 thread_id，防止历史 state 堆积撑爆上下文
    if mode == "🤖 多Agent协作":
        import time, uuid
        thread_id = f"{thread_id}_{uuid.uuid4().hex[:8]}"

    config = {"configurable": {"thread_id": thread_id}}
    state = None
    try:
        state = agent.get_state(config)
    except Exception:
        state = None

    existing_messages = state.values.get("messages", []) if state else []

    if existing_messages and _has_orphaned_tool_calls(existing_messages):
        # 孤儿 tool_calls：上次请求中断导致 checkpoint 中有未完成的工具调用。
        # LangGraph 会用 add_messages reducer 合并输入 → 历史消息 + 新消息，
        # 孤儿 AIMessage 仍在状态中 → 校验失败。只能换 thread_id 绕过。
        import time
        new_thread_id = f"{thread_id}_{int(time.time())}"
        print(f"⚠️ 检测到孤儿工具调用，创建新会话: {thread_id} → {new_thread_id}", flush=True)
        thread_id = new_thread_id
        config = {"configurable": {"thread_id": thread_id}}
        messages = [SystemMessage(content=SYSTEM_PROMPT), ("user", prompt)]
    elif not existing_messages:
        messages = [SystemMessage(content=SYSTEM_PROMPT), ("user", prompt)]
    else:
        messages = [("user", prompt)]

    # Reset per-request context
    request_contexts.set([])
    request_tool_records.set([])

    tool_records = []
    full_answer = ""
    contexts = []
    chunks = []

    # ── 多Agent协作：Supervisor 路由已内置在 StateGraph 中 ──
    # 不再单独调 classify_intent——graph 的 supervisor_node 自动做路由。
    # StateGraph 的 stream chunk 格式: {"node_name": output}
    # 嵌套的 React Agent worker 输出会带 "agent" 或 "tools" 子键。
    if mode == "🤖 多Agent协作":
        # 记录 supervisor 决策信息到 tool_records（通过解析第一个 chunk）
        pass  # routing 在 stream 中自然出现

    # ── 统一 stream 路径（单 Agent 和多 Agent 都走这里）──
    # 使用全局共享线程池，避免每个请求新建 ThreadPoolExecutor 的资源浪费。
    from server.core.executor import get_executor
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_sync():
        try:
            for chunk in stream_agent_with_retry(agent, messages, config):
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

    executor = get_executor()
    executor.submit(_run_sync)

    try:
        while True:
            # 硬超时保护：防止 Agent 调用永久挂起
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                yield _sse_event("error", {
                    "message": f"请求超时（{REQUEST_TIMEOUT_S:.0f}s），请简化问题或稍后重试"
                })
                return
            try:
                item = await asyncio.wait_for(queue.get(), timeout=min(remaining, 30))
            except asyncio.TimeoutError:
                yield _sse_event("error", {
                    "message": f"请求超时（{REQUEST_TIMEOUT_S:.0f}s），请简化问题或稍后重试"
                })
                return
            kind = item[0]
            if kind == "done":
                break
            if kind == "error":
                yield _sse_event("error", {"message": str(item[1])[:300]})
                return

            chunk = item[1]
            chunks.append(chunk)

            # ── HITL interrupt 检测：__interrupt__ 出现在 stream 中 ──
            if "__interrupt__" in chunk:
                interrupt_data = _extract_interrupt_value(chunk)
                if interrupt_data:
                    yield _sse_event("hitl", {
                        "type": "exam_review",
                        "content": interrupt_data.get("content", ""),
                        "thread_id": thread_id,
                        "mode": mode,
                    })
                    executor.shutdown(wait=False)
                    return  # 不 emit done，等用户恢复

            # ── 单 Agent 标准 stream 格式 ──
            if "tools" in chunk:
                tool_msg = chunk["tools"]["messages"][0]
                content = tool_msg.content or ""
                display = _normalize_pdf_text(content)
                if len(display) > 3000:
                    display = display[:3000] + "\n\n... (截断)"
                record = {"name": tool_msg.name, "content": display}
                tool_records.append(record)
                if tool_msg.name in RETRIEVAL_TOOLS:
                    tc = extract_contexts_from_response(tool_msg.content or "")
                    if tc:
                        contexts.extend(tc)
                if tool_msg.name == "grade_answer":
                    try:
                        from server.core.wrong_book import detect_and_record
                        detect_and_record(tool_msg.name, tool_msg.content or "")
                    except Exception:
                        pass
                yield _sse_event("tool", record)

            if "agent" in chunk:
                text = chunk["agent"]["messages"][0].content
                full_answer += text
                yield _sse_event("agent", {"content": text})

            # ── StateGraph 多 Agent 节点输出 ──
            # StateGraph 的 chunk key 是节点名（"supervisor"/"retrieval_worker"/...）
            # Worker 节点内部 agent.invoke() 返回 {"messages": [...]}，
            # 需要提取其中的 AI 回复和工具调用记录。
            if mode == "🤖 多Agent协作":
                for node_name in ("supervisor", "retrieval_worker", "exam_worker", "grader_worker"):
                    if node_name not in chunk:
                        continue
                    node_output = chunk[node_name]
                    if not isinstance(node_output, dict):
                        continue

                    # Supervisor 节点：提取路由决策展示给前端
                    if node_name == "supervisor":
                        routing = node_output.get("routing", {})
                        if routing:
                            workers = routing.get("workers", [])
                            rmode = routing.get("mode", "single")
                            mode_emoji = {"single": "➡️", "parallel": "⚡", "sequential": "🔗"}
                            mode_label = {"single": "路由到", "parallel": "并行分发到", "sequential": "串行调用"}
                            display = (
                                f"🔀 Supervisor\n"
                                f"{mode_emoji.get(rmode, '➡️')} {mode_label.get(rmode, rmode)}: {', '.join(workers)}\n"
                                f"分析: {routing.get('reasoning', '')}"
                            )
                            yield _sse_event("tool", {"name": "🤖 Supervisor", "content": display})
                            tool_records.append({"name": "🤖 Supervisor", "content": display})
                        continue

                    # Worker 节点：提取 AI 消息和工具调用
                    worker_msgs = node_output.get("messages", [])
                    for msg in worker_msgs:
                        msg_type = getattr(msg, 'type', None) or getattr(msg, 'role', '')
                        if msg_type in ('ai', 'assistant'):
                            text = getattr(msg, 'content', '') or ''
                            if isinstance(text, str) and text:
                                full_answer += text
                                yield _sse_event("agent", {"content": text})
                        elif msg_type == 'tool':
                            content = getattr(msg, 'content', '') or ''
                            display = _normalize_pdf_text(content)
                            if len(display) > 3000:
                                display = display[:3000] + "\n\n... (截断)"
                            name = getattr(msg, 'name', 'unknown')
                            record = {"name": name, "content": display}
                            tool_records.append(record)
                            if name in RETRIEVAL_TOOLS:
                                tc = extract_contexts_from_response(content or "")
                                if tc:
                                    contexts.extend(tc)
                            yield _sse_event("tool", record)

    # ── 收尾前检查：stream 正常结束但 state 中有未处理的 interrupt ──
    # 单 Agent: state.interrupts; 多 Agent/StateGraph: state.tasks
    try:
        state = agent.get_state(config)
        if state:
            interrupts_list = []
            if hasattr(state, 'interrupts') and state.interrupts:
                interrupts_list = state.interrupts
            if not interrupts_list and hasattr(state, 'tasks') and state.tasks:
                for task in state.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        interrupts_list.extend(task.interrupts)
            for iv in interrupts_list:
                if hasattr(iv, 'value') and isinstance(iv.value, dict):
                    yield _sse_event("hitl", {
                        "type": "exam_review",
                        "content": iv.value.get("content", ""),
                        "thread_id": thread_id,
                        "mode": mode,
                    })
                    return
    except Exception:
        pass

    # ── 多Agent: interrupt 已在 StateGraph 层，__interrupt__ 检测够用 ──

    # ── 收尾 ──
    contexts = deduplicate_contexts(contexts)

    if not any(r["name"] == "grade_answer" for r in tool_records) and "❌ 回答错误" in full_answer:
        try:
            from server.core.wrong_book import detect_from_agent_text
            detect_from_agent_text(full_answer)
        except Exception:
            pass

    # Token usage
    from server.core.llm_service import LLMService
    input_tok, output_tok = LLMService.extract_token_usage_from_stream(chunks)

    done_data = {
        "answer": full_answer,
        "tool_records": tool_records,
        "contexts": contexts,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "thread_id": thread_id,
    }
    yield _sse_event("done", done_data)


def _extract_interrupt_value(chunk: dict) -> dict | None:
    """从 LangGraph stream chunk 中提取 interrupt 数据。

    v1 mode chunk: {"__interrupt__": (Interrupt(value=..., id=...),)}
    """
    interrupt_tuple = chunk.get("__interrupt__")
    if not interrupt_tuple:
        return None
    for item in interrupt_tuple:
        if hasattr(item, "value"):
            return item.value if isinstance(item.value, dict) else {"content": str(item.value)}
    return None


def _sse_event(event: str, data: dict) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def resume_chat(mode: str, thread_id: str, action: str, modifications: dict = None):
    """恢复被 HITL interrupt 暂停的 Agent 执行。

    调用 agent.stream(Command(resume=...), config) 恢复 graph，
    产出与 stream_chat 相同的 SSE 事件流。
    """
    from langgraph.types import Command
    from server.core.agent import get_agent_for_mode, stream_agent_with_retry

    agent = get_agent_for_mode(mode)
    config = {"configurable": {"thread_id": thread_id}}

    resume_value = {"action": action}
    if modifications:
        resume_value["modifications"] = modifications

    tool_records = []
    full_answer = ""
    chunks = []
    contexts = []

    from server.core.executor import get_executor
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            for chunk in stream_agent_with_retry(
                agent, Command(resume=resume_value), config
            ):
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

    executor = get_executor()
    executor.submit(_run)

    try:
        while True:
            item = await queue.get()
            kind = item[0]
            if kind == "done":
                break
            if kind == "error":
                yield _sse_event("error", {"message": str(item[1])[:300]})
                return

            chunk = item[1]
            chunks.append(chunk)

            # 恢复流中也可能再次出现 interrupt（如修改后重新出卷）
            if "__interrupt__" in chunk:
                interrupt_data = _extract_interrupt_value(chunk)
                if interrupt_data:
                    yield _sse_event("hitl", {
                        "type": "exam_review",
                        "content": interrupt_data.get("content", ""),
                        "thread_id": thread_id,
                        "mode": mode,
                    })
                    executor.shutdown(wait=False)
                    return

            if "tools" in chunk:
                tool_msg = chunk["tools"]["messages"][0]
                content = tool_msg.content or ""
                display = _normalize_pdf_text(content)
                if len(display) > 3000:
                    display = display[:3000] + "\n\n... (截断)"
                record = {"name": tool_msg.name, "content": display}
                tool_records.append(record)
                if tool_msg.name in RETRIEVAL_TOOLS:
                    tc = extract_contexts_from_response(tool_msg.content or "")
                    if tc:
                        contexts.extend(tc)
                yield _sse_event("tool", record)

            if "agent" in chunk:
                text = chunk["agent"]["messages"][0].content
                full_answer += text
                yield _sse_event("agent", {"content": text})

            # ── StateGraph 多Agent节点输出（resume 路径）──
            if mode == "🤖 多Agent协作":
                for node_name in ("exam_worker", "retrieval_worker", "grader_worker"):
                    if node_name not in chunk:
                        continue
                    node_output = chunk[node_name]
                    if not isinstance(node_output, dict):
                        continue
                    worker_msgs = node_output.get("messages", [])
                    for msg in worker_msgs:
                        msg_type = getattr(msg, 'type', None) or getattr(msg, 'role', '')
                        if msg_type in ('ai', 'assistant'):
                            text = getattr(msg, 'content', '') or ''
                            if isinstance(text, str) and text:
                                full_answer += text
                                yield _sse_event("agent", {"content": text})
                        elif msg_type == 'tool':
                            content = getattr(msg, 'content', '') or ''
                            display = _normalize_pdf_text(content)
                            if len(display) > 3000:
                                display = display[:3000] + "\n\n... (截断)"
                            name = getattr(msg, 'name', 'unknown')
                            record = {"name": name, "content": display}
                            tool_records.append(record)
                            if name in RETRIEVAL_TOOLS:
                                tc = extract_contexts_from_response(content or "")
                                if tc:
                                    contexts.extend(tc)
                            yield _sse_event("tool", record)

    contexts = deduplicate_contexts(contexts)

    from server.core.llm_service import LLMService
    input_tok, output_tok = LLMService.extract_token_usage_from_stream(chunks)

    yield _sse_event("done", {
        "answer": full_answer,
        "tool_records": tool_records,
        "contexts": contexts,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "thread_id": thread_id,
    })
