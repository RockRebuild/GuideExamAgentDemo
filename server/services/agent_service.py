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
    thread_id = THREAD_IDS.get(mode, "guide_exam_default")
    agent = get_agent_for_mode(mode)

    # Check existing state for memory continuity
    config = {"configurable": {"thread_id": thread_id}}
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

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_sync():
        """Run the blocking stream generator in a background thread."""
        try:
            for chunk in stream_agent_with_retry(agent, messages, config):
                # Put chunk into async queue
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
        except Exception as e:
            import traceback
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait, e)

    # Start streaming in executor
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_run_sync)

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                yield _sse_event("error", {"message": str(chunk)[:300]})
                break

            chunks.append(chunk)

            if "tools" in chunk:
                tool_msg = chunk["tools"]["messages"][0]
                content = tool_msg.content or ""
                display = _normalize_pdf_text(content)
                if len(display) > 3000:
                    display = display[:3000] + f"\n\n... （共 {len(content)} 字符，已截断显示）"
                record = {"name": tool_msg.name, "content": display}
                tool_records.append(record)

                # Extract contexts from retrieval tools
                if tool_msg.name in RETRIEVAL_TOOLS:
                    tool_contexts = extract_contexts_from_response(tool_msg.content or "")
                    if tool_contexts:
                        contexts.extend(tool_contexts)

                # 自动记录错题（grade_answer 返回错误答案时）
                if tool_msg.name == "grade_answer":
                    try:
                        from server.core.wrong_book import detect_and_record
                        detect_and_record(tool_msg.name, tool_msg.content or "")
                    except Exception:
                        pass  # 错题记录失败不影响主流程

                yield _sse_event("tool", record)

            if "agent" in chunk:
                text = chunk["agent"]["messages"][0].content
                full_answer += text
                yield _sse_event("agent", {"content": text})

    finally:
        executor.shutdown(wait=False)

    # Dedupe contexts
    contexts = deduplicate_contexts(contexts)

    # Fallback：如果 LLM 没有调用 grade_answer 工具但回复中包含了批改结果，
    # 从 full_answer 中检测错题（智能出卷模式下 LLM 可能直接文本批改）
    if not any(r["name"] == "grade_answer" for r in tool_records) and "❌ 回答错误" in full_answer:
        try:
            from server.core.wrong_book import detect_from_agent_text
            detect_from_agent_text(full_answer)
        except Exception:
            pass

    # Compute token usage
    from server.core.llm_service import LLMService
    input_tok, output_tok = LLMService.extract_token_usage_from_stream(chunks)

    # Send done event
    done_data = {
        "answer": full_answer,
        "tool_records": tool_records,
        "contexts": contexts,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
    }
    yield _sse_event("done", done_data)


def _sse_event(event: str, data: dict) -> str:
    """Format a dict as an SSE event string."""
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
