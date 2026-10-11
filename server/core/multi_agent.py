# server/core/multi_agent.py
# ── 真 Multi-Agent Supervisor 编排 ──
# StateGraph 搭 Supervisor → Workers。
# exam_worker 直接调 tool 函数 + StateGraph interrupt()——不嵌套 ReAct Agent。
# retrieval/grader 复用已有的单模式 Agent。

import os
import json
import re
import logging
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import Send, Command, interrupt

logger = logging.getLogger(__name__)


# ═══════════ State ═══════════

def _safe_add_messages(left, right):
    if isinstance(right, Command):
        inner = right.update
        if isinstance(inner, dict):
            return _safe_add_messages(left, inner)
        return left
    if isinstance(right, dict) and 'messages' in right:
        return add_messages(left, right['messages'])
    if isinstance(right, list):
        return add_messages(left, right)
    return left


class MultiAgentState(TypedDict):
    messages: Annotated[list, _safe_add_messages]
    routing: Optional[dict]
    task_instructions: str


# ═══════════ Supervisor ═══════════

_HARDCODED_SUPERVISOR_FALLBACK = """你是 Multi-Agent 调度器。分析用户请求，输出路由决策 JSON。

## 可用 Worker
- retrieval_worker: 教材知识检索
- exam_worker: 智能出卷
- grader_worker: 阅卷批改

## 路由规则
- 出卷/出题/抽题 → exam_worker
- 批改答案 → grader_worker
- 知识问答/查资料 → retrieval_worker

## 输出（严格 JSON）
{"reasoning":"简短","workers":["exam_worker"],"mode":"single","task_instructions":"用户原话"}"""


def _load_supervisor_prompt() -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath = os.path.join(base, "prompts", "supervisor_prompt.md")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return _HARDCODED_SUPERVISOR_FALLBACK


SUPERVISOR_SYSTEM_PROMPT = _load_supervisor_prompt()


def classify_intent(prompt: str) -> dict:
    from server.core.llm_provider import get_active_llm
    llm = get_active_llm(temperature=0, streaming=False)
    try:
        resp = llm.invoke(f"{SUPERVISOR_SYSTEM_PROMPT}\n\n用户请求：{prompt}\n\n请输出JSON路由决策：")
        text = resp.content.strip()
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        decision = json.loads(m.group(1) if m else text)
        if "workers" not in decision:
            worker = decision.get("worker", "retrieval_worker")
            decision["workers"] = [worker] if isinstance(worker, str) else worker
        if "mode" not in decision:
            decision["mode"] = "single"
        return {
            "reasoning": decision.get("reasoning", ""),
            "workers": decision["workers"],
            "mode": decision["mode"],
            "task_instructions": decision.get("task_instructions", prompt),
        }
    except Exception as e:
        logger.warning("Supervisor classify_intent failed: %s", e)
        exam_kw = ["出题", "道单选", "道多选", "道判断", "选题", "出卷", "抽题", "生成试卷"]
        grader_kw = ["批改", "判对错", "帮我改", "打分"]
        if any(kw in prompt for kw in exam_kw):
            return {"reasoning": "keyword", "workers": ["exam_worker"], "mode": "single", "task_instructions": prompt}
        if any(kw in prompt for kw in grader_kw):
            return {"reasoning": "keyword", "workers": ["grader_worker"], "mode": "single", "task_instructions": prompt}
        return {"reasoning": "fallback", "workers": ["retrieval_worker"], "mode": "single", "task_instructions": prompt}


# ═══════════ Helper ═══════════

def _get_user_input(state: MultiAgentState) -> str:
    user_input = state.get("task_instructions", "")
    if user_input:
        return user_input
    for m in reversed(state.get("messages", [])):
        role, content = "", ""
        if isinstance(m, dict):
            role = str(m.get('role', '') or m.get('type', '')).lower()
            content = str(m.get('content', ''))
        elif hasattr(m, 'type'):
            role = str(getattr(m, 'type', '')).lower()
            content = str(getattr(m, 'content', ''))
        if role in ('user', 'human') and content.strip():
            return content.strip()
    return ""


# ═══════════ Graph Nodes ═══════════

def supervisor_node(state: MultiAgentState, config=None) -> dict:
    user_input = _get_user_input(state)
    decision = classify_intent(user_input)
    logger.info("Supervisor: workers=%s, mode=%s", decision["workers"], decision["mode"])
    return {"routing": decision, "task_instructions": user_input}


def retrieval_worker_node(state: MultiAgentState, config=None) -> dict:
    from server.core.agent import get_agent_for_mode
    user_input = _get_user_input(state)
    try:
        agent = get_agent_for_mode("📖 教材知识问答")
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": f"r_{id(state)}"}},
        )
        msgs = result.get("messages", [])
        # Debug: log message types
        msg_types = []
        for m in msgs:
            t = getattr(m, 'type', None) or getattr(m, 'role', 'unknown')
            c = getattr(m, 'content', '') or ''
            msg_types.append(f"{t}(len={len(str(c))})")
        logger.info("retrieval_worker: input=%s, msgs=%d, types=%s",
                     user_input[:80], len(msgs), msg_types)
        print(f"🔥 [DEBUG] retrieval_worker msgs: {msg_types}", flush=True)
        return {"messages": msgs}
    except Exception as e:
        logger.error("retrieval_worker failed: %s", e, exc_info=True)
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content=f"检索时出错：{str(e)[:200]}，请稍后重试。")]}


def exam_worker_node(state: MultiAgentState, config=None) -> dict:
    from server.core.tools import search_questions

    user_input = _get_user_input(state)

    # 用一次 LLM 调用提取章节名（不做 tool calling，纯提取）
    chapter, qtype, count = _extract_exam_params(user_input)

    # 没有章节信息 → 直接提示用户，不调 LLM 编造
    if not chapter:
        return {"messages": [AIMessage(
            content="请指定出卷范围，例如：\n"
                    "• \"导游业务 第三章 出3道单选题\"\n"
                    "• \"政策与法律法规 合同法律制度 出5道多选题\"\n"
                    "• \"全国导游基础知识 中国历史文化 出4道判断题\""
        )]}

    # 调 search_questions
    sq_result = search_questions.invoke({"chapter": chapter, "qtype": qtype, "count": count})
    sq_content = str(sq_result)

    if "未找到" in sq_content:
        confirm_text = (
            f"出卷请求：{user_input}\n"
            f"章节：{chapter} | 题型：{qtype} | 数量：{count}\n"
            f"⚠️ 题库中未找到相关题目。"
        )
    else:
        confirm_text = f"出卷请求：{user_input}\n\n📝 题库结果：\n{sq_content}"

    response = interrupt({"type": "exam_review", "content": confirm_text})

    action = response.get("action", "confirm") if isinstance(response, dict) else "confirm"
    if action == "cancel":
        return {"messages": [AIMessage(content="已取消出卷。")]}

    if "未找到" in sq_content:
        answer = f"⚠️ 题库中未找到匹配「{chapter}」的题目。请指定教材名称 + 章节编号重试。"
    else:
        answer = f"📝 试卷：\n\n{sq_content}\n\n请按格式作答。"

    return {"messages": [AIMessage(content=answer)]}


def _extract_exam_params(user_input: str) -> tuple:
    """用 LLM 从用户输入中提取出卷参数：章节、题型、数量。

    正则解析"导游业务 第三章"、"政策法规 合同法律制度 出5道多选题" 这类输入，
    质量很差。改用 LLM 做结构化提取，准确性高很多。
    """
    from server.core.llm_provider import get_active_llm
    prompt = (
        "从用户输入中提取出卷参数，返回 JSON：\n"
        '用户输入："""' + user_input + '"""\n\n'
        "字段说明：\n"
        "- chapter: 用户指定的章节或学科范围。如果用户没有指定（如只说\"出3道题\"），返回空字符串 \"\"\n"
        "- qtype: 题型。必须是\"单选\",\"多选\",\"判断\"之一。没指定则返回\"单选\"\n"
        "- count: 题目数量。没指定则返回 3，最大 10\n\n"
        "只返回 JSON，不要其他文字。示例输出：\n"
        '{"chapter":"政策与法律法规 第八章","qtype":"多选","count":5}\n'
        '{"chapter":"导游业务 第三章","qtype":"单选","count":3}\n'
        '{"chapter":"中国饮食文化","qtype":"判断","count":4}\n'
        '{"chapter":"","qtype":"单选","count":3}\n'
    )
    try:
        llm = get_active_llm(temperature=0, streaming=False)
        if hasattr(llm, 'max_tokens'):
            llm.max_tokens = 80
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        # Extract JSON from possible markdown wrapping
        import re as _re
        m = _re.search(r'\{[^}]+\}', text)
        if m:
            data = json.loads(m.group(0))
        else:
            data = json.loads(text)

        chapter = data.get("chapter", "")
        qtype = data.get("qtype", "单选")
        # Normalize qtype
        qtype_map = {"单选题": "单选", "多选题": "多选", "判断题": "判断",
                      "single": "单选", "multiple": "多选", "judge": "判断"}
        qtype = qtype_map.get(qtype, qtype)
        if qtype not in ("单选", "多选", "判断"):
            qtype = "单选"
        count = max(1, min(10, int(data.get("count", 3))))
        return (chapter or "", qtype, count)
    except Exception as e:
        logger.warning("LLM exam param extraction failed: %s, falling back to regex", e)
        return _extract_exam_params_regex(user_input)


def _extract_exam_params_regex(user_input: str) -> tuple:
    """正则兜底提取出卷参数（LLM 失败时使用）。"""
    qtype_map = {"单选": "单选", "多选": "多选", "判断": "判断"}
    qtype = "单选"
    for kw, mapped in qtype_map.items():
        if kw in user_input:
            qtype = mapped
            break
    count = 3
    import re as _re
    m = _re.search(r'(\d+)\s*道', user_input)
    if m:
        count = max(1, min(10, int(m.group(1))))
    chapter = user_input
    ch_match = _re.search(r'(第[一二三四五六七八九十\d]+章)', user_input)
    if ch_match:
        prefix = user_input[:ch_match.start()].strip()
        if prefix:
            prefix = _re.sub(r'(出|抽|要|请|帮我|给我)\s*$', '', prefix).strip()
            chapter = (prefix + " " + ch_match.group(1)) if prefix else ch_match.group(1)
        else:
            chapter = ch_match.group(1)
    else:
        chapter = _re.sub(r'\d+\s*道.*$', '', user_input).strip()
        chapter = _re.sub(r'(单选题|多选题|判断题|选择题|单选|多选|判断).*$', '', chapter).strip()
        chapter = chapter.rstrip('出抽请给我帮 ')
    if not chapter or chapter == user_input:
        chapter = _try_llm_extract(user_input)
    return (chapter or "", qtype, count)


def _try_llm_extract(user_input: str) -> str:
    """LLM 兜底提取章节名。

    注意: 仅当输入中有明确的学科/教材线索时才调 LLM。
    如果输入没有任何章节信息（如"出3道题"），不调 LLM——防止 LLM 幻觉编造章节名。
    """
    # 快速检查：输入中是否有任何教材/学科/章节关键词
    has_subject_clue = bool(re.search(
        r'(导游|政策|法律|全国|地方|科目|教材|第[一二三四五六七八九十\d]+[章节篇]|知识)',
        user_input
    ))
    if not has_subject_clue:
        # 用户没有指定章节，返回空字符串让上层处理
        return ""

    from server.core.llm_provider import get_active_llm
    try:
        llm = get_active_llm(temperature=0, streaming=False)
        # 覆盖 max_tokens 限制（提取章节名只需短输出）
        if hasattr(llm, 'max_tokens'):
            llm.max_tokens = 30
        resp = llm.invoke(
            f"提取章节名（如'导游业务 第三章'），如果用户没有指定章节，返回'NONE'。只返回章节名或NONE：{user_input}"
        )
        result = resp.content.strip()
        if not result or result.upper() == "NONE":
            return ""
        return result
    except Exception:
        return ""


def grader_worker_node(state: MultiAgentState, config=None) -> dict:
    from server.core.agent import get_agent_for_mode
    user_input = _get_user_input(state)
    try:
        agent = get_agent_for_mode("📊 阅卷批改")
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": f"g_{id(state)}"}},
        )
        return {"messages": result.get("messages", [])}
    except Exception as e:
        logger.error("grader_worker failed: %s", e, exc_info=True)
        return {"messages": [AIMessage(content=f"批改时出错：{str(e)[:200]}，请稍后重试。")]}


# ═══════════ Graph Build ═══════════

WORKER_MAP = {
    "retrieval_worker": "retrieval_worker",
    "exam_worker": "exam_worker",
    "grader_worker": "grader_worker",
}


def route_after_supervisor(state: MultiAgentState):
    routing = state.get("routing", {})
    workers = routing.get("workers", ["retrieval_worker"])
    mode = routing.get("mode", "single")

    if mode == "parallel":
        sends = []
        for w in workers:
            if w in WORKER_MAP:
                branch = dict(state)
                branch["task_instructions"] = state.get("task_instructions", "")
                sends.append(Send(w, branch))
        return sends if sends else END

    first = workers[0]
    return first if first in WORKER_MAP else END


def build_supervisor_graph():
    builder = StateGraph(MultiAgentState)
    for name in ("supervisor", "retrieval_worker", "exam_worker", "grader_worker"):
        node_fn = globals()[f"{name}_node" if name != "supervisor" else "supervisor_node"]
        builder.add_node(name, node_fn)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", route_after_supervisor, {w: w for w in WORKER_MAP} | {END: END})
    for w in WORKER_MAP:
        builder.add_edge(w, END)

    return builder
