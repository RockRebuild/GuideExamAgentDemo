# server/routes/agent_eval.py
# Agent 行为评估 API
# 支持手动触发评估、查看报告、趋势分析

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/api/agent-eval", tags=["agent-eval"])


class SingleEvalRequest(BaseModel):
    """单条消息的 Agent 评估请求"""
    prompt: str
    called_tools: list = []
    mode: str = ""
    answer: str = ""


class EvalTriggerRequest(BaseModel):
    """触发评估请求"""
    modes: Optional[List[str]] = None  # 限定模式，默认全部


@router.post("/single")
async def evaluate_single_message(request: SingleEvalRequest):
    """对单条消息做轻量级 Agent 行为评估（前端按钮触发）。

    不做完整的 10 个标注用例遍历，只评估这一条消息：
    - 工具选择：根据 prompt 模式判断是否调了正确的工具
    - 端到端：检查 answer 是否满足基本成功模式
    """
    from server.core.agent_eval import evaluate_tool_selection, evaluate_end_to_end

    prompt = request.prompt
    called = request.called_tools
    mode = request.mode
    answer = request.answer

    # 根据模式推断期望工具
    if "出卷" in mode or "出题" in mode or "出" in prompt:
        expected = {"search_questions"}
        forbidden = {"grade_answer"}
        success_patterns = [r"题目|选项|ID:", r"单选|多选|判断"]
    elif "批改" in mode or "批改" in prompt or "答案" in prompt:
        expected = {"grade_answer"}
        forbidden = {"search_questions"}
        success_patterns = [r"正确|错误|✔|❌"]
    elif any(kw in prompt for kw in ["你好", "谢谢", "再见"]):
        expected = set()
        forbidden = set()
        success_patterns = []
    else:
        # 默认：知识问答类
        expected = {"search_textbook"}
        forbidden = {"search_questions", "grade_answer"}
        success_patterns = []

    tool_eval = evaluate_tool_selection(
        called, expected, set(), forbidden,
    )
    e2e_eval = evaluate_end_to_end(answer, success_patterns)

    combined = round(tool_eval["score"] * 0.6 + e2e_eval.get("score", 1.0) * 0.4, 4)

    return {
        "result": {
            "tool_score": tool_eval["score"],
            "tool_grade": tool_eval["grade"],
            "tool_details": tool_eval["details"],
            "e2e_success": e2e_eval.get("success", True),
            "e2e_score": e2e_eval.get("score", 1.0),
            "e2e_details": e2e_eval.get("details", ""),
            "combined_score": combined,
            "called_tools": called,
            "expected_tools": list(expected),
        }
    }


@router.get("/reports")
async def list_reports(limit: int = 5):
    """获取最近的评估报告列表。"""
    from server.core.agent_eval import load_recent_reports
    reports = load_recent_reports(limit)
    # 返回摘要（不包含详细 cases）
    summaries = []
    for r in reports:
        summaries.append({
            "timestamp": r.get("timestamp", ""),
            "summary": r.get("summary", {}),
            "by_category": r.get("by_category", {}),
            "by_difficulty": r.get("by_difficulty", {}),
        })
    return {"reports": summaries, "total": len(summaries)}


@router.get("/reports/latest")
async def latest_report():
    """获取最新的完整评估报告。"""
    from server.core.agent_eval import load_recent_reports
    reports = load_recent_reports(1)
    if not reports:
        return {"report": None}
    return {"report": reports[-1]}


@router.get("/trend")
async def eval_trend():
    """获取评估趋势数据。"""
    from server.core.agent_eval import get_trend
    return {"trend": get_trend()}


@router.post("/run")
async def trigger_evaluation(request: EvalTriggerRequest = None):
    """手动触发 Agent 行为评估。

    对标注测试用例运行评估，返回报告。
    """
    from server.core.agent_eval import run_full_evaluation, LABELED_TEST_CASES

    # 构建 agent_fn
    def agent_fn(mode, prompt):
        from server.core.agent import get_agent_for_mode
        from server.services.agent_service import THREAD_IDS

        agent = get_agent_for_mode(mode)
        from langchain_core.messages import SystemMessage, HumanMessage
        from server.core.agent import SYSTEM_PROMPT

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        config = {"configurable": {"thread_id": f"eval_{mode}_{hash(prompt)}"}}

        try:
            result = agent.invoke({"messages": messages}, config=config)
            final_messages = result.get("messages", [])
            answer = ""
            tool_records = []

            for msg in final_messages:
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                    if msg.__class__.__name__ == "AIMessage":
                        answer = msg.content
                if hasattr(msg, 'name') and msg.name:
                    tool_records.append({"name": msg.name, "content": msg.content or ""})

            # 也可能从 multi_agent 结果中获取
            if not answer and result.get("final_answer"):
                answer = result["final_answer"]

            return answer, tool_records
        except Exception as e:
            return f"Error: {str(e)}", []

    report = run_full_evaluation(agent_fn)
    return {"report": report}


@router.get("/cache-stats")
async def cache_stats():
    """获取语义缓存命中统计。"""
    from server.core.semantic_cache import get_cache_stats
    return get_cache_stats()


@router.delete("/cache")
async def clear_cache():
    """清空语义缓存。"""
    from server.core.semantic_cache import clear_cache
    count = clear_cache()
    return {"ok": True, "cleared": count}
