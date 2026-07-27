# server/core/hitl_tool.py
# ── Human-in-the-Loop 工具：confirm_exam ──
# Agent 在输出试卷前调用此工具，内部使用 LangGraph interrupt() 暂停 graph 执行。
# 用户通过前端弹窗确认/修改/取消后，通过 Command(resume=...) 恢复执行。

from langgraph.types import interrupt


def confirm_exam(exam_content: str) -> str:
    """人在回路出卷确认工具。

    Agent 整理好试卷内容后，必须调用此工具进行确认。
    工具会暂停 graph 执行，将试卷内容展示给用户。
    用户可以选择：确认(confirm)、修改(modify)、取消(cancel)。

    Args:
        exam_content: 完整的试卷文本（题目 + 选项 + 答题格式提示）

    Returns:
        确认结果描述字符串，Agent 据此决定下一步操作。
    """
    response = interrupt({
        "type": "exam_review",
        "content": exam_content,
    })

    if isinstance(response, dict):
        action = response.get("action", "confirm")
    else:
        action = str(response) if response else "confirm"

    if action == "confirm":
        return "用户已确认出卷内容，请直接输出完整的试卷。"
    elif action == "modify":
        mods = response.get("modifications", {}) if isinstance(response, dict) else {}
        return f"用户要求修改：{mods}。请根据修改要求重新生成试卷。"
    elif action == "cancel":
        return "用户取消了出卷请求。请礼貌告知用户已取消。"
    return "用户已确认，请继续。"
