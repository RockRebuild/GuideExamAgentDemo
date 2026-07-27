# server/routes/hitl.py
# Human-in-the-Loop 中断恢复端点
# Agent 调用 confirm_exam 工具触发 LangGraph interrupt() 后，
# 前端通过此端点恢复执行：POST /api/hitl/resume

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/hitl", tags=["hitl"])


class HITLResumeRequest(BaseModel):
    """HITL 恢复请求"""
    thread_id: str = Field(..., description="暂停的对话 thread_id")
    mode: str = Field(..., description="当前模式（用于获取正确的 Agent）")
    action: str = Field(..., description="用户操作: confirm | modify | cancel")
    modifications: Optional[dict] = Field(None, description="修改参数（action=modify 时）")


@router.post("/resume")
async def resume(request: HITLResumeRequest):
    """恢复被 LangGraph interrupt() 暂停的 Agent 执行。

    使用 Command(resume=...) 恢复 graph，
    以 SSE 流式返回 Agent 的后续输出。
    """
    from server.services.agent_service import resume_chat

    return StreamingResponse(
        resume_chat(
            mode=request.mode,
            thread_id=request.thread_id,
            action=request.action,
            modifications=request.modifications,
        ),
        media_type="text/event-stream",
    )
