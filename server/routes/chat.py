# server/routes/chat.py
# SSE streaming chat endpoint

import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from server.models.schemas import ChatRequest, SanitizeResponse
from server.services.agent_service import stream_chat

router = APIRouter(prefix="/api/chat", tags=["chat"])

MAX_INPUT_LENGTH = 500
BLOCKED_KEYWORDS = [
    "system", "忽略", "ignore", "忘记", "重新开始", "越狱",
    "你是一个", "你的prompt", "你的system", "把你的指令给我"
]


def sanitize_input(user_input: str) -> tuple[str, str | None]:
    """Validate and sanitize user input."""
    if len(user_input) > MAX_INPUT_LENGTH:
        return user_input[:MAX_INPUT_LENGTH], f"输入已自动截断至 {MAX_INPUT_LENGTH} 字符"
    lower_input = user_input.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lower_input:
            return "", f"检测到不当关键词 '{keyword}'，请求被拒绝。如有疑问请联系管理员。"
    return user_input, None


@router.post("/sanitize", response_model=SanitizeResponse)
async def sanitize(request: ChatRequest):
    """Validate input before sending."""
    text, error = sanitize_input(request.prompt)
    if error:
        return SanitizeResponse(text=None, error=error)
    return SanitizeResponse(text=text, error=None)


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE streaming chat endpoint.
    Events:
      - event: tool  → data: {name, content}
      - event: agent → data: {content}
      - event: done  → data: {answer, tool_records, contexts, input_tokens, output_tokens}
      - event: error → data: {message}
    """
    # Sanitize
    text, error = sanitize_input(request.prompt)
    if error:
        async def error_gen():
            yield f"event: error\ndata: {json.dumps({'message': error}, ensure_ascii=False)}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    prompt = text

    async def generate():
        try:
            async for sse_event in stream_chat(request.mode, prompt):
                yield sse_event
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = json.dumps({"message": f"Agent 调用失败：{str(e)[:300]}"}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
