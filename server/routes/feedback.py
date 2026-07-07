# server/routes/feedback.py
# Feedback submission and statistics

import json
from datetime import datetime

import redis
from fastapi import APIRouter, Request

from server.models.schemas import FeedbackRequest, FeedbackStatsResponse
from server.core.eval_logger import update_last_feedback, log_feedback

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def get_redis(request: Request):
    """获取 Redis 客户端。不可用时返回 None。"""
    return getattr(request.app.state, "redis", None)


@router.post("")
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Submit user feedback (positive/negative)."""
    r = get_redis(request)
    if r is None:
        return {"status": "skipped", "reason": "Redis unavailable"}
    feedback_data = json.dumps({
        "user_input": body.question,
        "response": body.answer,
        "feedback": body.feedback_type,
        "comment": body.comment,
        "timestamp": datetime.now().isoformat()
    }, ensure_ascii=False)
    r.lpush("feedback:list", feedback_data)
    update_last_feedback(body.feedback_type)
    # 确保反馈一定落盘（即使没做过 RAGAS 评估）
    log_feedback(body.question, body.answer, body.feedback_type, body.comment or "")
    return {"status": "ok"}


@router.get("/stats", response_model=FeedbackStatsResponse)
async def feedback_stats(request: Request):
    """Get feedback statistics."""
    r = get_redis(request)
    if r is None:
        return FeedbackStatsResponse(total=0, positive_rate=0.0)
    total = r.llen("feedback:list")
    positive_count = 0
    if total > 0:
        for fb in r.lrange("feedback:list", 0, -1):
            data = json.loads(fb)
            if data.get("feedback") == "positive":
                positive_count += 1
        return FeedbackStatsResponse(
            total=total,
            positive_rate=round(positive_count / total, 2) if total > 0 else 0.0
        )
    return FeedbackStatsResponse(total=0, positive_rate=0.0)
