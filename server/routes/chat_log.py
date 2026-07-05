# server/routes/chat_log.py
# Save chat conversations to disk

import json
import os
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/api/chat-log", tags=["chat-log"])

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chat_logs")


class ChatLogEntry(BaseModel):
    mode: str
    question: str
    answer: str
    contexts: List[str] = []
    ragas_scores: Optional[dict] = None
    feedback: Optional[str] = None


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


@router.post("")
async def save_chat_log(entry: ChatLogEntry):
    """Save a conversation turn to disk."""
    _ensure_dir()
    today = str(datetime.now().date())
    filepath = os.path.join(LOG_DIR, f"chat_{today}.jsonl")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": entry.mode,
        "question": entry.question,
        "answer": entry.answer,
        "contexts": entry.contexts,
        "context_count": len(entry.contexts),
        "ragas_scores": entry.ragas_scores,
        "feedback": entry.feedback,
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "ok", "file": os.path.basename(filepath)}


@router.get("/recent")
async def recent_logs(days: int = 3):
    """List recent chat log files."""
    _ensure_dir()
    from datetime import timedelta
    files = []
    for i in range(days):
        d = (datetime.now().date() - timedelta(days=i)).isoformat()
        path = os.path.join(LOG_DIR, f"chat_{d}.jsonl")
        if os.path.exists(path):
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
            files.append({"date": d, "file": os.path.basename(path), "count": count})
    return {"files": files}
