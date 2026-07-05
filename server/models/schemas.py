# server/models/schemas.py
# All Pydantic request/response models for the API

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str = Field(..., max_length=500)
    mode: str = Field(..., description="📖 教材知识问答 | 📝 智能出卷 | 📊 阅卷批改")


class SanitizeResponse(BaseModel):
    text: Optional[str] = None
    error: Optional[str] = None


class ToolRecord(BaseModel):
    name: str
    content: str


# ── Feedback ──────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback_type: str = Field(..., pattern="^(positive|negative)$")
    comment: Optional[str] = ""


class FeedbackStatsResponse(BaseModel):
    total: int
    positive_rate: float


# ── Evaluation ────────────────────────────────────────

class EvalStartRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]


class EvalStartResponse(BaseModel):
    task_id: str
    status: str  # "running"


class EvalStatusResponse(BaseModel):
    task_id: str
    status: str  # "running" | "done" | "error"
    scores: Optional[Dict[str, Optional[float]]] = None
    error: Optional[str] = None


# ── Question Bank ─────────────────────────────────────

class QuestionItem(BaseModel):
    id: str
    type: str
    subject: str
    chapter: str
    question: str
    options: List[str]


class QuestionBankResponse(BaseModel):
    items: List[QuestionItem]
    total: int
    page: int
    total_pages: int
    filters: dict  # {subjects, types, chapters}


# ── Cost ──────────────────────────────────────────────

class CostModelDetail(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float


class DailyCostResponse(BaseModel):
    models: Dict[str, CostModelDetail]
    total_cost: float
    trend: list  # [{date, cost, is_today}]
    budget: float = 5.0


# ── Modes ─────────────────────────────────────────────

class ModeInfo(BaseModel):
    name: str
    samples: List[str]


class ModesResponse(BaseModel):
    modes: List[ModeInfo]
