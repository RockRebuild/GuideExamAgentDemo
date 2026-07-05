# server/routes/question_bank.py
# Question bank query endpoints

from typing import Optional

from fastapi import APIRouter, Query

from server.models.schemas import QuestionBankResponse
from server.services.qb_service import query_bank, get_filter_options

router = APIRouter(prefix="/api/question-bank", tags=["question-bank"])


@router.get("", response_model=QuestionBankResponse)
async def list_questions(
    subject: Optional[str] = Query(None),
    chapter: Optional[str] = Query(None),
    qtype: Optional[str] = Query(None, alias="type"),
    keyword: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
):
    """Query question bank with filters and pagination."""
    result = query_bank(
        subject=subject,
        chapter=chapter,
        qtype=qtype,
        keyword=keyword,
        page=page,
        per_page=per_page,
    )
    # Add filter options for frontend dropdowns
    filters = get_filter_options(subject=subject)
    result["filters"] = filters
    return result


@router.get("/filters")
async def list_filters(subject: Optional[str] = Query(None)):
    """Get available filter options (subjects, chapters, types)."""
    return get_filter_options(subject=subject)
