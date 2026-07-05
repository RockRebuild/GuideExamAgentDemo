# server/services/qb_service.py
# Question bank loading, filtering, pagination

import json
import math
import os
from typing import Optional

QUESTION_BANK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "question_bank.json")

_bank_cache: Optional[list[dict]] = None


def load_bank() -> list[dict]:
    """Load question bank from JSON file (cached in memory)."""
    global _bank_cache
    if _bank_cache is None:
        try:
            with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as f:
                _bank_cache = json.load(f)
        except FileNotFoundError:
            _bank_cache = []
    return _bank_cache


def query_bank(
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    qtype: Optional[str] = None,
    keyword: Optional[str] = None,
    page: int = 1,
    per_page: int = 10,
) -> dict:
    """Filter and paginate question bank."""
    questions = load_bank()

    if subject and subject != "全部":
        questions = [q for q in questions if q.get("subject") == subject]
    if chapter and chapter != "全部":
        questions = [q for q in questions if q.get("chapter") == chapter]
    if qtype and qtype != "全部":
        questions = [q for q in questions if q.get("type") == qtype]
    if keyword and keyword.strip():
        kw = keyword.strip()
        questions = [q for q in questions if
                     kw in q.get("question", "") or
                     kw in q.get("chapter", "") or
                     any(kw in opt for opt in q.get("options", []))]

    total = len(questions)
    total_pages = max(1, math.ceil(total / per_page))
    start = (page - 1) * per_page
    end = start + per_page

    # Strip answers from response (security: don't expose to frontend)
    items = []
    for q in questions[start:end]:
        items.append({
            "id": q.get("id", ""),
            "type": q.get("type", ""),
            "subject": q.get("subject", ""),
            "chapter": q.get("chapter", ""),
            "question": q.get("question", ""),
            "options": q.get("options", []),
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "total_pages": total_pages,
    }


def get_filter_options(subject: Optional[str] = None) -> dict:
    """Get available filter options. If subject is provided, chapters are filtered."""
    questions = load_bank()
    subjects = sorted(set(q.get("subject", "未知科目") for q in questions))

    if subject and subject != "全部":
        chapters = sorted(set(
            q.get("chapter", "未知章节") for q in questions
            if q.get("subject") == subject
        ))
    else:
        chapters = sorted(set(q.get("chapter", "未知章节") for q in questions))

    types = sorted(set(q.get("type", "未知题型") for q in questions))
    return {"subjects": subjects, "chapters": chapters, "types": types}
