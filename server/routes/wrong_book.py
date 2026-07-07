# server/routes/wrong_book.py
# 错题本 API 路由
from fastapi import APIRouter

from wrong_book import get_all, get_by_subject, get_stats, delete_wrong, clear_all

router = APIRouter(prefix="/api/wrong-book", tags=["wrong-book"])


@router.get("/list")
async def list_wrong(subject: str = "全部"):
    """获取错题列表，可按科目筛选。"""
    if subject and subject != "全部":
        items = get_by_subject(subject)
    else:
        items = get_all()
    return {"items": items, "total": len(items)}


@router.get("/stats")
async def wrong_stats():
    """错题统计信息。"""
    return get_stats()


@router.delete("/item/{question_id}")
async def remove_wrong(question_id: str):
    """删除单道错题（已掌握）。"""
    ok = delete_wrong(question_id)
    return {"ok": ok}


@router.delete("/clear")
async def clear_wrong():
    """清空错题本。"""
    count = clear_all()
    return {"ok": True, "deleted": count}
