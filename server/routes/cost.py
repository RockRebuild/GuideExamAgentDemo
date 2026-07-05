# server/routes/cost.py
# API cost query endpoints

from datetime import date, timedelta

from fastapi import APIRouter, Request

from llm_service import LLMService, PRICES

router = APIRouter(prefix="/api/cost", tags=["cost"])


def _empty_cost():
    return {"models": {}, "total_cost": 0.0, "trend": [], "total_5d": 0.0, "budget": 5.0}


@router.get("/daily")
async def daily_cost(request: Request):
    """Get today's cost detail with per-model breakdown and 5-day trend."""
    r = getattr(request.app.state, "redis", None)
    if r is None:
        return _empty_cost()
    try:
        detail = LLMService.get_daily_cost_detail(r)

        # 5-day trend
        today = date.today()
        trend = []
        total_5d = 0.0
        for i in range(5):
            d = today - timedelta(days=i)
            day_cost = 0.0
            for model_str in PRICES:
                key = f"token_usage:{d}:{model_str}"
                usage = r.hgetall(key)
                if usage:
                    inp = int(usage.get("input", 0))
                    out = int(usage.get("output", 0))
                    day_cost += LLMService._calc_cost(inp, out, model_str)
            # Embedding model
            emb_key = f"token_usage:{d}:text-embedding-v4"
            emb_usage = r.hgetall(emb_key)
            if emb_usage:
                day_cost += LLMService._calc_embedding_cost(int(emb_usage.get("total", 0)))
            trend.append({
                "date": str(d),
                "cost": round(day_cost, 4),
                "is_today": i == 0,
            })
            total_5d += day_cost

        return {
            "models": detail.get("models", {}),
            "total_cost": detail.get("total_cost", 0.0),
            "trend": trend,
            "total_5d": round(total_5d, 4),
            "budget": 5.0,
        }
    except Exception as e:
        print(f"⚠️ 费用查询失败: {e}")
        return _empty_cost()


@router.get("/eval-count")
async def eval_count():
    """Get evaluation log count."""
    from eval_logger import count_total
    return {"count": count_total()}
