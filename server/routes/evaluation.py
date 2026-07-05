# server/routes/evaluation.py
# RAGAS evaluation endpoints (async with polling)

from fastapi import APIRouter, HTTPException

from server.models.schemas import EvalStartRequest, EvalStartResponse, EvalStatusResponse
from server.services.eval_service import start_evaluation, get_eval_status

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


@router.post("/start", response_model=EvalStartResponse)
async def start_eval(request: EvalStartRequest):
    """Start RAGAS evaluation, return task_id for polling."""
    task_id = await start_evaluation(request.question, request.answer, request.contexts)
    return EvalStartResponse(task_id=task_id, status="running")


@router.get("/status/{task_id}", response_model=EvalStatusResponse)
async def eval_status(task_id: str):
    """Poll evaluation status/results."""
    status = get_eval_status(task_id)
    if status is None:
        raise HTTPException(404, "Evaluation task not found")
    return EvalStatusResponse(
        task_id=task_id,
        status=status["status"],
        scores=status["scores"],
        error=status["error"]
    )
