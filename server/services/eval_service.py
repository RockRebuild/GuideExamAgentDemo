# server/services/eval_service.py
# Background RAGAS evaluation with async task tracking

import asyncio
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
import os

from server.core.eval_logger import log_evaluation

# ── RAGAS Evaluation Models (same as app.py) ──────────

deepseek_client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)
judge_llm = llm_factory("deepseek-v4-pro", client=deepseek_client, max_tokens=4096,
                        extra_body={"thinking": {"type": "disabled"}})

async_client = AsyncOpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
ragas_embeddings = embedding_factory(
    "openai", model="text-embedding-v4",
    client=async_client, interface="modern"
)

RAGAS_MAX_CONTEXTS = 5

# ── In-memory task store ──────────────────────────────

_eval_tasks: dict[str, dict] = {}


def _safe_score(metric_name: str, score_call) -> Optional[float]:
    try:
        return score_call()
    except Exception as e:
        import traceback
        print(f"🔥 RAGAS {metric_name} 评估失败: {e}", flush=True)
        traceback.print_exc()
        return None


def _score_faithfulness(question: str, answer: str, contexts: list[str]) -> Optional[float]:
    return _safe_score("faithfulness", lambda: Faithfulness(llm=judge_llm).score(
        user_input=question, response=answer, retrieved_contexts=contexts))


def _score_answer_relevancy(question: str, answer: str) -> Optional[float]:
    return _safe_score("answer_relevancy",
        lambda: AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings).score(
            user_input=question, response=answer))


def _score_context_precision(question: str, answer: str, contexts: list[str]) -> Optional[float]:
    return _safe_score("context_precision",
        lambda: ContextPrecision(llm=judge_llm).score(
            user_input=question, retrieved_contexts=contexts, reference=answer))


def _score_context_recall(question: str, answer: str, contexts: list[str]) -> Optional[float]:
    return _safe_score("context_recall",
        lambda: ContextRecall(llm=judge_llm).score(
            user_input=question, retrieved_contexts=contexts, reference=answer))


def evaluate_current_answer(question: str, answer: str, contexts: list[str]) -> dict:
    """Run all 4 RAGAS metrics in parallel with staggered starts."""
    eval_contexts = contexts[:RAGAS_MAX_CONTEXTS] if len(contexts) > RAGAS_MAX_CONTEXTS else contexts

    scores: dict = {}
    metrics = [
        ("faithfulness",      lambda: _score_faithfulness(question, answer, eval_contexts)),
        ("answer_relevancy",  lambda: _score_answer_relevancy(question, answer)),
        ("context_precision", lambda: _score_context_precision(question, answer, eval_contexts)),
        ("context_recall",    lambda: _score_context_recall(question, answer, eval_contexts)),
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for i, (name, fn) in enumerate(metrics):
            futures[name] = pool.submit(fn)
            if i < len(metrics) - 1:
                time.sleep(3)
        for name, future in futures.items():
            scores[name] = future.result()

    return scores


# ── Async task management ────────────────────────────

async def start_evaluation(question: str, answer: str, contexts: list[str]) -> str:
    """Start evaluation in background, return task_id immediately."""
    task_id = str(uuid.uuid4())[:8]
    _eval_tasks[task_id] = {"status": "running", "scores": None, "error": None}

    asyncio.create_task(_run_evaluation(task_id, question, answer, contexts))
    return task_id


async def _run_evaluation(task_id: str, question: str, answer: str, contexts: list[str]):
    """Run RAGAS evaluation in thread pool (doesn't block event loop)."""
    loop = asyncio.get_event_loop()
    try:
        scores = await loop.run_in_executor(
            None, evaluate_current_answer, question, answer, contexts
        )
        _eval_tasks[task_id] = {"status": "done", "scores": scores, "error": None}

        # Log to JSONL
        log_evaluation(question=question, answer=answer, contexts=contexts, scores=scores)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _eval_tasks[task_id] = {"status": "error", "scores": None, "error": str(e)}


def get_eval_status(task_id: str) -> Optional[dict]:
    """Get evaluation task status."""
    return _eval_tasks.get(task_id)
