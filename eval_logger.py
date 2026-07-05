# eval_logger.py
# 评估日志：以 JSONL 格式记录每次提问的回答与 RAGAS 评估结果
# 可用于后续构建评估数据集、趋势分析、迭代优化

import json
import os
from datetime import datetime
from typing import Optional

EVAL_LOG_FILE = os.path.join(os.path.dirname(__file__), "eval_log.jsonl")


def _serialize_score(val) -> Optional[float]:
    """将 RAGAS 分数序列化为可读数字，None / 非数字返回 null"""
    if val is None:
        return None
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def log_evaluation(question: str, answer: str, contexts: list[str],
                   scores: dict, feedback: Optional[str] = None):
    """
    追加一条评估记录到 eval_log.jsonl

    参数
    ----
    question : str
        用户问题
    answer : str
        LLM 生成的回答
    contexts : list[str]
        经精排 + 去重后的检索上下文列表
    scores : dict
        RAGAS 四项指标分数，如 {"faithfulness": 0.92, ...}
    feedback : str | None
        用户反馈（"positive" / "negative"），评估时可能尚未反馈
    """
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "context_count": len(contexts) if contexts else 0,
        "contexts": contexts if contexts else [],
        "scores": {
            metric: _serialize_score(val)
            for metric, val in scores.items()
        },
        "feedback": feedback,
    }

    with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_last_feedback(feedback: str):
    """
    更新最近一条记录的 feedback 字段。
    因为评估和用户反馈通常不是同时发生的。
    """
    if not os.path.exists(EVAL_LOG_FILE):
        return

    lines = []
    with open(EVAL_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return

    # 找到最后一条，更新 feedback
    last = json.loads(lines[-1])
    last["feedback"] = feedback
    lines[-1] = json.dumps(last, ensure_ascii=False) + "\n"

    with open(EVAL_LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


def load_recent(n: int = 10) -> list[dict]:
    """读取最近 n 条记录"""
    if not os.path.exists(EVAL_LOG_FILE):
        return []
    records = []
    with open(EVAL_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[-n:]


def count_total() -> int:
    """统计总记录数"""
    if not os.path.exists(EVAL_LOG_FILE):
        return 0
    count = 0
    with open(EVAL_LOG_FILE, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count
