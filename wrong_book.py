# wrong_book.py
# 错题本：Redis 持久化，支持增删改查
import json
import os
from datetime import datetime
from typing import Optional

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
REDIS_KEY = "wrongbook:items"   # Redis Hash: {question_id: json}


def _get_redis() -> redis.Redis:
    """获取 Redis 连接，优先用 Docker 容器名，再试 localhost"""
    for host in ["redis", "localhost", "127.0.0.1"]:
        try:
            r = redis.Redis(host=host, port=6379, decode_responses=True, socket_connect_timeout=2)
            r.ping()
            return r
        except Exception:
            continue
    raise RuntimeError("Redis 不可用，错题本功能无法使用")


def _redis() -> Optional[redis.Redis]:
    """懒连接，兼容 Redis 不可用的情况"""
    try:
        return _get_redis()
    except Exception:
        return None


def record_wrong(
    question_id: str,
    question: str,
    user_answer: str,
    correct_answer: str,
    subject: str = "",
    chapter: str = "",
    qtype: str = "",
    explanation: str = "",
) -> dict:
    """记录一道错题。如果已存在则累加错误次数。返回更新后的条目。"""
    r = _redis()
    if r is None:
        return {}
    now = datetime.now().isoformat()
    existing_raw = r.hget(REDIS_KEY, question_id)
    if existing_raw:
        entry = json.loads(existing_raw)
        entry["wrong_count"] = entry.get("wrong_count", 1) + 1
        entry["user_answer"] = user_answer  # 更新为最新的错误答案
        entry["last_wrong_at"] = now
    else:
        entry = {
            "question_id": question_id,
            "question": question,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "subject": subject,
            "chapter": chapter,
            "type": qtype,
            "explanation": explanation,
            "wrong_count": 1,
            "first_wrong_at": now,
            "last_wrong_at": now,
        }
    r.hset(REDIS_KEY, question_id, json.dumps(entry, ensure_ascii=False))
    return entry


def delete_wrong(question_id: str) -> bool:
    """删除一道错题（用户掌握后手动移除）。"""
    r = _redis()
    if r is None:
        return False
    return r.hdel(REDIS_KEY, question_id) > 0


def get_all() -> list[dict]:
    """获取全部错题，按错误次数降序排列。"""
    r = _redis()
    if r is None:
        return []
    raw = r.hgetall(REDIS_KEY) or {}
    items = [json.loads(v) for v in raw.values()]
    items.sort(key=lambda x: x.get("wrong_count", 0), reverse=True)
    return items


def get_by_subject(subject: str) -> list[dict]:
    """按科目筛选错题。"""
    items = get_all()
    if not subject or subject == "全部":
        return items
    return [it for it in items if it.get("subject") == subject]


def get_by_chapter(subject: str, chapter: str) -> list[dict]:
    """按科目 + 章节筛选错题。"""
    items = get_by_subject(subject)
    if not chapter or chapter == "全部":
        return items
    return [it for it in items if it.get("chapter") == chapter]


def get_stats() -> dict:
    """错题统计：总数、按科目分布、按章节分布。"""
    items = get_all()
    by_subject = {}
    by_chapter = {}
    for it in items:
        s = it.get("subject", "未知")
        c = it.get("chapter", "未知")
        by_subject[s] = by_subject.get(s, 0) + 1
        key = f"{s} / {c}"
        by_chapter[key] = by_chapter.get(key, 0) + 1
    return {
        "total": len(items),
        "by_subject": by_subject,
        "by_chapter": by_chapter,
    }


def clear_all() -> int:
    """清空错题本。"""
    r = _redis()
    if r is None:
        return 0
    count = r.hlen(REDIS_KEY)
    r.delete(REDIS_KEY)
    return count


def detect_and_record(tool_name: str, tool_content: str) -> Optional[dict]:
    """从 grade_answer 工具返回内容中检测错题并自动记录。

    如果检测到 "❌ 回答错误"，解析内容并记录到错题本。
    """
    if tool_name != "grade_answer":
        return None
    if "❌ 回答错误" not in tool_content:
        return None

    # 从 grade_answer 返回文本中提取关键信息
    question = _extract_line(tool_content, "题目：")
    user_answer = _extract_line(tool_content, "你的答案：")
    correct_answer = _extract_line(tool_content, "正确答案：")
    explanation = _extract_line(tool_content, "解析：")

    if not question:
        return None

    # 尝试从 question_bank 中查找完整信息（补充 subject/chapter）
    qb = _load_question_bank()
    matched = _find_in_bank(qb, question)

    return record_wrong(
        question_id=matched.get("id", _make_id(question)),
        question=question,
        user_answer=user_answer or "",
        correct_answer=correct_answer or "",
        subject=matched.get("subject", ""),
        chapter=matched.get("chapter", ""),
        qtype=matched.get("type", ""),
        explanation=explanation or "",
    )


def detect_from_agent_text(agent_response: str) -> Optional[list[dict]]:
    """Fallback：从 LLM 的文本回复中检测错题（当 LLM 没调 grade_answer 工具时）。

    智能出卷模式下，LLM 可能直接在文本中批改而不调用工具。
    这里检测 "❌ 回答错误" 并尝试提取题目信息。
    """
    if "❌ 回答错误" not in agent_response:
        return None

    # 按 "❌ 回答错误" 拆分，可能有多道错题
    parts = agent_response.split("❌ 回答错误")
    if len(parts) < 2:
        return None

    results = []
    qb = _load_question_bank()

    for part in parts[1:]:  # 跳过第一个（"❌ 回答错误" 之前的内容）
        # 提取题目：优先找 "题目：" 行，否则用文本中间最长的句子
        question = _extract_line(part, "题目：")
        if not question:
            question = _extract_llm_question(part)

        # 提取答案
        user_answer = _extract_line(part, "你的答案：")
        if not user_answer:
            user_answer = _extract_llm_user_answer(part)
        correct_answer = _extract_line(part, "正确答案：")
        if not correct_answer:
            correct_answer = _extract_llm_correct_answer(part)
        explanation = _extract_line(part, "解析：")

        if not question:
            continue

        # 查题库补全科目章节
        matched = _find_in_bank(qb, question)

        result = record_wrong(
            question_id=matched.get("id", _make_id(question)),
            question=question,
            user_answer=user_answer or "",
            correct_answer=correct_answer or "",
            subject=matched.get("subject", ""),
            chapter=matched.get("chapter", ""),
            qtype=matched.get("type", ""),
            explanation=explanation or "",
        )
        if result:
            results.append(result)

    return results if results else None


def _extract_llm_question(text: str) -> str:
    """从 LLM 自由格式回复中提取题干。"""
    import re
    # 去掉 markdown 标记
    clean = re.sub(r'\*+|#+', '', text)
    # 找 "X题：" 或 "X题." 后面的内容
    m = re.search(r'(?:第[一二三四五六七八九十\d]+题|题目)[：:]\s*(.+?)(?:\n|你的|你选|正确|答案|解析)', clean)
    if m:
        return m.group(1).strip()
    # 兜底：取第一句超过15字的句子
    for line in clean.split('\n'):
        line = line.strip()
        if len(line) > 15 and '答案' not in line and '正确' not in line and '解析' not in line:
            return line
    return ""


def _extract_llm_user_answer(text: str) -> str:
    """从 LLM 自由格式回复中提取学员答案。"""
    import re
    m = re.search(r'(?:你选的答案是|你的答案[：:])\s*(.+?)(?:\n|。|$)', text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_llm_correct_answer(text: str) -> str:
    """从 LLM 自由格式回复中提取正确答案。"""
    import re
    m = re.search(r'(?:正确答案应为|正确答案[：:])\s*(.+?)(?:\n|。|（|$)', text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_line(text: str, prefix: str) -> str:
    """从文本中提取以 prefix 开头的那一行内容。"""
    for line in text.split("\n"):
        stripped = line.strip().lstrip('*').strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return ""


def _make_id(text: str) -> str:
    """根据题目文本生成简易 ID"""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _load_question_bank() -> list[dict]:
    """加载题库 JSON"""
    bank_path = os.path.join(os.path.dirname(__file__), "question_bank.json")
    try:
        with open(bank_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _find_in_bank(qb: list[dict], question_text: str) -> dict:
    """在题库中查找匹配的题目"""
    if not question_text:
        return {}
    text = question_text.strip().rstrip("。！？")
    # 精确匹配（截取前 30 字符）
    for q in qb:
        qtext = q.get("question", "").strip().rstrip("。！？")
        if qtext == text:
            return q
    # 模糊匹配：题目文本包含在题库中 或 题库题目包含在文本中
    for q in qb:
        qtext = q.get("question", "").strip()
        if len(qtext) >= 15 and (qtext in question_text or question_text in qtext):
            return q
    # 前30字符匹配
    short = text[:30]
    for q in qb:
        qtext = q.get("question", "").strip()
        if short in qtext or qtext[:30] in short:
            return q
    return {}
