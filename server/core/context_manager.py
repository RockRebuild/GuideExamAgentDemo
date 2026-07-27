# server/core/context_manager.py
# ── 上下文窗口管理：token 估算、滑动窗口、LLM 摘要压缩 ──
#
# 问题背景:
#   LangGraph + RedisSaver checkpointer 会将所有历史消息一字不落拼进 LLM 上下文。
#   ReAct Agent 每轮 tool_calls 往返产生 3000+ tokens 的工具返回内容，
#   实际 15~20 轮对话就可能撑爆 DeepSeek V4 的 128K 上下文窗口。
#
# 方案:
#   1. 在 agent.stream() 之前读取 checkpoint state
#   2. 估算消息总 token 数
#   3. 超出软上限 → 滑动窗口保留最近 N 轮 → 旧消息 LLM 压缩为摘要
#   4. agent.update_state() 写回裁剪后的 checkpoint
#   5. 对长工具返回做正文截断（保留头部 + 尾部，中间省略）

import logging
import os
import re
from typing import List, Tuple, Optional

from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)

# ── 配置 ────────────────────────────────────────────
DEEPSEEK_MAX_INPUT = int(os.environ.get("CONTEXT_MAX_TOKENS", "128000"))
SAFE_RATIO = float(os.environ.get("CONTEXT_SAFE_RATIO", "0.75"))
SOFT_LIMIT = int(DEEPSEEK_MAX_INPUT * SAFE_RATIO)   # 默认 96000
MIN_RECENT_KEEP = int(os.environ.get("CONTEXT_MIN_RECENT_KEEP", "20"))
SUMMARIZE_ENABLED = os.environ.get("CONTEXT_SUMMARIZE_ENABLED", "true").lower() == "true"
MAX_TOOL_CONTENT_CHARS = int(os.environ.get("CONTEXT_MAX_TOOL_CHARS", "3000"))
CHECKPOINT_TTL_SECONDS = int(os.environ.get("CHECKPOINT_TTL_SECONDS", str(3600 * 24 * 7)))
MAX_SUMMARY_CHARS = int(os.environ.get("CONTEXT_MAX_SUMMARY_CHARS", "300"))


# ── Token 估算（纯函数，不依赖任何外部状态）────────────────

def _safe_str(val) -> str:
    """安全转字符串，处理 dict / list / None / int 等非预期类型。"""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return str(val)
    except Exception:
        return ""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # 如果传入了非字符串，先安全转换
    if not isinstance(text, str):
        text = _safe_str(text)
    if not text:
        return 0
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    other = len(text) - chinese
    return max(1, int(chinese / 1.5 + other / 3.5))


def _msg_tokens(msg) -> int:
    """单条消息的 token 数（含 role framing ~4 tokens）。

    防御性：msg 可能是 BaseMessage / tuple / list / dict / 任意类型。
    """
    tokens = 4
    content = getattr(msg, 'content', None) if hasattr(msg, 'content') else None
    if content is None:
        # tuple/list 格式 e.g. ("user", "hello")
        if isinstance(msg, (tuple, list)) and len(msg) >= 2:
            content = msg[1]
        else:
            return tokens

    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, (list, tuple)):
        for block in content:
            if isinstance(block, dict):
                tokens += estimate_tokens(block.get('text', ''))
            elif isinstance(block, str):
                tokens += estimate_tokens(block)
            # 其他类型（int / None / ...）跳过
    elif isinstance(content, dict):
        tokens += estimate_tokens(content.get('text', ''))
    # int / None / float 等类型直接跳过

    return tokens


def estimate_messages_tokens(messages) -> int:
    """估算消息列表的总 token 数。对任意类型列表都安全。"""
    if not messages:
        return 0
    total = 0
    for msg in messages:
        try:
            total += _msg_tokens(msg)
        except Exception:
            total += 4  # 最坏情况估算
    return max(1, total)


# ── 长工具返回截断 ───────────────────────────────────

def _truncate_tool_content(content: str, max_chars: int = MAX_TOOL_CONTENT_CHARS) -> str:
    if not isinstance(content, str):
        return _safe_str(content)[:max_chars]
    if len(content) <= max_chars:
        return content
    head_size = int(max_chars * 0.6)
    tail_size = max_chars - head_size - 20
    return (
        content[:head_size]
        + f"\n\n... [中间 {len(content) - head_size - tail_size} 字符已省略] ...\n\n"
        + content[-tail_size:]
    )


def _truncate_long_tool_messages(messages) -> list:
    """对过长的 ToolMessage 做截断处理。"""
    from langchain_core.messages import ToolMessage
    result = []
    changed = False
    for msg in (messages or []):
        if not isinstance(msg, ToolMessage):
            result.append(msg)
            continue
        content = getattr(msg, 'content', None)
        if not isinstance(content, str) or len(content) <= MAX_TOOL_CONTENT_CHARS:
            result.append(msg)
            continue
        new_content = _truncate_tool_content(content)
        try:
            new_msg = ToolMessage(
                content=new_content,
                tool_call_id=getattr(msg, 'tool_call_id', ''),
                name=getattr(msg, 'name', None),
            )
            result.append(new_msg)
            changed = True
        except Exception:
            result.append(msg)  # 构造失败保留原消息
    if changed:
        logger.info("✂️ 已截断工具返回内容")
    return result


# ── LLM 摘要压缩 ────────────────────────────────────

def _build_conversation_text(messages, max_messages: int = 40) -> str:
    lines = []
    for msg in (messages or [])[-max_messages:]:
        role = getattr(msg, 'type', None) or 'unknown'
        content = getattr(msg, 'content', '')
        if isinstance(content, str) and content:
            short = content[:200].replace('\n', ' ') + ("..." if len(content) > 200 else "")
            lines.append(f"[{role}]: {short}")
    return "\n".join(lines)


def summarize_messages(messages) -> str:
    if not messages:
        return ""

    conversation_text = _build_conversation_text(messages)

    try:
        from langchain_openai import ChatOpenAI

        summarizer = ChatOpenAI(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            temperature=0,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )

        prompt = (
            "将以下对话历史压缩为一段不超过 300 字的简短摘要。"
            "保留关键事实、用户问题、以及重要的回答要点。"
            "不要添加任何前缀或后缀，直接输出摘要内容。\n\n"
            f"{conversation_text}\n\n摘要："
        )

        response = summarizer.invoke(prompt)
        summary = _safe_str(response.content).strip()
        logger.info("📝 对话摘要生成: %d 条消息 → %d 字摘要", len(messages), len(summary))
        return f"[对话历史摘要] {summary}"

    except Exception as e:
        logger.warning("对话摘要生成失败: %s，使用简化摘要", e)
        truncated = _safe_str(conversation_text)[:MAX_SUMMARY_CHARS * 3]
        return f"[对话历史摘要] {truncated}"


# ── 主入口：上下文管理 ────────────────────────────────

def manage_context(
    existing_messages,
    soft_limit: int = SOFT_LIMIT,
) -> Tuple[list, dict]:
    if not existing_messages:
        return [], {"trimmed": False, "estimated_tokens": 0}

    msgs = _truncate_long_tool_messages(list(existing_messages))

    total_tokens = estimate_messages_tokens(msgs)

    metadata = {
        "trimmed": False,
        "original_count": len(msgs),
        "original_tokens": total_tokens,
        "estimated_tokens": total_tokens,
        "summarized": False,
        "discarded_count": 0,
    }

    if total_tokens <= soft_limit:
        return msgs, metadata

    logger.warning(
        "⚠️ 上下文超限: %d > %d tokens, 开始裁剪 (%d 条消息)",
        total_tokens, soft_limit, len(msgs),
    )

    # 分离 SystemMessage 和普通消息
    system_msgs = []
    normal_msgs = []
    for m in msgs:
        try:
            is_system = (
                (hasattr(m, 'type') and m.type == 'system')
                or isinstance(m, SystemMessage)
            )
        except Exception:
            is_system = False

        if is_system:
            system_msgs.append(m)
        else:
            normal_msgs.append(m)

    system_tokens = estimate_messages_tokens(system_msgs)
    margin = 20000  # 留给新消息的预算
    available = max(1, soft_limit - system_tokens - margin)

    kept = []
    kept_tokens = 0
    for msg in reversed(normal_msgs):
        mt = _msg_tokens(msg)
        if kept_tokens + mt > min(available, soft_limit * 0.8):
            if len(kept) >= MIN_RECENT_KEEP:
                break
        kept.insert(0, msg)
        kept_tokens += mt

    discarded = [m for m in normal_msgs if m not in kept]

    logger.info(
        "✂️ 滑动窗口: %d 条 → 保留 %d 条 (%.0f%%), 丢弃 %d 条",
        len(normal_msgs), len(kept),
        100 * len(kept) / max(1, len(normal_msgs)),
        len(discarded),
    )

    if not discarded:
        trimmed = system_msgs + kept
    elif SUMMARIZE_ENABLED:
        summary_text = summarize_messages(discarded)
        summary_msg = SystemMessage(content=summary_text)
        trimmed = system_msgs + [summary_msg] + kept
        metadata["summarized"] = True
    else:
        trimmed = system_msgs + kept

    metadata["trimmed"] = True
    metadata["discarded_count"] = len(discarded)
    metadata["kept_count"] = len(kept)
    metadata["estimated_tokens"] = estimate_messages_tokens(trimmed)

    logger.info(
        "✂️ 上下文裁剪完成: %d → %d 条, %d → %d tokens, 摘要=%s",
        len(msgs), len(trimmed),
        total_tokens, metadata["estimated_tokens"],
        "是" if metadata["summarized"] else "否",
    )

    return trimmed, metadata


# ── Checkpoint 状态裁剪 ──────────────────────────────

def trim_checkpoint_state(agent, config: dict, soft_limit: int = SOFT_LIMIT) -> dict:
    try:
        state = agent.get_state(config)
    except Exception as e:
        logger.warning("读取 checkpoint state 失败: %s", e)
        return {"trimmed": False, "error": str(e)}

    if not state:
        return {"trimmed": False, "estimated_tokens": 0, "reason": "no_state"}

    # 安全获取 messages 列表
    try:
        values = getattr(state, 'values', None)
        if values is None:
            return {"trimmed": False, "estimated_tokens": 0, "reason": "no_values"}
        # values 可能是 dict / dict-like / 其他类型，全部走 getattr 安全路径
        if hasattr(values, 'get'):
            existing_messages = values.get("messages", [])
        elif isinstance(values, dict):
            existing_messages = values.get("messages", [])
        else:
            return {"trimmed": False, "estimated_tokens": 0, "reason": f"unexpected_values_type:{type(values).__name__}"}
    except Exception as e:
        logger.warning("读取 state.values 失败: %s", e)
        return {"trimmed": False, "error": str(e)}

    if not existing_messages:
        return {"trimmed": False, "estimated_tokens": 0, "reason": "no_messages"}

    trimmed, metadata = manage_context(existing_messages, soft_limit)

    if metadata.get("trimmed"):
        try:
            agent.update_state(config, {"messages": trimmed})
            logger.info("✅ Checkpoint 状态已裁剪并写回 Redis")
        except Exception as e:
            logger.warning("写回裁剪后的 checkpoint 失败: %s", e)
            metadata["write_error"] = str(e)

    return metadata
