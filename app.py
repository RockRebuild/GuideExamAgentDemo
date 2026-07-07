import os
import json
import re
import math
from datetime import datetime
from typing import Optional
from difflib import SequenceMatcher

import redis
import streamlit as st
import warnings

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from langfuse import Langfuse

from agent import stream_agent_with_retry, SYSTEM_PROMPT, get_agent_for_mode
from llm_service import LLMService
from eval_logger import log_evaluation, update_last_feedback, count_total

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── RAGAS 评估模型 ─────────────────────────────────
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

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*NoSessionContext.*")

# ── 题库加载（带缓存）────────────────────────────────
@st.cache_data
def load_question_bank() -> list[dict]:
    """加载题库 JSON，Streamlit 缓存避免每次刷新都读文件"""
    bank_path = os.path.join(os.path.dirname(__file__), "question_bank.json")
    try:
        with open(bank_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@st.cache_data
def get_question_bank_stats(questions: list[dict]) -> dict:
    """提取题库的科目、章节、题型列表（用于筛选器）"""
    subjects = sorted(set(q.get("subject", "未知科目") for q in questions))
    chapters = sorted(set(q.get("chapter", "未知章节") for q in questions))
    types = sorted(set(q.get("type", "未知题型") for q in questions))
    return {"subjects": subjects, "chapters": chapters, "types": types}

st.set_page_config(
    page_title="AI导游考试Agent-RAG智能问答系统",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAX_INPUT_LENGTH = 500
BLOCKED_KEYWORDS = [
    "system", "忽略", "ignore", "忘记", "重新开始", "越狱",
    "你是一个", "你的prompt", "你的system", "把你的指令给我"
]

feedback_redis = redis.Redis(host="redis", port=6379, decode_responses=True)

def sanitize_input(user_input: str) -> tuple[str, Optional[str]]:
    if len(user_input) > MAX_INPUT_LENGTH:
        return user_input[:MAX_INPUT_LENGTH], f"输入已自动截断至 {MAX_INPUT_LENGTH} 字符"
    lower_input = user_input.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lower_input:
            return "", f"检测到不当关键词 '{keyword}'，请求被拒绝。如有疑问请联系管理员。"
    return user_input, None

def _app_has_orphaned_tool_calls(messages: list) -> bool:
    """Check for AIMessages with tool_calls that lack a matching ToolMessage."""
    if not messages:
        return False
    answered_ids = {msg.tool_call_id for msg in messages
                    if isinstance(msg, ToolMessage) and msg.tool_call_id}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("id") not in answered_ids:
                    return True
    return False

def save_feedback(question, answer, feedback_type, comment=""):
    feedback_data = json.dumps({
        "user_input": question,
        "response": answer,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    }, ensure_ascii=False)
    feedback_redis.lpush("feedback:list", feedback_data)
    update_last_feedback(feedback_type)

RAGAS_MAX_CONTEXTS = 5

def _safe_score(metric_name: str, score_call) -> float | None:
    try:
        return score_call()
    except Exception as e:
        import traceback
        print(f"🔥 RAGAS {metric_name} 评估失败: {e}", flush=True)
        traceback.print_exc()
        return None

def _score_faithfulness(question: str, answer: str, contexts: list[str]) -> float | None:
    return _safe_score("faithfulness", lambda: Faithfulness(llm=judge_llm).score(
        user_input=question, response=answer, retrieved_contexts=contexts))

def _score_answer_relevancy(question: str, answer: str) -> float | None:
    return _safe_score("answer_relevancy",
        lambda: AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings).score(
            user_input=question, response=answer))

def _score_context_precision(question: str, answer: str, contexts: list[str]) -> float | None:
    return _safe_score("context_precision",
        lambda: ContextPrecision(llm=judge_llm).score(
            user_input=question, retrieved_contexts=contexts, reference=answer))

def _score_context_recall(question: str, answer: str, contexts: list[str]) -> float | None:
    return _safe_score("context_recall",
        lambda: ContextRecall(llm=judge_llm).score(
            user_input=question, retrieved_contexts=contexts, reference=answer))

def evaluate_current_answer(question: str, answer: str, contexts: list[str]) -> dict:
    import time
    import concurrent.futures

    eval_contexts = contexts[:RAGAS_MAX_CONTEXTS] if len(contexts) > RAGAS_MAX_CONTEXTS else contexts

    scores: dict = {}
    metrics = [
        ("faithfulness",      lambda: _score_faithfulness(question, answer, eval_contexts)),
        ("answer_relevancy",  lambda: _score_answer_relevancy(question, answer)),
        ("context_precision", lambda: _score_context_precision(question, answer, eval_contexts)),
        ("context_recall",    lambda: _score_context_recall(question, answer, eval_contexts)),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for i, (name, fn) in enumerate(metrics):
            futures[name] = pool.submit(fn)
            if i < len(metrics) - 1:
                time.sleep(3)
        for name, future in futures.items():
            scores[name] = future.result()

    return scores

@st.dialog("RAGAS 质量评估", width="small")
def _eval_progress_dialog():
    ctx = st.session_state.get("pending_eval_context", {})
    if not ctx:
        st.warning("没有可评估的内容，请先向助手提问。")
        if st.button("关闭"):
            st.rerun()
        return

    msg_id = st.session_state.get("last_msg_id", "")

    if st.session_state.pop("_do_eval", False):
        with st.spinner("⏳ 四项指标评估中（约需 1 分钟）..."):
            scores = evaluate_current_answer(ctx["prompt"], ctx["answer"], ctx["contexts"])

        log_evaluation(question=ctx["prompt"], answer=ctx["answer"],
                       contexts=ctx["contexts"], scores=scores)

        st.session_state[f"ragas_scores_{msg_id}"] = scores
        st.session_state.feedback_state[msg_id] = "evaluated"
        st.rerun()

    st.success("✅ 评估完成，点击关闭查看结果")
    if st.button("关闭", type="primary"):
        st.rerun()

# ── 工具展示辅助函数 ──────────────────────────────
def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def split_into_fragments(content: str) -> list[str]:
    parts = re.split(r'(?=【(?:片段\d+|来源)：)', content)
    return [p.strip() for p in parts if p.strip()]

# ── 侧边栏 & 模式切换 ───────────────────────────────
mode = st.sidebar.radio("选择模式", ["📖 教材知识问答", "📝 智能出卷", "📊 阅卷批改"])
agent = get_agent_for_mode(mode)
llm_service = LLMService(agent=agent, redis_client=feedback_redis)

if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode
elif st.session_state.last_mode != mode:
    st.session_state.feedback_state = {}
    st.session_state.last_prompt = None
    st.session_state.last_answer = None
    st.session_state.last_msg_id = None
    st.session_state.last_mode = mode
    st.session_state.last_ragas_scores = None
    st.session_state.pending_eval_context = None
    st.session_state.qb_page = 0  # 重置题库分页
    st.rerun()

with st.sidebar:
    st.title("📝 AI导游考试Agent-RAG智能问答系统")
    st.markdown("---")
    st.caption("技术栈：Python | LangChain | LangGraph | ChromaDB | Streamlit")
    st.caption("AI 模型：DeepSeek / 阿里云百炼")

    st.markdown("---")
    st.subheader("📊 反馈统计")
    total_feedback = feedback_redis.llen("feedback:list")
    positive_count = 0
    if total_feedback > 0:
        for fb in feedback_redis.lrange("feedback:list", 0, -1):
            data = json.loads(fb)
            if data.get("feedback") == "positive":
                positive_count += 1
        st.metric("总反馈数", total_feedback)
        st.metric("好评率", f"{positive_count / total_feedback * 100:.0f}%")
    else:
        st.caption("暂无反馈数据")

    total_evals = count_total()
    if total_evals > 0:
        st.caption(f"📝 已记录 {total_evals} 条评估日志（eval_log.jsonl）")

sample_questions = {
    "📖 教材知识问答": [
        "政策与法律法规的第二章主要讲了什么？",
        "查询未来五天的杭州天气",
        "《旅游法》第35条是什么？",
        "导游证的种类有哪些？"
    ],
    "📝 智能出卷": [
        "导游业务 团队导游服务规范 出3道单选题",
        "合同法律制度出5道多选题",
        "中国饮食文化 出4道判断题",
    ],
    "📊 阅卷批改": [
        "请批改题目 科目一的第一章单选第一题，我的答案是 B",
        "帮我批改题目 科目四的第十章多选第三题，我选 A，B",
    ]
}

st.title(mode)
st.markdown("---")
st.markdown("#### 💡 试试这些问题：")
cols = st.columns(2)
for i, sample in enumerate(sample_questions.get(mode, [])):
    with cols[i % 2]:
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            old_msg_id = st.session_state.get("last_msg_id", "")
            if old_msg_id:
                st.session_state.feedback_state.pop(old_msg_id, None)
                st.session_state.pop(f"ragas_scores_{old_msg_id}", None)
            st.session_state.pending_eval_context = None
            st.session_state.last_ragas_scores = None
            st.session_state.current_prompt = sample
            st.rerun()
st.markdown("---")

# ── 阅卷批改模式：题库浏览器 ────────────────────────────
if mode == "📊 阅卷批改":
    questions = load_question_bank()
    stats = get_question_bank_stats(questions)

    if questions:
        with st.expander("📚 题库浏览（点击展开）", expanded=False):
            # 筛选器
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.2, 1.5, 0.8, 1])
            with filter_col1:
                selected_subject = st.selectbox(
                    "科目", ["全部"] + stats["subjects"],
                    key="qb_subject"
                )
            with filter_col2:
                # 章节能选的随科目联动
                if selected_subject != "全部":
                    available_chapters = sorted(set(
                        q["chapter"] for q in questions
                        if q.get("subject") == selected_subject
                    ))
                else:
                    available_chapters = stats["chapters"]
                selected_chapter = st.selectbox(
                    "章节", ["全部"] + available_chapters,
                    key="qb_chapter"
                )
            with filter_col3:
                selected_type = st.selectbox(
                    "题型", ["全部"] + stats["types"],
                    key="qb_type"
                )
            with filter_col4:
                per_page = st.selectbox("每页", [10, 20, 50], key="qb_per_page", index=0)

            # 关键词搜索
            keyword = st.text_input("🔍 搜索题目关键词", key="qb_keyword",
                                    placeholder="输入关键词筛选题目...")

            # 筛选逻辑
            filtered = questions
            if selected_subject != "全部":
                filtered = [q for q in filtered if q.get("subject") == selected_subject]
            if selected_chapter != "全部":
                filtered = [q for q in filtered if q.get("chapter") == selected_chapter]
            if selected_type != "全部":
                filtered = [q for q in filtered if q.get("type") == selected_type]
            if keyword.strip():
                kw = keyword.strip()
                filtered = [q for q in filtered if
                            kw in q.get("question", "") or
                            kw in q.get("chapter", "") or
                            any(kw in opt for opt in q.get("options", []))]

            total_filtered = len(filtered)

            # 分页
            if "qb_page" not in st.session_state:
                st.session_state.qb_page = 0

            total_pages = max(1, math.ceil(total_filtered / per_page))
            if st.session_state.qb_page >= total_pages:
                st.session_state.qb_page = 0

            page_start = st.session_state.qb_page * per_page
            page_end = min(page_start + per_page, total_filtered)
            page_data = filtered[page_start:page_end]

            # 分页导航
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("⬅ 上一页", disabled=(st.session_state.qb_page == 0),
                             key="qb_prev", use_container_width=True):
                    st.session_state.qb_page -= 1
                    st.rerun()
            with nav_col2:
                st.caption(f"共 {total_filtered} 题 | 第 {st.session_state.qb_page + 1}/{total_pages} 页")
            with nav_col3:
                if st.button("下一页 ➡", disabled=(st.session_state.qb_page >= total_pages - 1),
                             key="qb_next", use_container_width=True):
                    st.session_state.qb_page += 1
                    st.rerun()

            st.divider()

            # 题目列表
            if not page_data:
                st.info("没有匹配的题目，请调整筛选条件。")
            else:
                # 题型着色
                TYPE_COLORS = {"单选": "blue", "多选": "orange", "判断": "green"}

                for idx, q in enumerate(page_data):
                    qtype = q.get("type", "")
                    color = TYPE_COLORS.get(qtype, "gray")
                    q_num = page_start + idx + 1

                    with st.container(border=True):
                        st.caption(
                            f"**#{q_num}** · :{color}[{qtype}] · "
                            f"{q.get('subject', '')} · {q.get('chapter', '')}"
                        )

                        # 题干
                        st.markdown(q.get('question', ''))

                        # 选项
                        opts = q.get("options", [])
                        if opts:
                            opt_text = "　".join(opts)
                            st.markdown(opt_text)
    else:
        st.warning("题库文件未找到，请检查 question_bank.json 是否存在。")

# ── 聊天输入处理 ────────────────────────────────────
if prompt := st.chat_input("请输入你的问题，或点击上方的示例问题..."):
    sanitized, error_msg = sanitize_input(prompt)
    if error_msg:
        st.warning(error_msg)
        st.stop()
    prompt = sanitized
    old_msg_id = st.session_state.get("last_msg_id", "")
    if old_msg_id:
        st.session_state.feedback_state.pop(old_msg_id, None)
        st.session_state.pop(f"ragas_scores_{old_msg_id}", None)
    st.session_state.pending_eval_context = None
    st.session_state.last_ragas_scores = None
    st.session_state.current_prompt = prompt

if "current_prompt" in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    st.session_state.current_prompt = None

    st.chat_message("user").write(prompt)

    tool_expander = st.expander("🔧 查看 Agent 思考过程", expanded=False)
    tool_records = []

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

    thread_id = "guide_exam_memory_0006"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception:
        state = None

    existing_messages = state.values.get("messages", []) if state else []

    # 检测孤儿 tool_calls（上次请求中断导致的状态损坏）
    if existing_messages and _app_has_orphaned_tool_calls(existing_messages):
        import time
        new_thread_id = f"{thread_id}_{int(time.time())}"
        print(f"⚠️ 检测到孤儿工具调用，创建新会话: {thread_id} → {new_thread_id}", flush=True)
        thread_id = new_thread_id
        config = {"configurable": {"thread_id": thread_id}}
        messages = [SystemMessage(content=SYSTEM_PROMPT), ("user", prompt)]
    elif not existing_messages:
        messages = [SystemMessage(content=SYSTEM_PROMPT), ("user", prompt)]
    else:
        messages = [("user", prompt)]

    RETRIEVAL_TOOLS = {"search_textbook", "hybrid_search", "multi_search",
                       "rewritten_search", "parent_child_search"}

    try:
        chunks = []
        final_answer = ""
        contexts = []

        for chunk in stream_agent_with_retry(agent, messages, config):
            chunks.append(chunk)
            if "tools" in chunk:
                tool_msg = chunk["tools"]["messages"][0]
                content = tool_msg.content or ""
                if len(content) > 3000:
                    display = (content[:3000]
                               + f"\n\n... （共 {len(content)} 字符，已截断显示）")
                else:
                    display = content
                tool_records.append(
                    f"🛠️ **调用工具：{tool_msg.name}**\n\n{display}"
                )

                if tool_msg.name in RETRIEVAL_TOOLS:
                    from tools import extract_contexts_from_response
                    tool_contexts = extract_contexts_from_response(tool_msg.content or "")
                    if tool_contexts:
                        contexts.extend(tool_contexts)

                # 自动记录错题
                if tool_msg.name == "grade_answer":
                    try:
                        from wrong_book import detect_and_record
                        detect_and_record(tool_msg.name, tool_msg.content or "")
                    except Exception:
                        pass

            if "agent" in chunk:
                final_answer += chunk["agent"]["messages"][0].content
                answer_placeholder.markdown(final_answer)

        input_tok, output_tok = llm_service.extract_token_usage_from_stream(chunks)
        llm_service._record_usage(input_tok, output_tok)

        # Fallback：从 LLM 文本回复中检测错题（智能出卷模式下可能没调 grade_answer 工具）
        if not any("grade_answer" in r for r in tool_records) and "❌ 回答错误" in final_answer:
            try:
                from wrong_book import detect_from_agent_text
                detect_from_agent_text(final_answer)
            except Exception:
                pass

        # ── 工具调用展示（终极完美版）──
        with tool_expander:
            if tool_records:
                for rec in tool_records:
                    parts = rec.split("\n\n", 1)
                    if len(parts) == 2:
                        tool_name, content = parts
                    else:
                        tool_name, content = rec, ""

                    st.markdown(tool_name)

                    fragments = split_into_fragments(content)
                    for frag in fragments:
                        escaped = escape_html(frag)
                        st.markdown(
                            f"<pre style='white-space: pre-wrap; width: 100%; margin: 0.5em 0;'>{escaped}</pre>",
                            unsafe_allow_html=True
                        )
                    st.divider()
            else:
                st.caption("本次未调用任何工具。")

        if final_answer:
            if contexts:
                deduped = []
                for ctx in contexts:
                    is_dup = False
                    for seen in deduped:
                        if SequenceMatcher(None, ctx, seen).ratio() > 0.8:
                            is_dup = True
                            break
                    if not is_dup:
                        deduped.append(ctx)
                contexts = deduped

            st.session_state.last_prompt = prompt
            st.session_state.last_answer = final_answer
            st.session_state.last_msg_id = str(hash(prompt))

            if contexts:
                st.session_state.pending_eval_context = {
                    "prompt": prompt,
                    "answer": final_answer,
                    "contexts": contexts
                }
            else:
                st.session_state.pending_eval_context = None
            st.session_state.last_ragas_scores = None
        else:
            st.warning("Agent 没有返回回答，请稍后重试。")

    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"Agent 调用失败：{str(e)[:300]}")

# ── 反馈 + 质量评估 ──────────────────────────────────
_METRIC_LABELS = {
    "faithfulness":       ("忠实度", "回答是否忠实于检索到的上下文，是否存在幻觉或曲解原文。"),
    "answer_relevancy":   ("答案相关性", "回答与用户问题的相关程度，是否偏离问题或答非所问。"),
    "context_precision":  ("上下文精度", "检索到的上下文中，真正对回答有用的比例。信号噪声比。"),
    "context_recall":     ("上下文召回", "回答中涉及的知识点，有多少被检索到的上下文覆盖。"),
}

def _render_ragas_scores_inline(scores: dict):
    cols = st.columns(4)
    for i, (metric, score) in enumerate(scores.items()):
        with cols[i]:
            label_cn, label_help = _METRIC_LABELS.get(metric, (metric, ""))
            if score is None:
                st.metric(label=f"⚪ {label_cn}", value="失败",
                          help=f"{label_help}\n（API 调用超时或出错，可重试）")
            else:
                numeric_score = float(score) if not isinstance(score, (int, float)) else score
                color = "green" if numeric_score >= 0.9 else "orange" if numeric_score >= 0.7 else "red"
                dot = "🟢" if color == "green" else "🟠" if color == "orange" else "🔴"
                st.metric(label=f"{dot} {label_cn}",
                          value=f"{numeric_score:.0%}",
                          help=label_help)
    st.caption(f"ℹ️ 评估基于前 {RAGAS_MAX_CONTEXTS} 条上下文（已精排）")

def render_feedback_and_eval_section():
    prompt = st.session_state.get("last_prompt")
    final_answer = st.session_state.get("last_answer")
    msg_id = st.session_state.get("last_msg_id")

    if not msg_id:
        return

    if "feedback_state" not in st.session_state:
        st.session_state.feedback_state = {}

    fb_state = st.session_state.feedback_state.get(msg_id, None)

    if fb_state is None:
        has_contexts = st.session_state.get("pending_eval_context") is not None
        if has_contexts:
            col1, col2, col3 = st.columns([1, 1, 1.3])
        else:
            col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("👍 有用", key=f"pos_{msg_id}", use_container_width=True):
                save_feedback(prompt, final_answer, "positive")
                st.session_state.feedback_state[msg_id] = "positive"
                st.rerun()
        with col2:
            if st.button("👎 无用", key=f"neg_{msg_id}", use_container_width=True):
                st.session_state.feedback_state[msg_id] = "pending_comment"
                st.rerun()
        if has_contexts:
            with col3:
                if st.button("🔍 质量评估", key=f"eval_{msg_id}",
                             type="secondary", use_container_width=True):
                    st.session_state.show_eval_dialog = True
                    st.session_state._do_eval = True
                    st.rerun()

    elif fb_state == "pending_comment":
        comment = st.text_area("请告诉我们哪里回答得不好（可选）：",
                              key=f"comment_{msg_id}")
        if st.button("提交反馈", key=f"submit_{msg_id}"):
            save_feedback(prompt, final_answer, "negative", comment)
            st.session_state.feedback_state[msg_id] = "done"
            st.rerun()

    elif fb_state == "evaluated":
        scores = st.session_state.get(f"ragas_scores_{msg_id}")
        if scores:
            with st.container(border=True):
                st.caption("📊 RAGAS 评估结果")
                _render_ragas_scores_inline(scores)
        else:
            st.caption("⚠️ 评估结果不可用")

    else:
        st.caption("✅ 感谢你的反馈！")

render_feedback_and_eval_section()

if st.session_state.get("show_eval_dialog"):
    st.session_state.show_eval_dialog = False
    _eval_progress_dialog()

llm_service.sidebar_usage()