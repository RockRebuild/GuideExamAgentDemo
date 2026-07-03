import os

from retrieval_utils import refine_and_rerank

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["PYTHONIOENCODING"] = "utf-8"
import json
from datetime import datetime
from typing import Optional

import redis
import streamlit as st
import warnings

from langchain_core.messages import SystemMessage

from langfuse import Langfuse

from agent import stream_agent_with_retry, SYSTEM_PROMPT, get_agent_for_mode
from llm_service import LLMService

from openai import OpenAI, AsyncOpenAI
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.embeddings import embedding_factory, LangchainEmbeddingsWrapper
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)




# 如果你的系统支持，也强制 locale
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass

deepseek_client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 模型名填 deepseek-chat 或 deepseek-reasoner
judge_llm = llm_factory("deepseek-chat", client=deepseek_client, max_tokens=4096)

# Embedding 客户端
async_client = AsyncOpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 创建 modern embeddings
ragas_embeddings = embedding_factory(
    "openai",                              # 固定写法，表示 OpenAI 兼容协议
    model="text-embedding-v3",             # 你的模型名（DeepSeek: deepseek-embedding, DashScope: text-embedding-v1/v2）
    client=async_client,
    interface="modern"                     # 必须
)

# ======================= Langfuse 初始化 =======================
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# ======================= 全局配置 =======================
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*NoSessionContext.*")

st.set_page_config(
    page_title="导游考试 AI 助手",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAX_INPUT_LENGTH = 500
BLOCKED_KEYWORDS = [
    "system", "忽略", "ignore", "忘记", "重新开始", "越狱",
    "你是一个", "你的prompt", "你的system", "把你的指令给我"
]

# ======================= Redis 初始化 =======================
feedback_redis = redis.Redis(host="redis", port=6379, decode_responses=True)

# ======================= 输入过滤 =======================
def sanitize_input(user_input: str) -> tuple[str, Optional[str]]:
    if len(user_input) > MAX_INPUT_LENGTH:
        return user_input[:MAX_INPUT_LENGTH], f"输入已自动截断至 {MAX_INPUT_LENGTH} 字符"
    lower_input = user_input.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lower_input:
            return "", f"检测到不当关键词 '{keyword}'，请求被拒绝。如有疑问请联系管理员。"
    return user_input, None

# ======================= 反馈存储 =======================
def save_feedback(question, answer, feedback_type, comment=""):
    feedback_data = json.dumps({
        "user_input": question,
        "response": answer,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    }, ensure_ascii=False)
    feedback_redis.lpush("feedback:list", feedback_data)

# ======================= RAGAS 实时评估函数 =======================
def evaluate_current_answer(question: str, answer: str, contexts: list[str]) -> dict | None:
    try:
        scores = {}

        # Faithfulness
        faith = Faithfulness(llm=judge_llm)
        scores["faithfulness"] = faith.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts        # ← 改这里
        )

        # AnswerRelevancy
        relevancy = AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
        scores["answer_relevancy"] = relevancy.score(
            user_input=question,
            response=answer
        )

        # ContextPrecision
        precision = ContextPrecision(llm=judge_llm)
        scores["context_precision"] = precision.score(
            user_input=question,
            retrieved_contexts=contexts,       # ← 改这里
            reference=answer                   # 或你的标准答案
        )

        # ContextRecall
        recall = ContextRecall(llm=judge_llm)
        scores["context_recall"] = recall.score(
            user_input=question,
            retrieved_contexts=contexts,       # ← 改这里
            reference=answer
        )

        return scores
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.warning(f"RAGAS 评估失败: {e}")
        return None

# ======================= 侧边栏配置 =======================
mode = st.sidebar.radio("选择模式", ["📖 教材知识问答", "📝 智能出卷", "📊 阅卷批改"])
agent = get_agent_for_mode(mode)

llm_service = LLMService(agent=agent, redis_client=feedback_redis)

# 会话状态初始化
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
    st.rerun()

with st.sidebar:
    st.title("📝 导游考试 AI 助手")
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

    st.markdown("---")
    st.subheader("📊 本次回答评估")

    # 指标中文名及悬停说明
    _METRIC_LABELS = {
        "faithfulness":       ("忠实度", "回答是否忠实于检索到的上下文，是否存在幻觉或曲解原文。"),
        "answer_relevancy":   ("答案相关性", "回答与用户问题的相关程度，是否偏离问题或答非所问。"),
        "context_precision":  ("上下文精度", "检索到的上下文中，真正对回答有用的比例。信号噪声比。"),
        "context_recall":     ("上下文召回", "回答中涉及的知识点，有多少被检索到的上下文覆盖。"),
    }

    if "last_ragas_scores" in st.session_state and st.session_state.last_ragas_scores:
        for metric, score in st.session_state.last_ragas_scores.items():
            numeric_score = float(score) if not isinstance(score, (int, float)) else score
            color = "green" if numeric_score >= 0.9 else "orange" if numeric_score >= 0.7 else "red"
            dot = "🟢" if color == "green" else "🟠" if color == "orange" else "🔴"
            label_cn, label_help = _METRIC_LABELS.get(metric, (metric, ""))
            st.metric(label=f"{dot} {label_cn}", value=f"{numeric_score:.0%}", help=label_help)
    else:
        st.caption("暂无评估数据，点击回答下方的评估按钮开始")

# ======================= 示例问题 =======================
sample_questions = {
    "📖 教材知识问答": [
        "政策与法律法规的第二章主要讲了什么？",
        "全陪导游的职责是什么？",
        "《旅游法》第35条是什么？",
        "导游证的种类有哪些？"
    ],
    "📝 智能出卷": [
        "帮我出两道导游业务第三章的单选题目",
        "帮我找三道关于旅游法的多选题",
        "出五道判断题，范围是政策法规",
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
            st.session_state.current_prompt = sample
            st.rerun()
st.markdown("---")

# ======================= 聊天输入处理 =======================
if prompt := st.chat_input("请输入你的问题，或点击上方的示例问题..."):
    sanitized, error_msg = sanitize_input(prompt)
    if error_msg:
        st.warning(error_msg)
        st.stop()
    prompt = sanitized
    st.session_state.pending_eval_context = None
    st.session_state.current_prompt = prompt

# 处理来自按钮或输入的提示词
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    st.session_state.current_prompt = None

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        tool_expander = st.expander("🔧 查看 Agent 思考过程", expanded=False)
        tool_records = []
        answer_placeholder = st.empty()

        thread_id = "guide_exam_memory_0004"
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = agent.get_state(config)
        except Exception:
            state = None

        if state is None or not state.values.get("messages"):
            messages = [SystemMessage(content=SYSTEM_PROMPT), ("user", prompt)]
        else:
            messages = [("user", prompt)]

        try:
            chunks = []
            final_answer = ""
            contexts = []   # 用于 RAGAS 评估

            # 流式调用
            for chunk in stream_agent_with_retry(agent, messages, config):
                chunks.append(chunk)
                if "tools" in chunk:
                    tool_msg = chunk["tools"]["messages"][0]
                    tool_records.append(
                        f"🛠️ **调用工具：{tool_msg.name}**\n\n"
                        f"输入参数：{tool_msg.content}"
                    )
                    # 如果是教材检索，记录返回内容作为评估上下文
                    if tool_msg.name in ["search_textbook", "hybrid_search", "multi_search", "rewritten_search", "parent_child_search"]:
                        # tool_msg.content 是格式化后的片段文本，可以按段落拆分成多个context
                        raw_content = tool_msg.content
                        raw_contexts = []
                        segments = [s.strip() for s in raw_content.split("\n") if len(s.strip()) > 20]
                        raw_contexts.extend(segments)  # 先全部收集
                        # 最终处理
                        contexts = list(dict.fromkeys(raw_contexts))  # 去重（保持顺序）
                        contexts = contexts[:5]  # 只留前 5 个
                        contexts.append(tool_msg.content)

                if "agent" in chunk:
                    final_answer += chunk["agent"]["messages"][0].content
                    answer_placeholder.markdown(final_answer)

            # Token 统计
            input_tok, output_tok = llm_service.extract_token_usage_from_stream(chunks)
            llm_service._record_usage(input_tok, output_tok)

            # 工具调用展示
            with tool_expander:
                if tool_records:
                    for rec in tool_records:
                        st.markdown(rec)
                        st.divider()
                else:
                    st.caption("本次未调用任何工具。")

            # 最终回答与反馈
            if final_answer:
                st.session_state.last_prompt = prompt
                st.session_state.last_answer = final_answer
                st.session_state.last_msg_id = str(hash(prompt))

                # 存储评估上下文，等待用户手动点击评估按钮
                if contexts:
                    st.session_state.pending_eval_context = {
                        "prompt": prompt,
                        "answer": final_answer,
                        "contexts": contexts
                    }
                    st.session_state.last_ragas_scores = None
                else:
                    st.session_state.pending_eval_context = None
                    st.session_state.last_ragas_scores = None
            else:
                st.warning("Agent 没有返回回答，请稍后重试。")

        except Exception as e:
            import traceback
            traceback.print_exc()
            st.error(f"Agent 调用失败：{str(e)[:300]}")

# ======================= 反馈区域 =======================
def render_feedback_section():
    prompt = st.session_state.get("last_prompt")
    final_answer = st.session_state.get("last_answer")
    msg_id = st.session_state.get("last_msg_id")

    if not msg_id:
        return

    if "feedback_state" not in st.session_state:
        st.session_state.feedback_state = {}

    fb_state = st.session_state.feedback_state.get(msg_id, None)

    if fb_state is None:
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 有用", key=f"pos_{msg_id}"):
                save_feedback(prompt, final_answer, "positive")
                st.session_state.feedback_state[msg_id] = "positive"
                st.rerun()
        with col2:
            if st.button("👎 无用", key=f"neg_{msg_id}"):
                st.session_state.feedback_state[msg_id] = "pending_comment"
                st.rerun()

    elif fb_state == "pending_comment":
        comment = st.text_area("请告诉我们哪里回答得不好（可选）：", key=f"comment_{msg_id}")
        if st.button("提交反馈", key=f"submit_{msg_id}"):
            save_feedback(prompt, final_answer, "negative", comment)
            st.session_state.feedback_state[msg_id] = "done"
            st.rerun()

    else:
        st.caption("✅ 感谢你的反馈！")

render_feedback_section()

# ======================= RAG 评估按钮 =======================
if "pending_eval_context" in st.session_state and st.session_state.pending_eval_context:
    st.markdown("---")
    if st.button("🔍 评估回答质量", type="secondary"):
        ctx = st.session_state.pending_eval_context
        with st.spinner("🔍 正在评估回答质量..."):
            final_contexts = refine_and_rerank(ctx["contexts"], ctx["prompt"], top_k=5)
            scores = evaluate_current_answer(ctx["prompt"], ctx["answer"], final_contexts)
            st.session_state.last_ragas_scores = scores
        del st.session_state.pending_eval_context
        st.rerun()

# Token 使用量侧边栏展示
llm_service.sidebar_usage()