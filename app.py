import asyncio
import hashlib
from typing import Dict, Any

import redis
import streamlit as st

from llm_service import LLMService
from agent import stream_agent_with_retry, SYSTEM_PROMPT, get_agent_for_mode
from langchain_core.messages import SystemMessage
import warnings
import json
from datetime import datetime, date

AGENT_SEMAPHORE = asyncio.Semaphore(3)




# 屏蔽无关警告，保持界面干净
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*NoSessionContext.*")

FEEDBACK_FILE = "feedback.json"
st.set_page_config(
    page_title="导游考试 AI 助手",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAX_INPUT_LENGTH = 500          # 限制用户输入字符数
BLOCKED_KEYWORDS = [            # 敏感词清单（可根据需要扩充）
    "system", "忽略", "ignore",
    "忘记", "重新开始", "越狱",
    "你是一个", "你的prompt",
    "你的system", "把你的指令给我"
]

mode = st.sidebar.radio("选择模式", ["📖 教材知识问答", "📝 智能出卷", "📊 阅卷批改"])
agent = get_agent_for_mode(mode)
# 动态获取 Agent

def sanitize_input(user_input: str) -> tuple[str, str | None]:
    """
    输入过滤：长度截断 + 敏感词检测
    返回 (处理后的输入, 错误信息)
    """
    # 长度检查
    if len(user_input) > MAX_INPUT_LENGTH:
        return user_input[:MAX_INPUT_LENGTH], f"输入已自动截断至 {MAX_INPUT_LENGTH} 字符"

    # 敏感词检查
    lower_input = user_input.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lower_input:
            return "", f"检测到不当关键词 '{keyword}'，请求被拒绝。如有疑问请联系管理员。"

    return user_input, None

# 复用已有的 Redis 连接（本地默认端口）
feedback_redis = redis.Redis(host="redis", port=6379, decode_responses=True)
llm_service = LLMService(agent=agent, redis_client=feedback_redis)

async def async_stream_agent(messages: list, config: Dict[str, Any] | None = None):
    """
    异步流式调用 Agent，自动受 Semaphore 控制。
    返回异步生成器，每次 yield 一个 chunk。
    """
    async with AGENT_SEMAPHORE:
        async for chunk in agent.astream(
            {"messages": messages},
            config=config
        ):
            yield chunk

def run_async_stream(messages: list, config: Dict[str, Any] | None = None):
    """
    同步调用异步流式 Agent，并逐块返回结果。
    """
    async def _run():
        result_chunks = []
        async for chunk in async_stream_agent(messages, config):
            result_chunks.append(chunk)
        return result_chunks

    # 在事件循环中执行异步任务，并返回结果列表
    return asyncio.run(_run())

def save_feedback(question, answer, feedback_type, comment=""):
    st.write(f"DEBUG: 正在写入反馈，类型={feedback_type}")  # 临时调试
    """保存反馈到 Redis List"""
    feedback_data = json.dumps({
        "question": question,
        "answer": answer,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    }, ensure_ascii=False)
    feedback_redis.lpush("feedback:list", feedback_data)

# ============================================================
# 侧边栏配置
# ============================================================
with st.sidebar:
    st.title("📝 导游考试 AI 助手")
    st.markdown("---")

    # 当模式切换时，清除上一次的反馈状态
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = mode
    elif st.session_state.last_mode != mode:
        # 模式变了，重置反馈相关状态
        st.session_state.feedback_state = {}
        st.session_state.last_prompt = None
        st.session_state.last_answer = None
        st.session_state.last_msg_id = None
        st.session_state.last_mode = mode
        st.rerun()

    st.caption("技术栈：Python | LangChain | LangGraph | ChromaDB | Streamlit")
    st.caption("AI 模型：DeepSeek / 阿里云百炼")
    # 放在 with st.sidebar 块里
    # ... 你已有的侧边栏内容 ...

    st.markdown("---")
    st.subheader("📊 反馈统计")
    total_feedback = feedback_redis.llen("feedback:list")
    positive_count = sum(
        1 for fb in feedback_redis.lrange("feedback:list", 0, -1)
        if json.loads(fb).get("feedback") == "positive"
    )
    if total_feedback > 0:
        st.metric("总反馈数", total_feedback)
        st.metric("好评率", f"{positive_count / total_feedback * 100:.0f}%")
    else:
        st.caption("暂无反馈数据")

# ============================================================
# 示例问题（根据模式动态显示）
# ============================================================
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

# ============================================================
# 主界面
# ============================================================
st.title(mode)
st.markdown("---")

# 显示示例问题
st.markdown("#### 💡 试试这些问题：")
cols = st.columns(2)
for i, sample in enumerate(sample_questions.get(mode, [])):
    with cols[i % 2]:
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.current_prompt = sample
            st.rerun()

st.markdown("---")

# ============================================================
# 聊天输入
# ============================================================

def cached_qa_answer(query: str) -> str | None:
    """
    检查是否为完全相同的问题，如果是，返回缓存的最终答案。
    缓存键 = MD5(query)，过期时间 1 小时。
    """
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"qa_cache:{query_hash}"
    cached = feedback_redis.get(cache_key)
    if cached:
        return cached.decode()
    return None

def cache_qa_answer(query: str, answer: str, expire: int = 300):
    """缓存问答对"""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"qa_cache:{query_hash}"
    feedback_redis.setex(cache_key, expire, answer)

if prompt := st.chat_input("请输入你的问题，或点击上方的示例问题..."):
    sanitized, error_msg = sanitize_input(prompt)
    if error_msg:
        st.warning(error_msg)   # 显示拦截提示，不调用 Agent
        st.stop()
    prompt = sanitized          # 使用过滤后的输入
    st.session_state.current_prompt = prompt
def click_pos(msg_id):

    save_feedback(prompt, final_answer, "positive")
    st.session_state.feedback_state[msg_id] = "positive"
    st.rerun()

def click_neg(msg_id) :
    # 点踩后弹出评论框
    st.session_state.feedback_state[msg_id] = "pending_comment"
    st.rerun()

def click_comment(prompt, final_answer,comment, msg_id) :
    save_feedback(prompt, final_answer, "negative", comment)
    st.session_state.feedback_state[msg_id] = "done"
    st.rerun()

# 处理提示词（可能来自示例按钮或手动输入）
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    st.session_state.current_prompt = None  # 清空，避免重复发送

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # 工具调用展示
        tool_expander = st.expander("🔧 查看 Agent 思考过程", expanded=False)
        tool_records = []
        answer_placeholder = st.empty()

        thread_id ="guide_exam_memory_0004"

        # 判断是否需要添加系统消息（首次对话）
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = agent.get_state(config)
        except Exception:
            state = None

        if state is None or not state.values.get("messages"):
            messages = [SystemMessage(content=SYSTEM_PROMPT), ("user", prompt)]
        else:
            messages = [("user", prompt)]

        # 调用 Agent
        try:
            chunks = []  # 收集所有 chunk
            final_answer = ""
            for chunk in stream_agent_with_retry(agent,
                messages,
                config
            ):
                chunks.append(chunk)  # 保存下来，后续提取 Token
                if "tools" in chunk:
                    tool_msg = chunk["tools"]["messages"][0]
                    tool_records.append(
                        f"🛠️ **调用工具：{tool_msg.name}**\n\n"
                        f"输入参数：{tool_msg.content}"
                    )
                if "agent" in chunk:
                    final_answer += chunk["agent"]["messages"][0].content
                    answer_placeholder.markdown(final_answer)  # 实时更新显示

            # 流式结束后，提取 Token 并写入 Redis
            input_tok, output_tok = llm_service.extract_token_usage_from_stream(chunks)
            llm_service._record_usage(input_tok, output_tok)
            # 显示工具调用过程
            with tool_expander:
                if tool_records:
                    for rec in tool_records:
                        st.markdown(rec)
                        st.divider()
                else:
                    st.caption("本次未调用任何工具。")

            # 显示最终回答
            if final_answer:
                # ================= 反馈系统 =================
                # 用 prompts 的哈希值作为这条问答的唯一标识
                # ========== 修复关键：保存本次问答的状态 ==========
                st.session_state.last_prompt = prompt
                st.session_state.last_answer = final_answer
                st.session_state.last_msg_id = str(hash(prompt))
            else:
                st.warning("Agent 没有返回回答，请稍后重试。")

        except Exception as e:
            import traceback
            traceback.print_exc()
            st.error(f"Agent 调用失败：{str(e)[:300]}")

# ============================================================
# 连续对话状态提示
# ============================================================
st.sidebar.success("✅ 连续对话模式已开启，Agent 会记住上下文")

def render_feedback_section():
    # 1. 从 st.session_state 中获取上一次问答的状态
    prompt = st.session_state.get("last_prompt")
    final_answer = st.session_state.get("last_answer")
    msg_id = st.session_state.get("last_msg_id")

    # 2. 如果还没有发生过任何对话，什么也不渲染
    if not msg_id:
        return

    # 3. 初始化或读取当前消息的反馈状态
    if "feedback_state" not in st.session_state:
        st.session_state.feedback_state = {}

    fb_state = st.session_state.feedback_state.get(msg_id, None)

    # 4. 根据状态渲染不同的 UI
    if fb_state is None:
        # 状态：未反馈 -> 显示点赞/点踩按钮
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
        # 状态：点踩后待评论 -> 显示评论框
        comment = st.text_area("请告诉我们哪里回答得不好（可选）：", key=f"comment_{msg_id}")
        if st.button("提交反馈", key=f"submit_{msg_id}"):
            save_feedback(prompt, final_answer, "negative", comment)
            st.session_state.feedback_state[msg_id] = "done"
            st.rerun()

    else:
        # 状态：已反馈 -> 显示感谢语
        st.caption("✅ 感谢你的反馈！")

def generate_chunks(agent, messages, config):
    """将 agent.stream 包装成文本生成器"""
    for chunk in agent.stream({"messages": messages}, config=config):
        if "agent" in chunk:
            yield chunk["agent"]["messages"][0].content
        # 如果有工具调用，可以选择不输出，或者另外展示

render_feedback_section()
llm_service.sidebar_usage()



