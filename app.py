import random
import secrets

import redis
import streamlit as st
from agent import agent
from langchain_core.messages import SystemMessage
import uuid
import warnings
import json
from datetime import datetime
from functools import partial



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

# 复用已有的 Redis 连接（本地默认端口）
feedback_redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

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
# 系统提示词
# ============================================================
SYSTEM_PROMPT = """
你是一个导游考试智能助手，可以帮助用户完成以下任务：
- 查询教材知识点（使用 search_textbook 工具）
- 检索考试题目（使用 search_questions 工具）
- 批改学员答案（使用 grade_answer 工具）

重要规则：
1. 当用户询问任何与教材相关的内容时，必须调用 search_textbook 获取真实内容，严禁自己编造。
2. 当用户要求出题、找题目时，必须调用 search_questions 获取真实题目。
3. 当用户要求批改题目时，必须调用 grade_answer。
4. 批改完如果学员答错，应主动调用 search_textbook 帮学员复习相关知识点。
"""

# ============================================================
# 侧边栏配置
# ============================================================
with st.sidebar:
    st.title("📝 导游考试 AI 助手")
    st.markdown("---")

    # 模式选择
    mode = st.radio(
        "选择模式",
        ["📖 教材知识问答", "📝 智能出卷", "📊 阅卷批改"],
        index=0
    )

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
        "地陪导游在接团前需要准备哪些证件？",
        "全陪导游的职责是什么？",
        "《旅游法》规定了旅游者的哪些权利？",
        "导游证的种类有哪些？"
    ],
    "📝 智能出卷": [
        "帮我出两道导游业务第三章的单选题目",
        "帮我找三道关于旅游法的多选题",
        "出五道判断题，范围是政策法规",
    ],
    "📊 阅卷批改": [
        "请批改题目 q_001，我的答案是 B",
        "帮我批改题目 q_003，我选 A",
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
if prompt := st.chat_input("请输入你的问题，或点击上方的示例问题..."):
    st.session_state.current_prompt = prompt
def click_pos():

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

        thread_id ="guide_exam_memory_0001"

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
            final_answer = ""
            for chunk in agent.stream(
                {"messages": messages},
                config=config
            ):
                if "tools" in chunk:
                    tool_msg = chunk["tools"]["messages"][0]
                    tool_records.append(
                        f"🛠️ **调用工具：{tool_msg.name}**\n\n"
                        f"输入参数：{tool_msg.content}"
                    )
                if "agent" in chunk:
                    final_answer = chunk["agent"]["messages"][0].content

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
                st.write(final_answer)
                # ================= 反馈系统 =================
                # 用 prompt 的哈希值作为这条问答的唯一标识
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

render_feedback_section()

