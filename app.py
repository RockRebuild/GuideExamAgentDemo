import streamlit as st
from agent import agent
from langchain_core.messages import SystemMessage
import uuid
import warnings

# 屏蔽无关警告，保持界面干净
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*NoSessionContext.*")

st.set_page_config(
    page_title="导游考试 AI 助手",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    st.markdown("---")

    # 记忆开关
    use_memory = st.checkbox("🧠 开启连续对话", value=False, help="开启后 Agent 能记住上下文")

    st.markdown("---")
    st.caption("技术栈：Python | LangChain | LangGraph | ChromaDB | Streamlit")
    st.caption("AI 模型：DeepSeek / 阿里云百炼")

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

# 处理提示词（可能来自示例按钮或手动输入）
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    st.session_state.current_prompt = None  # 清空，避免重复发送

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # 工具调用展示
        tool_expander = st.expander("🔧 查看 Agent 思考过程", expanded=False)
        tool_records = []

        # 决定 thread_id
        if use_memory:
            if "memory_thread_id" not in st.session_state:
                st.session_state.memory_thread_id = f"memory_{uuid.uuid4().hex[:8]}"
            thread_id = st.session_state.memory_thread_id
        else:
            thread_id = str(uuid.uuid4())

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
            else:
                st.warning("Agent 没有返回回答，请稍后重试。")

        except Exception as e:
            import traceback
            traceback.print_exc()
            st.error(f"Agent 调用失败：{str(e)[:300]}")

# ============================================================
# 连续对话状态提示
# ============================================================
if use_memory:
    st.sidebar.success("✅ 连续对话模式已开启，Agent 会记住上下文")
else:
    st.sidebar.info("ℹ️ 独立问答模式，每次提问都是新的开始")