import time
import traceback
from datetime import datetime

import streamlit as st
from agent import agent
from langchain_core.messages import SystemMessage
import uuid
import os
import logging


os.environ["STREAMLIT_SERVER_ENABLE_STATS"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
st.set_page_config(page_title="导游考试AI助手", page_icon="📝")
logging.basicConfig(level=logging.INFO)

# 系统提示词
SYSTEM_PROMPT = """
你是一个导游考试智能助手。你拥有以下工具：
- search_textbook: 从教材中检索任何内容（知识点、目录、习题等）。
- grade_answer: 批改学员的答案。

重要规则：
1. 当用户询问任何与教材相关的内容时，**必须**先调用 search_textbook 工具获取真实内容，不得凭记忆编造。
2. 如果需要批改，调用 grade_answer。
4. 严禁在未调用工具的情况下直接回答与教材或题库有关的问题。
"""
state_modifier = SYSTEM_PROMPT  # 系统提示在这里
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "memory_mode"  # 固定 ID，启用记忆

if prompt := st.chat_input("请输入你的问题或需求..."):
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        tool_expander = st.expander("🔧 查看 Agent 思考过程", expanded=False)
        tool_records = []

        # 重要：每次调用使用全新的 thread_id，避免旧消息干扰
        # 调用前
        start = time.time()
        logging.info(f"开始调用 Agent，当前线程: {st.session_state.thread_id}")
        try:
            final_answer = ""
            for chunk in agent.stream(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        ("user", prompt)
                    ]
                },
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            ):
                if "tools" in chunk:
                    tool_msg = chunk["tools"]["messages"][0]
                    tool_records.append(f"🛠️ 调用工具: **{tool_msg.name}**\n输入: {tool_msg.content}")
                if "agent" in chunk:
                    final_answer = chunk["agent"]["messages"][0].content

            with tool_expander:
                if tool_records:
                    for rec in tool_records:
                        st.markdown(rec)
                        st.divider()
                else:
                    st.caption("本次未调用任何工具。")

            if final_answer:
                st.write(final_answer)
            else:
                st.warning("Agent 没有返回回答，请检查控制台日志。")


        except Exception as e:
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(f"\n{datetime.now()}\n")
                traceback.print_exc(file=f)
            # 尝试优雅降级
            st.error("系统繁忙，请稍后再试。")
        # 调用后
        duration = time.time() - start
        logging.info(f"Agent 响应耗时: {duration:.2f}秒")
