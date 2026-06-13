from agent import agent

# 模拟用户输入（需要用工具的需求）
user_input = "帮我找两道导游业务里关于地陪的单选题"
# user_input = "你是谁啊"


# 调用 Agent（非流式，方便断点调试）
result = agent.invoke({"messages": [("user", user_input)]})
print("Agent result：\n", result)
# 打印最终回答
final_message = result["messages"][-1].content
print("Agent 回答：\n", final_message)