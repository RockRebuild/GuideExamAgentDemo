from agent import agent

from tools import initialize_vectorstore   # 或直接重新创建 vectorstore 对象

vectorstore = initialize_vectorstore()

print("向量库文档数量:", vectorstore._collection.count())

def test_rag():
    # 问题1：纯知识询问（应触发 search_textbook）
    print("=" * 50)
    print("测试1: 地陪导游接团时需要准备哪些证件？")
    result = agent.invoke({"messages": [("user", "地陪导游接团时需要准备哪些证件？")]})
    print("Agent回答:", result["messages"][-1].content)

    # 问题2：出卷需求（应触发 search_questions）
    print("\n" + "=" * 50)
    print("测试2: 帮我找两道导游业务第三章的单选题目")
    result = agent.invoke({"messages": [("user", "帮我找两道导游业务第三章的单选题目")]})
    print("Agent回答:", result["messages"][-1].content)

if __name__ == "__main__":
    test_rag()