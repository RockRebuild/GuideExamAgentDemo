# tests/test_prompt.py

import pytest
from agent import agent  # 你的 Agent 实例

# 测试集：每条包含问题、期望调用的工具名（如果期望不调用工具，填 None）
TEST_CASES = [
    # 知识点检索（期望调用 search_textbook）
    ("地陪导游接团前需要准备哪些证件？", "search_textbook"),
    ("全陪导游的职责是什么？", "search_textbook"),
    ("《旅游法》第35条是什么？", "search_textbook"),
    ("全国导游基础知识第二章目录", "search_textbook"),
    ("导游证的种类有哪些？", "search_textbook"),

    # 出卷请求（期望调用 search_questions）
    ("帮我出两道导游业务第三章的单选题目", "search_questions"),
    ("给我找三道关于旅游法的多选题", "search_questions"),
    ("出五道判断题，范围是政策法规", "search_questions"),
    "",
    "帮我批改题目 科目四的第十章多选第三题，我选 A",
    # 批改请求（期望调用 grade_answer）
    ("请批改题目 科目一的第一章单选第一题，我的答案是 A", "grade_answer"),
    ("帮我批改题目 科目四的第十章多选第三题，我选 A，B", "grade_answer"),

    # 边界场景：模糊问题、无关问题
    ("你好", None),  # 简单问候，不应调用工具
    ("今天天气怎么样？", None),  # 无关问题，应拒绝或引导
    ("给我讲个笑话", None),  # 无关问题
    ("帮我写一个爬虫脚本", None),  # 无关问题
]


@pytest.mark.parametrize("query, expected_tool", TEST_CASES)
def test_tool_calling(query, expected_tool):
    """验证 Agent 在特定问题下是否调用了正确的工具"""
    response = agent.invoke({"messages": [("user", query)]})

    # 从响应中提取实际调用的工具名
    actual_tools = []
    for msg in response.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                actual_tools.append(tc["name"])

    if expected_tool is None:
        # 期望不调用任何工具
        assert len(actual_tools) == 0, f"问题 '{query}' 不应调用工具，但调用了 {actual_tools}"
    else:
        assert expected_tool in actual_tools, f"问题 '{query}' 应调用 {expected_tool}，但未找到。实际工具: {actual_tools}"