#!/usr/bin/env python3
# scripts/enrich_question_bank.py
# ── 题库搜索字段增强 ──
# 为每条题目添加 search_key 字段，拼接教材别名 + subject + chapter，
# 使 search_questions 工具可以用简单的 substring 匹配找到题目。
#
# 原理:
#   题库原始结构: {"subject": "科目一", "chapter": "第八章　合同法律制度"}
#   用户输入:       "政策与法律法规 合同法律制度 出5道多选题"
#   问题: "政策与法律法规" 不在 subject 也不在 chapter 中
#
#   解决方案: 为每条题目添加 search_key 字段，包含所有可能的搜索词:
#     "科目一 政策与法律法规 第八章 合同法律制度"
#
# 运行: python scripts/enrich_question_bank.py

import json
import os
import sys

# 教材名映射（subject 编号 → 常用教材名）
SUBJECT_NAMES = {
    "科目一": ["政策与法律法规", "政策法规", "法律法规"],
    "科目二": ["导游业务"],
    "科目三": ["全国导游基础知识", "全国导基", "导游基础知识"],
    "科目四": ["地方导游基础知识", "地方导基"],
}

# 科目编号也是有效的搜索词
SUBJECT_CODES = ["科目一", "科目二", "科目三", "科目四"]

# 章节名中的"第X章"提取 → 同时保留数字形式和汉字形式
import re

CHAPTER_NUM_MAP = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
    "十一": "11", "十二": "12", "十三": "13", "十四": "14", "十五": "15",
    "十六": "16", "十七": "17", "十八": "18", "十九": "19", "二十": "20",
}


def build_search_key(subject: str, chapter: str) -> str:
    """构建搜索键：教材名 + 科目编号 + 章节名 + 章节号。

    例如 subject="科目一", chapter="第八章　合同法律制度"
      → "科目一 政策与法律法规 政策法规 法律法规 第八章 第8章 合同法律制度"
    """
    parts = []

    # 1. 科目编号
    parts.append(subject)

    # 2. 教材别名（从科目编号查）
    for code, names in SUBJECT_NAMES.items():
        if code in subject:
            parts.extend(names)
            break

    # 3. 章节号（汉字 + 数字）
    ch_match = re.search(r'第([一二三四五六七八九十\d]+)[章节篇]', chapter)
    if ch_match:
        ch_num = ch_match.group(1)
        parts.append(ch_match.group(0))  # 如"第八章"
        if ch_num in CHAPTER_NUM_MAP:
            parts.append(f"第{CHAPTER_NUM_MAP[ch_num]}章")
        elif ch_num.isdigit():
            parts.append(f"第{ch_num}章")

    # 4. 章节标题（去除"第X章"后的内容）
    chapter_title = re.sub(r'第[一二三四五六七八九十\d]+[章节篇]\s*', '', chapter).strip()
    if chapter_title:
        parts.append(chapter_title)

    # 5. 去重 + 拼接
    seen = set()
    unique = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)

    return " ".join(unique)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qb_path = os.path.join(base_dir, "question_bank.json")

    print(f"Reading: {qb_path}")
    with open(qb_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Total questions: {len(questions)}")

    # 添加 search_key
    unique_keys = set()
    for q in questions:
        key = build_search_key(q.get("subject", ""), q.get("chapter", ""))
        q["search_key"] = key
        unique_keys.add(key)

    # 备份
    backup_path = qb_path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in q.items() if k != "search_key"} for q in questions], f, ensure_ascii=False, indent=2)
        print(f"Backup saved: {backup_path}")

    # 写入
    with open(qb_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Done. Added search_key to {len(questions)} questions.")
    print(f"Unique search_keys: {len(unique_keys)}")
    print(f"\nSample search_keys:")
    for k in sorted(unique_keys)[:10]:
        print(f"  {k}")


if __name__ == "__main__":
    main()
