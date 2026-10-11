#!/usr/bin/env python3
# scripts/clean_question_bank.py
# 修复题库数据质量问题：
# 1. 题目文本末尾混入选项片段（如 "。E.其他组织"）
# 2. 选项内容混入多余文字

import json, re, os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(base, "question_bank.json")

with open(path, "r", encoding="utf-8") as f:
    questions = json.load(f)

fixed = 0
for q in questions:
    question = q["question"]
    options = q.get("options", [])

    # 1. 清理 question 末尾混入的选项标签: "A.xxx" "B.xxx" 等
    #    模式: 中文末尾跟大写字母+中英文标点
    question = re.sub(r'[。；，]\s*[A-E][.、][^。；，]*$', '', question)
    #    模式: 末尾有中文 + 字母选项开头
    question = re.sub(r'[一-鿿]\s*[A-E][.、][^A-E\n]*$', '', question)

    # 2. 清理每个选项末尾多余的 "的E." 之类片段
    clean_opts = []
    for opt in options:
        # 去掉选项末尾可能跟的下一个选项标记片段
        opt = re.sub(r'[.、]\s*[A-E][.、].*$', '', opt)
        clean_opts.append(opt.strip())

    if question != q["question"] or clean_opts != options:
        fixed += 1
        q["question"] = question.strip()
        q["options"] = clean_opts

# Backup
bak = path + ".bak2"
with open(bak, "w", encoding="utf-8") as f:
    json.dump([{k: v for k, v in q.items()} for q in questions], f, ensure_ascii=False, indent=2)

with open(path, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"Fixed {fixed} of {len(questions)} questions. Backup: {bak}")
