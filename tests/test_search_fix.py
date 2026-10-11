#!/usr/bin/env python3
"""Test that search_key matching works end-to-end for all expected inputs."""
import json, re, os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

with open(os.path.join(_project_root, 'question_bank.json'), 'r', encoding='utf-8') as f:
    questions = json.load(f)

tests = [
    ('政策与法律法规 合同法律制度', '多选题', 'exact chapter'),
    ('导游业务 第三章', '单选题', 'subject+chapter'),
    ('全国导游基础知识 中国饮食文化', '判断题', 'full subject name'),
    ('科目一 第一章', '单选题', '科目编号'),
    ('全国导基 中国历史文化', '单选题', 'alias'),
]

def search(chapter, qtype):
    keywords = chapter.split()
    result = []
    for q in questions:
        sk = q.get('search_key', '')
        if not sk:
            sk = q['subject'] + ' ' + q['chapter']
        if all(kw in sk for kw in keywords):
            result.append(q)
    if qtype != '全部':
        result = [q for q in result if q['type'] == qtype]
    return result

all_ok = True
for chapter, qtype, label in tests:
    r = search(chapter, qtype)
    ok = len(r) > 0
    if not ok:
        all_ok = False
        # Debug: check each keyword
        for kw in chapter.split():
            found = any(kw in q.get('search_key', '') for q in questions)
            if not found:
                print(f'FAIL: keyword {kw!r} not found in any search_key')
    status = 'OK' if ok else 'FAIL'
    print(f'[{status}] {label}: {chapter!r} + {qtype!r} -> {len(r)} results')

if all_ok:
    print('\nAll PASS')
else:
    print('\nSOME FAILED')
    sys.exit(1)
