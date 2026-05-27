import re
import json
from pypdf import PdfReader

# 1. 从 PDF 提取文本
reader = PdfReader("题库2.pdf")
lines = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        # 按行拆分，保留空行用于判断段落边界
        lines.extend(text.split('\n'))

# 2. 状态机解析
questions = []
current = None  # 当前正在构建的题目
state = "idle"  # idle / in_question / in_options / in_explanation
option_buffer = []
question_buffer = []

def save_current():
    """保存当前题目并重置"""
    global current, state, option_buffer, question_buffer
    if current and current.get("question") and current.get("options"):
        questions.append(current)
    current = None
    state = "idle"
    option_buffer = []
    question_buffer = []

for line in lines:
    line = line.strip()
    if not line:
        continue

    # 跳过目录、出版信息等非题目内容
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩\d]+[、.．]', line) and '模拟试题' in line:
        continue
    if '出版说明' in line or 'ISBN' in line or 'Digitized by' in line:
        continue
    if re.match(r'^第[一二三四五六七八九十\d]+章', line) and '参考答案' not in line:
        continue

    # 检测题目开始：数字 + . + ［题型］
    q_start = re.match(r'^(\d+)\.\s*[\[［]([^\]］]+)[\]］]\s*(.*)', line)
    if q_start:
        save_current()
        q_number = q_start.group(1)
        q_type = q_start.group(2)
        # 统一题型
        if '单选' in q_type: q_type = '单选'
        elif '多选' in q_type: q_type = '多选'
        elif '判断' in q_type: q_type = '判断'
        rest = q_start.group(3)
        current = {
            "id": f"q_{int(q_number):03d}",
            "type": q_type,
            "chapter": "待整理",
            "question": rest,
            "options": [],
            "answer": "",
            "explanation": ""
        }
        question_buffer = [rest]
        state = "in_question"
        continue

    # 如果还没开始任何题目，跳过
    if current is None:
        continue

    # 检测选项开始：以 A. 或 A． 开头
    if re.match(r'^[A-D][.．]', line):
        state = "in_options"
        option_buffer.append(line)
        # 如果一行包含多个选项（如 "A.xxx　　B.xxx"），拆分
        if re.search(r'[A-D][.．].*[A-D][.．]', line):
            parts = re.split(r'(?=[A-D][.．])', line)
            option_buffer = [p.strip() for p in parts if p.strip()]
        continue

    # 检测解析开始
    if line.startswith('【解析】') or line.startswith('【答案】') or line.startswith('答案'):
        state = "in_explanation"
        # 保存之前收集的选项
        if option_buffer:
            current["options"] = option_buffer[:4]  # 只取前4个选项
            option_buffer = []
        # 如果是解析行
        if '【解析】' in line:
            current["explanation"] = line.replace('【解析】', '').strip()
        # 如果是答案行
        if '答案' in line:
            ans = re.search(r'答案[：:]\s*([A-D对错√×]+)', line)
            if ans:
                current["answer"] = ans.group(1)
        continue

    # 根据状态处理
    if state == "in_question":
        # 题干可能跨多行
        question_buffer.append(line)
        current["question"] = ' '.join(question_buffer)

    elif state == "in_options":
        # 继续收集选项
        option_buffer.append(line)

    elif state == "in_explanation":
        # 解析可能跨多行
        if '答案' in line:
            ans = re.search(r'答案[：:]\s*([A-D对错√×]+)', line)
            if ans:
                current["answer"] = ans.group(1)
        else:
            current["explanation"] += line

# 保存最后一道题
save_current()

# 3. 后处理：清理选项格式
for q in questions:
    clean_options = []
    for opt in q["options"]:
        opt = opt.strip()
        # 去掉多余全角空格
        opt = re.sub(r'[\u3000]+', ' ', opt)
        # 确保选项以字母 + 点号开头
        if not re.match(r'^[A-D][.．]', opt):
            opt = f"{chr(65+len(clean_options))}. {opt}"
        clean_options.append(opt)
    q["options"] = clean_options[:4]  # 只保留 A-D

# 4. 写入 JSON
with open("question_bank2.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"解析完成！共提取 {len(questions)} 道题。")
print(f"单选: {sum(1 for q in questions if q['type']=='单选')}")
print(f"多选: {sum(1 for q in questions if q['type']=='多选')}")
print(f"判断: {sum(1 for q in questions if q['type']=='判断')}")