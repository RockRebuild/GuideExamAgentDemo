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

def merge_wrapped_chapter_titles(lines):
    """合并跨行的章标题，更稳健的版本"""
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 检测章标题开始（第X章、第XX章、第X篇、第X部分等）
        if re.match(r'^第[一二三四五六七八九十\d]+(章|篇|部分)', line):
            combined = line
            i += 1
            # 持续合并后续非空行，直到遇到明显的非标题内容
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:  # 跳过空行
                    i += 1
                    continue
                # 终止条件：题目开始、选项、解析、答案、新章节、明显不是标题的内容（如数字开头）
                if (re.match(r'^\d+\.', next_line) or
                    re.match(r'^[A-D][.．]', next_line) or
                    re.match(r'^第[一二三四五六七八九十\d]+(章|篇|部分)', next_line) or
                    next_line.startswith('【解析】') or
                    next_line.startswith('答案') or
                    next_line.startswith('参考答案') or
                    re.match(r'^\d+', next_line)):
                    break
                # 如果下一行看起来像标题的续行（不是题目），就合并
                combined += ' ' + next_line
                i += 1
                # 如果合并后已经比较长，或者以标点结束，可提前停止
                if len(combined) > 40 or re.search(r'[。！？]$', combined):
                    break
            merged.append(combined)
        else:
            merged.append(line)
            i += 1
    return merged

# 应用预处理
lines = merge_wrapped_chapter_titles(lines)

# 2. 状态机解析
class QuestionParser:
    def __init__(self):
        self.current = None
        self.state = "idle"
        self.option_buffer = []
        self.question_buffer = []
        self.chapter_type_counter = {}
        self.current_chapter = "未知章节"
        self.current_subject = "未知科目"   # 新增：当前科目
        self.questions = []

    def save_current(self):
        if self.current and self.current.get("question") and self.current.get("options"):
            subject = self.current.get("subject", "未知科目")
            chapter = self.current.get("chapter", "未知章节")
            qtype = self.current.get("type", "未知题型")
            key = (subject, chapter, qtype)
            self.chapter_type_counter[key] = self.chapter_type_counter.get(key, 0) + 1
            index = self.chapter_type_counter[key]
            # 生成唯一ID
            self.current["id"] = f"{subject}_{chapter}_{qtype}_{index}"
            self.questions.append(self.current)
        self.current = None
        self.state = "idle"
        self.option_buffer = []
        self.question_buffer = []

parser = QuestionParser()

for line in lines:
    line = line.strip()
    if not line:
        continue

    # 跳过目录、出版信息等非题目内容
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩\d]+[、.．]', line) and '模拟试题' in line:
        continue
    if '出版说明' in line or 'ISBN' in line or 'Digitized by' in line:
        continue
    # 检测科目标题，例如 "科目一 政策法规" 或 "科目1"
    subj_match = re.match(r'^科目[一二三四五六七八九十\d]+', line)
    if subj_match:
        parser.current_subject = line.strip()
        continue
    if re.match(r'^第[一二三四五六七八九十\d]+章', line) and '参考答案' not in line:
        parser.current_chapter = line.strip()  # 记录当前章节名
        continue

    # 检测题目开始：数字 + . + ［题型］
    q_start = re.match(r'^(\d+)\.\s*[\[［]([^\]］]+)[\]］]\s*(.*)', line)
    if q_start:
        parser.save_current()
        q_type = q_start.group(2)
        # 统一题型
        if '单选' in q_type: q_type = '单选'
        elif '多选' in q_type: q_type = '多选'
        elif '判断' in q_type: q_type = '判断'
        rest = q_start.group(3)
        parser.current = {
            "id": "",
            "type": q_type,
            "subject": parser.current_subject,   # 新增科目
            "chapter": parser.current_chapter,
            "question": rest,
            "options": [],
            "answer": "",
            "explanation": ""
        }
        parser.question_buffer = [rest]
        parser.state = "in_question"
        continue

    # 如果还没开始任何题目，跳过
    if parser.current is None:
        continue

    # 检测选项开始：以 A. 或 A． 开头
    if re.match(r'^[A-D][.．]', line):
        state = "in_options"
        if re.search(r'[A-D][.．].*[A-D][.．]', line):
            # 一行包含多个选项，拆分后逐个追加
            parts = re.split(r'(?=[A-D][.．])', line)
            for p in parts:
                p = p.strip()
                if p and re.match(r'^[A-D][.．]', p):
                    parser.option_buffer.append(p)
        else:
            # 单个选项直接追加
            parser.option_buffer.append(line)
        continue

    # 检测解析开始
    if line.startswith('【解析】') or line.startswith('【答案】') or line.startswith('答案'):
        parser.state = "in_explanation"
        # 保存之前收集的选项
        if parser.option_buffer:
            parser.current["options"] = parser.option_buffer[:4]  # 只取前4个选项
            parser.option_buffer = []
        # 如果是解析行
        if '【解析】' in line:
            parser.current["explanation"] = line.replace('【解析】', '').strip()
        # 如果是答案行
        if '答案' in line:
            ans = re.search(r'答案[：:]\s*([A-D对错√×]+)', line)
            if ans:
                parser.current["answer"] = ans.group(1)
        continue

    # 根据状态处理
    if parser.state == "in_question":
        # 题干可能跨多行
        parser.question_buffer.append(line)
        parser.current["question"] = ' '.join(parser.question_buffer)

    elif parser.state == "in_options":
        # 继续收集选项
        parser.option_buffer.append(line)

    elif parser.state == "in_explanation":
        # 解析可能跨多行
        if '答案' in line:
            ans = re.search(r'答案[：:]\s*([A-D对错√×]+)', line)
            if ans:
                parser.current["answer"] = ans.group(1)
        else:
            parser.current["explanation"] += line

# 循环结束后，保存最后一道题
parser.save_current()
questions = parser.questions

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