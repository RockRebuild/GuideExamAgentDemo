import os
import re
from difflib import SequenceMatcher

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter

sentence_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20,
    separators=["\n", "。", "！", "？", "；", " ", ""]
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "bge_reranker_cache", "BAAI", "bge-reranker-base")
_tokenizer = None
_model = None


def _ensure_model_loaded():
    """懒加载 BGE-Reranker 模型，避免模块导入时因模型文件缺失而崩溃"""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()


def compute_scores_batch(query: str, passages: list[str]) -> list[float]:
    _ensure_model_loaded()
    inputs = _tokenizer([[query, p] for p in passages], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
    return torch.sigmoid(logits).flatten().tolist()

def rerank_contexts(query: str, contexts: list[str], top_k: int = 5) -> list[str]:
    if not contexts:
        return []
    valid = [ctx for ctx in contexts if len(ctx) >= 15]
    if not valid:
        return contexts[:top_k]
    scores = compute_scores_batch(query, valid)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [valid[i] for i in sorted_indices[:top_k]]


def refine_and_rerank(raw_contexts: list[str], question: str, top_k: int = 5,
                       min_score: float = 0.5) -> list[str]:
    """
    对原始段落做精排，返回 top-K **完整段落**（而非句子碎片）。

    策略：拆句 → BGE-Reranker 给每句话打分 → 每个段落取最佳句子分作为段落分
    → 过滤低分段落 → 按段落分排序，返回完整段落。

    min_score: BGE-Reranker 最低相关性阈值（0~1）。低于此分数的段落被视为无关噪音，
               直接丢弃，不参与 top-K 竞争。这是 ContextPrecision 的关键保障。
    """
    print(f"🔥 refine_and_rerank 被调用, contexts数量={len(raw_contexts)}, query={question[:50]}", flush=True)

    # 1. 判断是否属于"列举/概述/介绍"类问题
    listing_keywords = ["有哪些", "包括什么", "主要景点", "主要文旅资源", "列举", "哪些",
                        "什么景区", "介绍", "概况", "概述", "基本情况", "主要资源"]
    is_listing_question = any(kw in question for kw in listing_keywords)

    # 2. 从问题中提取核心关键词（用于非列举类问题的预筛）
    keywords = re.findall(r'[一-龥]{2,}', question)

    # 3. 原始段落去重，过滤标记行和过短文本
    clean_paras = []
    for p in raw_contexts:
        # 过滤纯标记行（如单独的"【片段1】"），但保留带有来源信息的行
        if re.match(r'^【片段\d+】$', p.strip()):
            continue
        if len(p) < 10:
            continue
        if p not in clean_paras:
            clean_paras.append(p)

    if len(clean_paras) <= top_k:
        # 段落数不超过 top_k，直接返回全部（保留完整上下文）
        return clean_paras

    # 4. 拆句，记录每句话属于哪个段落 → sent_to_para[global_idx] = para_idx
    all_sentences_flat: list[str] = []
    sent_to_para: list[int] = []

    def keep_sentence(text: str) -> bool:
        # 始终保留来源标注行（如 【来源：第一章，第1页】、【片段1】）
        if re.match(r'^【(?:来源|片段)', text):
            return True
        if len(text) < 15:
            return False
        if is_listing_question:
            return True
        return any(kw in text for kw in keywords) or len(text) > 50

    for pi, para in enumerate(clean_paras):
        for chunk in sentence_splitter.split_text(para):
            cleaned = chunk.strip()
            if keep_sentence(cleaned):
                all_sentences_flat.append(cleaned)
                sent_to_para.append(pi)

    if not all_sentences_flat:
        return clean_paras[:top_k]

    # 5. BGE-Reranker 给所有句子打分
    sentence_scores = compute_scores_batch(question, all_sentences_flat)

    # 6. 每句话的分数映射回所属段落，段落分 = 段落内最佳句子分
    para_best_score: list[float] = [0.0] * len(clean_paras)
    for global_idx, score in enumerate(sentence_scores):
        pi = sent_to_para[global_idx]
        if score > para_best_score[pi]:
            para_best_score[pi] = score

    # 7. 过滤低分段落（噪音段落会直接拖低 ContextPrecision）
    #    段落分 < min_score 的视为无关，丢弃
    qualified = [i for i in range(len(clean_paras)) if para_best_score[i] >= min_score]
    if not qualified:
        # 兜底：所有段落都低于阈值时，至少保留最高分的 1 个
        qualified = [max(range(len(clean_paras)), key=lambda i: para_best_score[i])]
    qualified.sort(key=lambda i: para_best_score[i], reverse=True)

    print(f"🔥 Reranker: {len(clean_paras)} 段落 → {len(qualified)} 段超过阈值 {min_score}",
          flush=True)

    # 8. 按段落分排序 + 相似度去重，返回完整段落
    final_paras = []
    for idx in qualified:
        para = clean_paras[idx]
        is_dup = False
        for existing in final_paras:
            if SequenceMatcher(None, para, existing).ratio() > 0.85:
                is_dup = True
                break
        if not is_dup:
            final_paras.append(para)
        if len(final_paras) >= top_k:
            break

    return final_paras


def determine_top_k(question: str) -> int:
    """
    根据问题类型和信息需求，动态决定重排序后的 top_k 值。
    """
    question_lower = question.lower()
    q_len = len(question)

    # 1. 列举/枚举类
    listing_keywords = ["有哪些", "包括什么", "主要景点", "主要文旅资源", "列举", "哪些", "什么景区"]
    if any(kw in question_lower for kw in listing_keywords):
        return 15

    # 2. 对比/比较类
    comparison_patterns = [
        r"区别", r"异同", r"对比", r"比较",
        r"和.*什么不同", r"与.*什么区别"
    ]
    if any(re.search(pat, question_lower) for pat in comparison_patterns):
        return 12

    # 3. 流程/步骤类
    process_keywords = ["流程", "步骤", "怎么做", "如何", "怎样", "方法", "程序"]
    if any(kw in question_lower for kw in process_keywords):
        return 10

    # 4. 综合/概述类：问题较长且包含介绍性词汇
    overview_keywords = ["介绍", "概况", "概述", "基本情况", "主要资源"]
    if q_len > 15 and any(kw in question_lower for kw in overview_keywords):
        return 12

    # 5. 推理/解释类
    reasoning_keywords = ["为什么", "原因", "理由", "怎么解释"]
    if any(kw in question_lower for kw in reasoning_keywords):
        return 12

    # 6. 短事实查询
    if q_len < 15:
        return 3

    # 7. 默认
    return 5
