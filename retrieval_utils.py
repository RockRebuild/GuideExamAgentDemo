import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter

sentence_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20,
    separators=["\n", "。", "！", "？", "；", " ", ""]
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "bge_reranker_cache", "BAAI", "bge-reranker-base")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
_model.eval()

def compute_scores_batch(query: str, passages: list[str]) -> list[float]:
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

def refine_and_rerank(raw_contexts: list[str], question: str, top_k: int = 5) -> list[str]:
    # 1. 从问题中提取核心关键词（用于预筛）
    keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', question)
    # 2. 原始段落去重，并过滤掉明显无意义的标记行（如“【片段X】”）
    clean_paras = []
    for p in raw_contexts:
        # 去除纯标记行和过短碎片
        if re.match(r'^【片段\d+】$', p.strip()) or len(p) < 10:
            continue
        if p not in clean_paras:
            clean_paras.append(p)

    # 3. 拆句，并只保留包含至少一个关键词或长度足够长的句子
    all_sentences = []
    for ctx in clean_paras:
        for chunk in sentence_splitter.split_text(ctx):
            cleaned = chunk.strip()
            if len(cleaned) >= 15:
                # 必须包含问题中的至少一个中文关键词，或者本身很长（可能为定义句）
                if any(kw in cleaned for kw in keywords) or len(cleaned) > 50:
                    all_sentences.append(cleaned)

    # 4. 去重
    seen = set()
    unique_sentences = []
    for s in all_sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    # 5. BGE-Reranker 精排，取 top_k
    return rerank_contexts(question, unique_sentences, top_k)