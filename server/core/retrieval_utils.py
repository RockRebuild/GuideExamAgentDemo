import os
import re
import gc
import sys
from difflib import SequenceMatcher

# ECS 小内存实例上 torch 默认会用所有 CPU 核心 + 大 batch 导致 OOM
# 这些环境变量必须在 import torch 之前设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter

sentence_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20,
    separators=["\n", "。", "！", "？", "；", " ", ""]
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bge_reranker_cache", "BAAI", "bge-reranker-base")
MODEL_ID = "BAAI/bge-reranker-base"  # HuggingFace fallback
_tokenizer = None
_model = None
_reranker_disabled = False  # 运行时标志，OOM 后设为 True，当前进程内不再尝试

# 持久化禁用标记：如果上一次进程因 OOM 被 kill，重启后不再尝试加载模型
_RERANKER_DISABLED_FILE = "/tmp/bge_reranker_disabled"

# 安全内存阈值（字节）：可用内存低于此值时不加载 BGE-Reranker，避免 OOM
# BGE-Reranker-base 模型约 1.1GB，加上推理开销，保守估计需要 1.5GB 可用内存
MIN_FREE_MEMORY_FOR_RERANKER = int(os.environ.get("RERANKER_MIN_FREE_MEMORY", str(1500 * 1024 * 1024)))


def _is_reranker_permanently_disabled() -> bool:
    """检查持久化标记文件，判断 reranker 是否已被永久禁用。"""
    return os.path.exists(_RERANKER_DISABLED_FILE)


def _disable_reranker_permanently():
    """写入持久化标记文件，重启后也不再尝试加载模型。"""
    try:
        with open(_RERANKER_DISABLED_FILE, "w") as f:
            f.write("disabled due to OOM")
    except Exception:
        pass


def _get_available_memory() -> int:
    """获取当前系统可用内存（字节）。仅在 Linux 下有效，其他平台返回一个大值。"""
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # 格式: "MemAvailable:    1234567 kB"
                        return int(line.split()[1]) * 1024
        # 非 Linux 平台不做限制
        return sys.maxsize
    except Exception:
        return sys.maxsize


def _ensure_model_loaded():
    """懒加载 BGE-Reranker 模型。

    优先从本地 MODEL_PATH 加载，如果本地缓存文件不完整（例如缺少 tokenizer 文件），
    自动从 HuggingFace Hub 下载缺失的文件（通过 HF_ENDPOINT 环境变量，默认 hf-mirror.com）。

    如果可用内存不足或加载/推理过程中发生 OOM，设置 _reranker_disabled = True 并抛出异常，
    由上层 refine_and_rerank 降级处理。
    """
    global _tokenizer, _model, _reranker_disabled
    if _reranker_disabled:
        raise RuntimeError("BGE-Reranker 已被禁用（此前 OOM），使用原始上下文")

    # 检查持久化标记（上次进程被 OOM kill 后留下的标记文件）
    if _is_reranker_permanently_disabled():
        _reranker_disabled = True
        print(f"⚠️ 检测到 BGE-Reranker 已被永久禁用（{_RERANKER_DISABLED_FILE}），使用原始上下文", flush=True)
        raise RuntimeError("BGE-Reranker 已被永久禁用")

    # 检查可用内存
    avail_mem = _get_available_memory()
    if avail_mem < MIN_FREE_MEMORY_FOR_RERANKER:
        print(f"⚠️ 可用内存不足 ({avail_mem // (1024*1024)}MB < {MIN_FREE_MEMORY_FOR_RERANKER // (1024*1024)}MB)，"
              f"禁用 BGE-Reranker，使用原始上下文", flush=True)
        _reranker_disabled = True
        _disable_reranker_permanently()
        raise RuntimeError(f"可用内存不足，禁用 BGE-Reranker")

    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except (ValueError, OSError, ImportError) as e:
            print(f"⚠️ 本地 tokenizer 加载失败 ({e})，尝试在线下载...", flush=True)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if _model is None:
        try:
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        except (ValueError, OSError, ImportError) as e:
            print(f"⚠️ 本地模型加载失败 ({e})，尝试在线下载...", flush=True)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        _model.eval()
        print(f"✅ BGE-Reranker 模型加载成功（可用内存: {avail_mem // (1024*1024)}MB）", flush=True)


def compute_scores_batch(query: str, passages: list[str], sub_batch: int = 4) -> list[float]:
    """BGE-Reranker 打分，按 sub_batch 分批处理以防小内存实例 OOM。

    如果发生 OOM，标记 _reranker_disabled 并让异常向上传播，
    由 refine_and_rerank 降级为返回原始上下文。
    """
    global _reranker_disabled
    _ensure_model_loaded()
    all_scores = []
    for start in range(0, len(passages), sub_batch):
        batch = passages[start:start + sub_batch]
        inputs = _tokenizer(
            [[query, p] for p in batch],
            padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        try:
            with torch.no_grad():
                logits = _model(**inputs).logits
            all_scores.extend(torch.sigmoid(logits).flatten().tolist())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ BGE-Reranker OOM！禁用 reranker，降级为原始上下文", flush=True)
                _reranker_disabled = True
                _disable_reranker_permanently()
                # 释放显存/内存
                del inputs
                gc.collect()
                raise  # 抛给 refine_and_rerank 处理
            raise
        finally:
            # 每批处理完主动释放张量，减少峰值内存
            del inputs
    return all_scores

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

    如果 BGE-Reranker 加载或推理时发生 OOM，自动降级为返回去重后的原始上下文，
    确保知识问答功能不会因模型推理失败而完全不可用。
    """
    print(f"🔥 refine_and_rerank 被调用, contexts数量={len(raw_contexts)}, query={question[:50]}", flush=True)

    # 0. 原始段落去重，过滤标记行和过短文本（无论是否走 Reranker 都需要）
    clean_paras = []
    for p in raw_contexts:
        if re.match(r'^【片段\d+】$', p.strip()):
            continue
        if len(p) < 10:
            continue
        if p not in clean_paras:
            clean_paras.append(p)

    if not clean_paras:
        return []

    # 如果 Reranker 已被禁用（此前 OOM），直接返回去重后的原始上下文
    global _reranker_disabled
    if _reranker_disabled:
        print(f"⚠️ BGE-Reranker 已禁用，返回原始上下文 ({len(clean_paras[:top_k])} 段)", flush=True)
        return clean_paras[:top_k]

    # 1. 判断是否属于"列举/概述/介绍"类问题
    listing_keywords = ["有哪些", "包括什么", "主要景点", "主要文旅资源", "列举", "哪些",
                        "什么景区", "介绍", "概况", "概述", "基本情况", "主要资源"]
    is_listing_question = any(kw in question for kw in listing_keywords)

    # 2. 从问题中提取核心关键词（用于非列举类问题的预筛）
    keywords = re.findall(r'[一-龥]{2,}', question)

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

    # 5. BGE-Reranker 给所有句子打分（外层 try/except 防 OOM 降级）
    try:
        sentence_scores = compute_scores_batch(question, all_sentences_flat)
    except (RuntimeError, MemoryError) as e:
        print(f"⚠️ BGE-Reranker 打分失败 ({str(e)[:120]})，降级为原始上下文", flush=True)
        # 相似度去重后返回原始段落
        deduped = []
        for p in clean_paras:
            is_dup = False
            for existing in deduped:
                if SequenceMatcher(None, p, existing).ratio() > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(p)
            if len(deduped) >= top_k:
                break
        return deduped

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
