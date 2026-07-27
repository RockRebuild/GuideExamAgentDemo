# server/core/semantic_cache.py
# ── 语义缓存层 ──
# 基于 Embedding 相似度缓存检索结果，减少重复 API 调用和计算开销。
#
# 原理:
#   1. 新查询 → 计算 embedding → 在缓存中搜索相似历史查询
#   2. 余弦相似度 > 阈值 → 直接返回缓存结果（Cache Hit）
#   3. 否则 → 执行实际检索 → 结果存入缓存（Cache Miss）
#
# 优势:
#   - 高频重复问题（如"导游证有几种"）无需重复检索 + 精排
#   - 减少 DeepSeek API 调用（rewritten_search 的 LLM 改写步骤也可跳过）
#   - ChromaDB 原生支持向量搜索，零额外依赖

import hashlib
import json
import os
import time
from typing import Optional, Tuple

from langchain_chroma import Chroma

CACHE_COLLECTION_NAME = "semantic_cache"
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db"
)

# 缓存配置（环境变量可覆盖）
CACHE_SIMILARITY_THRESHOLD = float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.95"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(3600 * 24)))  # 默认24小时
CACHE_MAX_ENTRIES = int(os.environ.get("CACHE_MAX_ENTRIES", "500"))

# 统计指标
_cache_stats = {"hits": 0, "misses": 0, "total": 0}


def get_cache_stats() -> dict:
    """获取缓存命中统计。"""
    total = _cache_stats["total"]
    hit_rate = _cache_stats["hits"] / total if total > 0 else 0.0
    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "total": total,
        "hit_rate": round(hit_rate, 4),
    }


def _get_cache_collection() -> Chroma:
    """获取或创建缓存专用 ChromaDB 集合。"""
    from server.core.tools import get_embeddings
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=CACHE_COLLECTION_NAME,
    )


def _hash_query(query: str) -> str:
    """生成查询的短哈希，用于缓存 ID。"""
    return hashlib.md5(query.encode()).hexdigest()[:16]


def _is_expired(timestamp: float) -> bool:
    """检查缓存条目是否过期。"""
    return (time.time() - timestamp) > CACHE_TTL_SECONDS


def lookup(query: str) -> Optional[str]:
    """在缓存中查找语义相似的查询结果。

    Args:
        query: 用户查询文本

    Returns:
        缓存命中时返回之前存储的结果字符串，未命中返回 None。
    """
    _cache_stats["total"] += 1

    try:
        collection = _get_cache_collection()
        if collection._collection.count() == 0:
            _cache_stats["misses"] += 1
            return None

        # 语义相似搜索
        results = collection.similarity_search_with_relevance_scores(query, k=1)

        if not results:
            _cache_stats["misses"] += 1
            return None

        doc, score = results[0]

        # 相似度阈值检查
        if score < CACHE_SIMILARITY_THRESHOLD:
            _cache_stats["misses"] += 1
            return None

        # TTL 检查
        cached_at = doc.metadata.get("cached_at", 0)
        if _is_expired(cached_at):
            # 删除过期条目
            try:
                collection.delete(ids=[doc.metadata.get("query_hash", "")])
            except Exception:
                pass
            _cache_stats["misses"] += 1
            return None

        _cache_stats["hits"] += 1
        return doc.page_content

    except Exception:
        _cache_stats["misses"] += 1
        return None


def store(query: str, result: str):
    """将查询-结果对存入缓存。

    Args:
        query: 用户查询文本
        result: 检索结果字符串
    """
    try:
        collection = _get_cache_collection()

        # 限制缓存条目数量（FIFO 策略：删最旧的）
        count = collection._collection.count()
        if count >= CACHE_MAX_ENTRIES:
            # 获取所有条目，按时间排序，删除最旧的 20%
            all_data = collection.get()
            if all_data["ids"]:
                # 简单策略：删除前 N 个
                delete_count = max(1, int(CACHE_MAX_ENTRIES * 0.2))
                oldest_ids = all_data["ids"][:delete_count]
                collection.delete(ids=oldest_ids)

        query_hash = _hash_query(query)
        timestamp = time.time()

        # 删除同 hash 旧条目（如有）
        try:
            collection.delete(ids=[query_hash])
        except Exception:
            pass

        collection.add_texts(
            texts=[result],
            metadatas=[{
                "query": query,
                "query_hash": query_hash,
                "cached_at": timestamp,
                "ttl": CACHE_TTL_SECONDS,
            }],
            ids=[query_hash],
        )

    except Exception:
        pass  # 缓存写入失败不应影响主流程


def lookup_or_compute(query: str, compute_fn, auto_store: bool = True) -> Tuple[str, bool]:
    """语义缓存的统一入口：命中返回缓存，未命中计算后可选择是否缓存。

    Args:
        query: 用户查询文本
        compute_fn: 实际检索函数（无参数的可调用对象）
        auto_store: 是否自动存入缓存（默认 True）。设为 False 时仅查找不写入，
                    缓存写入由外部在确认质量后调用 store()。

    Returns:
        (result_text, is_cache_hit): 结果字符串和是否命中缓存
    """
    # 尝试缓存命中
    cached = lookup(query)
    if cached is not None:
        return cached, True

    # 缓存未命中 → 执行实际检索
    result = compute_fn()

    # auto_store=False 时跳过写入，等待外部确认质量后再 store()
    if auto_store and result:
        store(query, result)

    return result, False


def remove_by_query(query: str) -> bool:
    """语义搜索并删除匹配的缓存条目。

    用户点"无用"时调用，防止低质量缓存被反复命中。
    查找与 query 语义最相似的缓存条目，相似度超过阈值则删除。

    Returns:
        True 表示找到并删除了，False 表示没有匹配的缓存条目。
    """
    try:
        collection = _get_cache_collection()
        if collection._collection.count() == 0:
            return False

        results = collection.similarity_search_with_relevance_scores(query, k=1)
        if not results:
            return False

        doc, score = results[0]
        if score < CACHE_SIMILARITY_THRESHOLD:
            return False

        query_hash = doc.metadata.get("query_hash", "")
        if query_hash:
            collection.delete(ids=[query_hash])
            return True
        return False

    except Exception:
        return False


def clear_cache() -> int:
    """清空所有缓存条目。返回删除的条目数。"""
    try:
        collection = _get_cache_collection()
        count = collection._collection.count()
        # 删除集合中的所有数据
        all_data = collection.get()
        if all_data["ids"]:
            collection.delete(ids=all_data["ids"])
        return count
    except Exception:
        return 0
