#!/usr/bin/env python3
# tests/bench_chromadb.py
# ── ChromaDB 检索性能基准测试 ──
#
# 原理:
#   ChromaDB 底层是 SQLite + HNSW 索引。随着数据量增长，查询性能非线性退化。
#   本 benchmark 在可控条件下测量不同并发、不同 k 值下的延迟分布，
#   用于容量规划和迁移决策（何时该从 ChromaDB 迁移到 Milvus/Qdrant）。
#
# 运行:
#   python tests/bench_chromadb.py
#
# 输出 (示例):
#   ==================== ChromaDB Benchmark ====================
#   集合: guide_child (12,341 文档)
#   并发级别: [1, 5, 10, 20]
#   每级迭代: 20
#
#   并发=1   │ avg= 45ms  p50= 42ms  p95= 68ms  p99= 92ms  QPS= 22.2
#   并发=5   │ avg= 82ms  p50= 75ms  p95=145ms  p99=210ms  QPS= 61.0
#   并发=10  │ avg=156ms  p50=142ms  p95=289ms  p99=420ms  QPS= 64.1  ← 拐点
#   并发=20  │ avg=320ms  p50=298ms  p95=610ms  p99=890ms  QPS= 62.5
#
# 解读:
#   - QPS 在并发=10 时达到峰值 64.1 → 系统容量瓶颈，再加并发效率下降
#   - P99 延迟在并发=20 时接近 1s → 不可接受
#   - 建议: 并发超过 10 时考虑向量库升级或加缓存层

import os
import sys
import time
import json
import statistics
import concurrent.futures
from typing import List, Dict

# 确保项目根在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ── 测试查询池 ─────────────────────────────────────────

TEST_QUERIES = [
    "导游证的种类有哪些？",
    "旅游法第35条是什么？",
    "地陪导游接团前需要准备哪些证件？",
    "全陪导游和地陪导游的工作有什么区别？",
    "什么是旅游合同？",
    "合同法律制度包括哪些内容？",
    "中国历史文化概述",
    "导游考试一共考几科？",
    "旅行社保姆规范有哪些要求？",
    "导游证吊销的条件是什么？",
]


# ── 统计工具 ─────────────────────────────────────────

def percentiles(values: List[float], ps: List[float]) -> Dict[float, float]:
    """计算百分位数（线性插值）。"""
    if not values:
        return {p: 0 for p in ps}
    sorted_vals = sorted(values)
    result = {}
    for p in ps:
        k = (len(sorted_vals) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_vals):
            result[p] = sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
        else:
            result[p] = sorted_vals[f]
    return result


def run_benchmark(
    queries: List[str],
    concurrency: int,
    iterations: int = 20,
) -> Dict:
    """在给定并发级别下运行基准测试。"""
    from server.core.tools import get_vectorstore, hybrid_search

    # 确保索引已初始化
    from server.core.tools import _init_bm25
    _init_bm25()

    latencies = []
    errors = 0

    start_wall = time.monotonic()

    if concurrency == 1:
        # 串行模式：直接调，消除线程池开销
        for _ in range(iterations):
            for q in queries:
                try:
                    t0 = time.monotonic()
                    result = hybrid_search.invoke({"query": q, "k": 12})
                    latencies.append((time.monotonic() - t0) * 1000)
                except Exception as e:
                    errors += 1
                    print(f"  Error: {e}")
    else:
        # 并发模式
        def worker(query: str):
            try:
                t0 = time.monotonic()
                result = hybrid_search.invoke({"query": query, "k": 12})
                return (time.monotonic() - t0) * 1000
            except Exception as e:
                return None  # 标记错误

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            tasks = []
            for _ in range(iterations):
                for q in queries:
                    tasks.append(pool.submit(worker, q))

            for future in concurrent.futures.as_completed(tasks):
                result = future.result()
                if result is not None:
                    latencies.append(result)
                else:
                    errors += 1

    wall_time = time.monotonic() - start_wall
    total_requests = len(latencies) + errors
    qps = total_requests / wall_time if wall_time > 0 else 0

    if not latencies:
        return {"concurrency": concurrency, "error": "No successful requests"}

    p = percentiles(latencies, [50, 75, 90, 95, 99])

    return {
        "concurrency": concurrency,
        "requests": total_requests,
        "errors": errors,
        "error_rate": round(errors / total_requests * 100, 2) if total_requests else 0,
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(p[50], 1),
        "p75_ms": round(p[75], 1),
        "p90_ms": round(p[90], 1),
        "p95_ms": round(p[95], 1),
        "p99_ms": round(p[99], 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "wall_time_s": round(wall_time, 2),
        "qps": round(qps, 1),
    }


def print_table(results: List[Dict]):
    """格式化打印结果表格。"""
    print(f"\n{'='*80}")
    print(f"ChromaDB Benchmark: hybrid_search (语义 + BM25 + BGE 精排)")
    print(f"{'='*80}")
    header = (
        f"{'并发':>4}  {'avg':>7}  {'p50':>7}  {'p95':>7}  "
        f"{'p99':>7}  {'min':>7}  {'max':>7}  {'QPS':>7}  "
        f"{'错误率':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"  {r['concurrency']:>4}  ERROR: {r['error']}")
            continue
        print(
            f"{r['concurrency']:>4}  "
            f"{r['avg_latency_ms']:>6.0f}ms  "
            f"{r['p50_ms']:>6.0f}ms  "
            f"{r['p95_ms']:>6.0f}ms  "
            f"{r['p99_ms']:>6.0f}ms  "
            f"{r['min_ms']:>6.0f}ms  "
            f"{r['max_ms']:>6.0f}ms  "
            f"{r['qps']:>6.1f}  "
            f"{r['error_rate']:>5.1f}%"
        )

    print(f"\n{'='*80}")
    # 容量建议
    valid = [r for r in results if "error" not in r]
    if len(valid) >= 2:
        # 找 QPS 峰值
        peak_qps = max(valid, key=lambda r: r["qps"])
        peak_latency_p95 = peak_qps["p95_ms"]
        print(f"📊 容量分析:")
        print(f"   QPS 峰值: {peak_qps['qps']:.1f} (并发={peak_qps['concurrency']})")
        print(f"   峰值时 P95: {peak_latency_p95:.0f}ms")

        # 判断是否到达拐点
        last_valid = valid[-1]
        if last_valid["p99_ms"] > 1000:
            print(f"   ⚠️  并发={last_valid['concurrency']} 时 P99>{1000}ms — 建议控制并发")
        if peak_qps["qps"] < last_valid["qps"] * 0.7:
            print(f"   ⚠️  QPS 随并发下降 — 已达到系统瓶颈")
        elif last_valid["p95_ms"] > 500:
            print(f"   ⚡ P95 延迟 {last_valid['p95_ms']:.0f}ms — 用户体验下降")

        # 与配置对比
        cfg_rpm = int(os.environ.get("CONCURRENCY_GLOBAL_RPM", "50"))
        peak_rpm = peak_qps["qps"] * 60
        print(f"   当前限流 global_rpm={cfg_rpm}，峰值可达 {peak_rpm:.0f} RPM")
        if peak_rpm < cfg_rpm:
            print(f"   ✅ ChromaDB 不是瓶颈（峰值 RPM {peak_rpm:.0f} < 限流 {cfg_rpm}）")
        else:
            print(f"   ⚠️  ChromaDB 可能先于限流成为瓶颈")


# ── 单工具对比 ─────────────────────────────────────────

def bench_individual_tools():
    """对比各检索策略的单次延迟。"""
    from server.core.tools import (
        search_textbook, hybrid_search, multi_search,
        parent_child_search, rewritten_search,
    )

    tools = {
        "search_textbook": lambda q: search_textbook.invoke({"query": q}),
        "hybrid_search": lambda q: hybrid_search.invoke({"query": q, "k": 12}),
        "multi_search": lambda q: multi_search.invoke({"query": q, "k": 12}),
        "parent_child_search": lambda q: parent_child_search.invoke({"query": q, "k": 12}),
        "rewritten_search": lambda q: rewritten_search.invoke({"original_query": q, "k": 12}),
    }

    print(f"\n{'='*80}")
    print(f"单工具延迟对比 (query: '导游证的种类有哪些？')")
    print(f"{'='*80}")
    print(f"{'工具':>25}  {'avg':>7}  {'min':>7}  {'max':>7}")

    for name, fn in tools.items():
        latencies = []
        for _ in range(3):  # 每个工具跑 3 次取平均
            try:
                t0 = time.monotonic()
                result = fn("导游证的种类有哪些？")
                latencies.append((time.monotonic() - t0) * 1000)
            except Exception as e:
                print(f"  {name:>25}  ERROR: {e}")
                continue

        if latencies:
            print(
                f"  {name:>25}  "
                f"{statistics.mean(latencies):>6.0f}ms  "
                f"{min(latencies):>6.0f}ms  "
                f"{max(latencies):>6.0f}ms"
            )
            # 特殊标注: rewritten_search 有额外的 LLM 调用
            if name == "rewritten_search":
                print(f"  {'':>25}  ⚠️ 含 LLM 改写 (DeepSeek API 调用)")


# ── Main ──────────────────────────────────────────────

def main():
    print("🔬 ChromaDB 检索性能基准测试\n")

    # 先看集合信息
    try:
        from server.core.tools import get_vectorstore
        for coll in ["guide_child", "guide_parent", "guide_summary", "semantic_cache"]:
            try:
                vs = get_vectorstore(coll)
                count = vs._collection.count()
                print(f"   集合 {coll}: {count} 条")
            except Exception:
                print(f"   集合 {coll}: 不可用")
    except Exception as e:
        print(f"⚠️ ChromaDB 初始化失败: {e}")
        print("   请先确保 chroma_db/ 目录中有数据")
        return 1

    # 预热：首次调用加载 BM25 + BGE 模型
    print("\n🔥 预热中（加载 BM25 索引 + BGE Reranker）...")
    try:
        from server.core.tools import hybrid_search, _init_bm25
        _init_bm25()
        hybrid_search.invoke({"query": "导游证", "k": 5})
        print("   预热完成\n")
    except Exception as e:
        print(f"   预热失败: {e}\n")

    # 并发阶梯测试
    concurrency_levels = [1, 3, 5, 8, 10, 15, 20]
    # 默认只跑低并发（高并发需要较长时间）
    max_concurrency = int(os.environ.get("BENCH_MAX_CONCURRENCY", "10"))
    concurrency_levels = [c for c in concurrency_levels if c <= max_concurrency]

    results = []
    for c in concurrency_levels:
        print(f"  并发={c} ... ", end="", flush=True)
        r = run_benchmark(TEST_QUERIES[:3], c, iterations=5)
        results.append(r)
        if "error" not in r:
            print(f"avg={r['avg_latency_ms']:.0f}ms, QPS={r['qps']:.1f}")
        else:
            print(f"ERROR: {r['error']}")

    print_table(results)

    # 单工具对比
    bench_individual_tools()

    # 输出 JSON（方便自动化对比）
    json_output = os.environ.get("BENCH_JSON_OUTPUT", "")
    if json_output:
        with open(json_output, "w") as f:
            json.dump({"summary": results}, f, ensure_ascii=False, indent=2)
        print(f"\n📄 结果已导出: {json_output}")

    return 0


if __name__ == "__main__":
    exit(main())
