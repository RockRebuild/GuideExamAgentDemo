# 语义缓存设计文档

## 1. 问题背景

RAG 系统中用户提问具有明显的高频重复特征。以导游考试场景为例，"导游证的种类有哪些""旅游法第35条是什么""地陪导游接团前需要准备哪些证件"这类问题会被反复询问。每次请求都执行完整的检索 + BGE-Reranker 精排链路，造成不必要的计算开销和 API 调用。

传统 Redis key-value 缓存只能做**精确匹配**。用户问"导游证的分类"和"导游证的类别"本质是同一个问题，但字符串不同 → 缓存 miss。需要一种能识别**语义等价**的缓存机制。

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| **零额外依赖** | 复用项目已有的 ChromaDB + DashScope Embedding，不引入 Redis、Memcached 等新组件 |
| **语义级命中** | 相似度 ≥0.95 的两个查询共享缓存，不需要逐字相同 |
| **非侵入式** | 对检索调用方透明，`retrieve_with_rerank()` 内部集成，外部代码零改动 |
| **失败无影响** | 缓存读写异常时静默降级，不阻塞主流程 |
| **可观测** | 提供 `GET /api/agent-eval/cache-stats` 查看命中率 |

## 3. 架构设计

### 3.1 缓存存储层

缓存数据存在 ChromaDB 的一个独立集合 `semantic_cache` 中，和业务向量库（`guide_child` / `guide_parent` / `guide_summary` / `guide_sentence`）物理隔离。

```
chroma_db/
├── guide_child/           # 段落级（业务库）
├── guide_parent/          # 父切片（业务库）
├── guide_summary/         # 摘要（业务库）
├── guide_sentence/        # 句子级（业务库）
└── semantic_cache/        # 语义缓存（独立集合）
```

**为什么不另建一个 ChromaDB 实例？**
- ChromaDB 已在本进程中加载，复用同一个 persist_directory 开销为零
- 集合名不同即可做到物理隔离，没必要起第二个进程

### 3.2 缓存数据结构

```
每条缓存条目：

id:         MD5(query)[:16]           — 查询文本的短哈希，用于去重和删除
text:       检索结果（精排后的 contexts 拼接字符串）
metadata:
  query:       原始查询文本
  query_hash:  MD5(query)[:16]
  cached_at:   Unix 时间戳（用于 TTL 检查）
  ttl:         过期时间（秒）
```

**为什么 text 存的是精排后的结果而不是原始检索结果？**
精排（BGE-Reranker）是整个检索链路中计算最重的环节（CPU 逐句打分 + 排序）。如果缓存的是原始检索结果，命中后仍需走精排，缓存价值减半。缓存最终结果直接跳过整条链路。

## 4. 核心流程

### 4.1 缓存读取

```
用户查询 "导游证有哪些种类"
        │
        ▼
计算 query embedding (DashScope text-embedding-v4)
        │
        ▼
在 semantic_cache 集合中做相似度搜索 (k=1)
        │
        ├── cos_sim ≥ 0.95 → CACHE HIT
        │   │
        │   ├── TTL 未过期 → 直接返回缓存结果 ✅
        │   └── TTL 已过期 → 删除过期条目，算 MISS
        │
        └── cos_sim < 0.95 → CACHE MISS
            │
            ▼
        执行完整检索 + 精排 → 返回结果 + 写入缓存
```

### 4.2 缓存写入

```
新检索完成 → 结果字符串 ready
        │
        ▼
检查缓存条目数 ≥ 500（MAX_ENTRIES）？
        │
        ├── 是 → 淘汰最旧的 20%（100 条），腾出空间
        └── 否 → 继续
        │
        ▼
计算 query MD5 hash
        │
        ├── 相同 hash 的旧条目 → 删除（覆盖更新）
        └── 新条目 → ChromaDB.add_texts()
```

### 4.3 检索管线的集成点

缓存包裹的是 `retrieve_with_rerank()` 整段逻辑，不是某个单独的检索工具：

```python
# server/core/tools.py — 调用方视角（无感知）
def retrieve_with_rerank(query, raw_contexts):
    from server.core.semantic_cache import lookup_or_compute

    def _do_retrieve_and_rerank():
        top_k = determine_top_k(query)          # 动态 Top-K
        refined = refine_and_rerank(...)         # BGE 精排
        return CONTEXT_SEPARATOR.join(refined)

    result, is_cache_hit = lookup_or_compute(query, _do_retrieve_and_rerank)
    # ... 后续处理
```

所有检索工具（`search_textbook`、`hybrid_search`、`multi_search`、`parent_child_search`、`rewritten_search`）都经过这个入口 → 全部享受缓存加速。

## 5. 淘汰策略

三种淘汰机制组合：

| 机制 | 触发条件 | 动作 |
|------|---------|------|
| **TTL 过期** | 写入时间距今 > 24 小时 | 读取时检测，过期则删除 + 算 miss |
| **FIFO 淘汰** | 缓存条目数 ≥ 500 | 写入前删除最旧的 20% 条目 |
| **覆盖更新** | 相同 query hash 再次写入 | 删除旧条目，写入新条目（刷新 TTL） |

**为什么选 FIFO 而不是 LRU？**

ChromaDB 不提供访问次数统计，实现 LRU 需要额外维护一个 Redis/内存计数器。对于考试场景，热点问题具有时间稳定性（核心考点不会变），FIFO 的行为和 LRU 趋近一致——高频问题本身就会不断被重新写入，自动排在队列尾部不被淘汰。

**为什么是 20% 而不是逐条淘汰？**

逐条淘汰意味着每次写入都触发一次删除操作，增加写入延迟。批量淘汰 20%（约 100 条）将删除操作摊还到每 100 次新写入才触发一次，对时延的影响可以忽略。

## 6. 容错设计

```python
# 所有异常静默降级
def lookup(query):
    try:
        # ... 缓存逻辑
    except Exception:
        _cache_stats["misses"] += 1
        return None       # 出错 → 走正常检索链路

def store(query, result):
    try:
        # ... 写入逻辑
    except Exception:
        pass               # 写入失败 → 不影响主流程，下次重新检索即可
```

| 故障场景 | 缓存行为 | 业务影响 |
|---------|---------|---------|
| ChromaDB 客户端初始化失败 | lookup 返回 None | 每次走完整检索，功能无影响 |
| 向量搜索超时 | lookup 返回 None | 同上 |
| 写入时磁盘满 | store 静默跳过 | 当前结果正常返回，只是没被缓存 |
| 缓存集合损坏 | lookup/store 均捕获 Exception | 完全等同于无缓存 |

## 7. 配置参数

| 参数 | 环境变量 | 默认值 | 含义 |
|------|---------|--------|------|
| 相似度阈值 | `CACHE_SIMILARITY_THRESHOLD` | 0.95 | 语义匹配的严格程度。越高 → 命中越少但准确 |
| TTL | `CACHE_TTL_SECONDS` | 86400 (24h) | 缓存有效期。考题答案稳定，24h 合理 |
| 容量上限 | `CACHE_MAX_ENTRIES` | 500 | 最大条目数。500 条约占用 ~50MB 向量空间 |

**相似度阈值为什么是 0.95 而不是 0.8？**

RAG 场景下假阳性（不相似的查询命中同一缓存）的代价远高于假阴性（相似查询未命中）。比如"导游证种类"和"导游证申请流程"完全是两个问题，如果阈值设为 0.8，这两个查询可能共享缓存 → 返回错误答案。0.95 确保只有高度语义等价的查询才共享缓存。

## 8. 监控指标

```
GET /api/agent-eval/cache-stats

Response:
{
  "hits": 42,        // 缓存命中次数
  "misses": 158,     // 缓存未命中次数
  "total": 200,      // 总查询次数
  "hit_rate": 0.21   // 命中率 21%
}
```

| 指标 | 健康值 | 含义 |
|------|--------|------|
| hit_rate > 30% | 优秀 | 高频考点被有效缓存 |
| hit_rate 10-30% | 正常 | 缓存正常工作中 |
| hit_rate < 10% | 需排查 | 可能阈值过高或问题过于分散 |
| misses 快速增长 | 正常 | 新问题首次访问均需要走检索 |

## 9. 与 Redis 缓存的对比

| 维度 | Redis Key-Value | ChromaDB 语义缓存 |
|------|-----------------|-------------------|
| 匹配方式 | 精确字符串匹配 | 向量余弦相似度匹配 |
| "导游证分类" vs "导游证有哪些种类" | MISS | HIT (≥0.95) |
| 额外依赖 | 需要 Redis 实例 | 零（复用业务 ChromaDB） |
| 模糊匹配能力 | 无 | 天然支持 |
| 写入延迟 | <1ms | ~10ms（向量化 + 写入） |
| 适用场景 | Token/会话缓存 | NLP 查询缓存 |

本项目两种缓存各司其职：Redis 负责 checkpoint 会话记忆和错题本持久化（精确 key-value），ChromaDB 负责语义缓存（模糊语义匹配）。

## 10. 扩展方向

- **多级缓存**：热点问题自动提升到进程内存缓存（LRU dict），减少 ChromaDB 查询开销
- **写入异步化**：`store()` 改后台线程执行，不阻塞检索结果返回
- **预热机制**：将高频考题（如 7000 题题库对应的知识查询）预先检索并写入缓存，上线即命中
- **按模式区分 TTL**：知识点缓存 24h，天气查询缓存 30min，避免过期信息
