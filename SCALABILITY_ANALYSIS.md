# 知识库扩展到 100 个领域：架构瓶颈分析

## 前提

当前导游考试知识库规模：
- 4 本 PDF → 12,341 段落切片 (`guide_child`) + 1,791 父切片 (`guide_parent`) + 328 摘要 (`guide_summary`)
- 所有数据存入单一的 ChromaDB 持久化目录 `chroma_db/`
- 按 100 个同等级别的领域推算：**~120 万段落 + ~18 万父切片 + ~3 万摘要 = ~141 万向量**

---

## 瓶颈排序：谁最先撑不住

```
影响程度
  ▲
  │ ██████████████████████████████  #1 ChromaDB 单实例
  │ ██████████████████████████      #2 BM25 全局索引
  │ ██████████████████████          #3 领域路由缺失
  │ ███████████████████            #4 检索精度崩塌
  │ ██████████████                #5 BGE-Reranker 内存
  │ ██████████                    #6 Embedding API 吞吐
  │ ██████                        #7 LLM 上下文
  │ ████                          #8 Redis 会话
  └──→ 迟早（从左到右先炸）
```

---

## #1 ChromaDB 单实例（第一个撑不住，也最致命）

### 当前代码

```python
# server/core/tools.py:49,116-121
CHROMA_PERSIST_DIR = os.path.join(..., "chroma_db")
COLLECTION_PARAGRAPH = "guide_child"
COLLECTION_SUMMARY = "guide_summary"
COLLECTION_SENTENCE = "guide_sentence"

def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )
```

### 问题本质

ChromaDB 是一个**嵌入式、单进程、基于 SQLite** 的向量数据库。它不是为百万级向量设计的。

| 维度 | 1 个领域（当前） | 100 个领域 | 后果 |
|---|---|---|---|
| 向量数量 | 1.4 万 | **~141 万** | HNSW 索引退化，SQLite 文件 >10GB |
| 写入吞吐 | 几千条/秒 | 百万条需数小时，中途崩了无法恢复 | 数据导入变成灾难 |
| 查询延迟 | ~50ms | **500ms~2s**（HNSW 遍历层数增加） | 用户体验不可接受 |
| 并发读写 | 锁竞争低 | SQLite 写锁 → 写入时查询全部阻塞 | 不能一边更新知识库一边服务 |
| 内存 | ~200MB | 141万×1024维×4字节 = **~5.7GB** 纯向量 + 索引开销 >8GB | OOM，ECS 4GB 实例直接挂 |
| 元数据过滤 | `where={"chapter": "xxx"}` 可以 | `where={"domain": "medicine"}` 扫全表 | 过滤不改变扫描量 |

### 根本原因

ChromaDB 没有分布式架构——不能分片、不能水平扩展、不能读写分离。它本质是"SQLite + HNSW 的 Python 包装"，适合原型和中小规模，不适合 100 领域生产场景。

### 解决方案

**必须更换向量数据库**，按投入成本从低到高：

| 方案 | 说明 | 适用 |
|---|---|---|
| ChromaDB → **Milvus** | 分布式、支持分片、Mmap 优化内存、GPU 索引 | 推荐，生产验证最充分 |
| ChromaDB → **Qdrant** | Rust 实现、低内存、高性能 | 内存敏感场景的首选 |
| 保留 ChromaDB + 分实例 | 每个领域一个 `chroma_db` 目录 + 路由层分发 | 最小改动，但运维复杂 |
| 保留 ChromaDB + 分区索引 | 利用 `collection` 按领域拆分 + 领域分类器路由 | 过渡方案，延迟最低 |

---

## #2 BM25 全局内存索引

### 当前代码

```python
# server/core/tools.py:127-145
_bm25_index = None
_bm25_docs = None

def _init_bm25():
    global _bm25_index, _bm25_docs
    parent_store = get_vectorstore("guide_parent")
    all_data = parent_store.get()
    _bm25_docs = all_data['documents']                    # 全量加载到内存
    tokenized_corpus = [list(jieba.cut(doc)) for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized_corpus)              # 全量分词索引
```

然后每次搜索做全量打分：

```python
# server/core/tools.py:222-228
def hybrid_search(query):
    _init_bm25()
    scores = _bm25_index.get_scores(tokenized_query)       # O(18万) 次乘法
    top_indices = np.argsort(scores)[-k:][::-1]            # O(18万 log k)
```

### 问题本质

这是一个**模块级全局变量 + O(n) 线性扫描**的设计模式。

| 维度 | 1 个领域（当前） | 100 个领域 | 后果 |
|---|---|---|---|
| `_bm25_docs` 内存 | 1,791 条 × 800 字符 ≈ **1.4MB** | 18 万条 ≈ **144MB** | 纯文本还好 |
| 分词后内存 | 1,791 × ~400 tokens ≈ 几 MB | 18 万 × ~400 tokens ≈ **~1GB** | jieba 分词后内存爆炸 |
| `get_scores()` 计算 | O(1,791) ≈ 微秒 | O(180,000) ≈ **几十毫秒** | 每次搜索都卡 |
| `argsort` | O(1,791 log k) | O(180,000 log k) | 额外开销 |
| 领域混杂 | 只有一个领域 | 所有领域混在一起打分 | 法律领域的词命中医学文档，**噪音淹没信号** |

### 根本原因

Python `rank_bm25` 库是一个教学级的实现——没有倒排索引（只有 TF-IDF 的变体），`get_scores()` 需要遍历每一个文档。真实搜索引擎用的是倒排索引 + 跳跃表（如 Elasticsearch 的 Lucene），时间复杂度 O(query_terms × avg_posting_length)，与文档总量基本无关。

### 解决方案

| 方案 | 改动 | 效果 |
|---|---|---|
| **Elasticsearch** | 代替 BM25 Python 库，独立部署 | O(1) 级别的关键词检索，天然支持分片 |
| **Milvus 混合检索** | 开 Milvus 的 BM25 功能 | 向量 + 关键词一体，不需要两个系统 |
| **RedisSearch** | 已有 Redis，开 RediSearch 模块 | 最小运维增量 |
| 按领域拆分 BM25 索引 | 每个请求先做领域分类，再 load 对应索引 | 过渡方案，但 100 个索引的管理麻烦 |

---

## #3 领域路由缺失

### 当前代码

所有检索工具都硬编码了 collection 名：

```python
# 5 个检索函数，全部直接调用
get_vectorstore("guide_child")
get_vectorstore("guide_parent")
get_vectorstore("guide_summary")
```

没有"用户问的是哪个领域"的识别步骤。`rewritten_search` 会做问题改写，但改写的目的是"同一个问题的不同表述"，不是领域分类。

### 问题

用户问"什么是急性胰腺炎" → 系统会在导游教材里搜 → 搜不到 → 返回"未找到相关内容"。用户问"合同的生效要件" → 可能在法律教材里找到，但在导游教材里也搜出了"旅游合同" → 两个领域混合。

当前 **Supervisor 路由**（`server/core/multi_agent.py`）只区分"知识问答/出卷/批改"三种意图，不区分领域。

### 解决方案

在限流 guard 和检索之间插入一个**领域分类器**：

```
用户问题 → 领域分类器（LLM/轻量模型）→ {domain_id}
         → 路由到对应 domain 的 collection → 检索
```

领域分类器可以用：
- 一个短 prompt 的 LLM 调用（增加 ~200ms）
- 一个本地 text classification 模型（更快，但需要训练/标注）

---

## #4 检索精度崩塌

### 问题

当前检索是"全量向量搜索"：

```python
# hybrid_search: k=12 个结果
para_store.similarity_search(query, k=k)

# parent_child_search: k=24 个子切片 → 合并 parent_id → 取前 12 个
child_store.similarity_search(query, k=k * 2)

# multi_search: 三个 collection 各取 k=12 → 合并去重取前 k
summary_store.similarity_search(query, k=k)
para_store.similarity_search(query, k=k)
sent_store.similarity_search(query, k=k)
```

100 个领域各有 ~12K 段落。总共 120 万段落中取 top-12 → **每个段落需要跟 120 万个向量比距离**。

精度问题：
- 高维向量的"最近邻"在大规模数据中会退化——噪声向量多了，`k=12` 里混入不相关结果的概率大幅上升
- 跨领域的语义重叠：医学术语和法学用语在某些向量空间方向上的投影可能相近
- `deduplicate_docs` 用 `SequenceMatcher.ratio() > 0.8` 做 O(n²) 去重，12 条还好，但从 120 万中捞出的 12 条之间的重叠率下降

### 解决方案

必须**先分领域再检索**，而不是在全量向量中捞。这回到 #3 的领域路由。同时需要：
- 引入 **Reranker 评分阈值**（当前就有 `min_score=0.5`，但需要按领域调整）
- 考虑 **HyDE**（Hypothetical Document Embeddings）：让 LLM 先想象答案长什么样，再拿想象去搜

---

## #5 BGE-Reranker 内存

### 当前代码

```python
# retrieval_utils.py:145-151
def rerank_contexts(query, contexts, top_k=5):
    scores = compute_scores_batch(query, valid)     # 全量跑 BGE cross-encoder
    sorted_indices = sorted(range(len(scores)), ...)
    return [valid[i] for i in sorted_indices[:top_k]]
```

### 问题

BGE-Reranker-base 模型约 1.1GB，推理时临时张量峰值到 1.5GB。当前 Docker ECS 实例 4GB 内存，已经因为这个模型多次 OOM（代码里甚至做了 `_disable_reranker_permanently` 的持久化降级逻辑）。

扩展到 100 领域时，Reranker 本身不需要更多内存（它只处理 top-k 个候选），所以这个问题**不是扩展带来的新问题，而是已有的老问题**。但如果候选来源从单一领域变为多领域混合，需要加大 `k`（从 12 变成 24 甚至 48），Reranker 的输入增大 → 推理内存和延迟线性增长。

### 当前代码已有降级

```python
# retrieval_utils.py:87-92
if avail_mem < MIN_FREE_MEMORY_FOR_RERANKER:
    _reranker_disabled = True
    _disable_reranker_permanently()
    raise RuntimeError("可用内存不足，禁用 BGE-Reranker")
```

当 OOM 或内存不足时，自动跳过 Reranker，用原始向量检索结果。这在 100 领域场景下意味着**失去精排能力**，检索精度进一步下降。

---

## #6 Embedding API 吞吐

### 当前代码

```python
# tools.py:61-65
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    return _embeddings
```

### 问题

数据导入阶段：141 万段落 × 1 次 API 调用/段落（或 batch 模式）= 大量 API 调用。DashScope embedding API 有 QPS 限制和配额。

查询阶段：每次用户搜索才调用 1 次 embedding API（对用户问题编码），所以查询阶段 **DashScope 不是瓶颈**。

真正的问题在**数据导入**阶段。如果每个领域都要重新 embedding，141 万次 API 调用按 50 QPS 算 = 需要约 8 小时持续调用。中途网络断了怎么办？

### 解决方案

- 批量 embedding API（DashScope 支持 batch 模式，一次传多条文本）
- 导入失败的重试 + 断点续传
- 如果切换到 Milvus/Qdrant，其生态有更成熟的批量导入工具

---

## #7 LLM 上下文窗口

### 问题

当前 RAG 给 LLM 喂 ~5 个精排后的段落（top_k=5）。LLM 上下文不会被 100 个领域撑爆，因为检索阶段已经缩小到 top-k。

但**问题质量会下降**：如果没有领域路由（#3），LLM 收到的 5 个段落可能来自 3 个不同领域，它会"强行"把不相关的段落拼在一起回答，产生幻觉。

DeepSeek V4 的 128K 上下文窗口在这里不是瓶颈。真正的瓶颈是**检索精度**决定了 LLM 收到的内容质量。

---

## #8 Redis 会话

### 问题

`RedisSaver` 存储 LangGraph checkpoint，每个会话线程一个 key。100 个领域的知识问答量增大，checkpoint 数据量线性增长。但 Redis 本身可以轻松处理百万级别的 key → 这不是瓶颈。

---

## 爆炸顺序总结

```
领域数  1 ──────── 10 ──────── 30 ────────────── 50 ───────────────── 100
        │          │          │                  │                      │
ChromaDB │ 正常     │ 延迟上升 │ SQLite 锁竞争     │ HNSW 退化           │ 查询 2s+
         │          │          │ 文件 >1GB        │ 内存 >4GB           │ 不可用
         │          │          │                                     │
BM25     │ 正常     │ 内存增长 │ 分词 1GB+         │ 全量扫描 100ms+     │ 单次搜索卡顿
         │          │          │                                     │
领域路由 │ 不需要   │ 开始混   │ 噪音 > 信号       │ 精度崩塌            │ 答非所问
         │          │          │                                     │
Reranker │ 勉强     │ 内存紧   │ OOM 频繁          │ 降级（跳过精排）     │ 精度下降
```

**ChromaDB 是第一个硬瓶颈**，大约在 30~50 个领域时开始明显退化。

**BM25 和领域路由**紧随其后，大约在 20~30 个领域时开始暴露精度问题。

**Reranker** 是已有的老问题，不是因为扩展才出现，但扩展会恶化它。

---

## 推荐的迁移路径

### 第一阶段：解决硬瓶颈（ChromaDB）

```
ChromaDB → Milvus (或 Qdrant)
  - 每个领域一个 Partition/Collection
  - 利用 Milvus MMap 降低内存
  - 批量导入 + 断点续传
```

### 第二阶段：解决检索精度（领域路由 + BM25）

```
新增 DomainClassifier（轻量 LLM prompt 或 分类模型）
  → 用户问题 → {domain_id, confidence}
  → 查询路由到对应 domain 的 Partition
BM25 → Elasticsearch 或 Milvus 内置 BM25
  → 向量 + 关键词混合检索
```

### 第三阶段：优化精排和缓存

```
Reranker 替换为 API 服务（BGE-Reranker 部署为独立微服务，GPU 实例）
  → 主服务内存不再被 reranker 占用
  → 多个 worker 共享一个 reranker 服务
语义缓存：按领域分片，提高命中率
  → domain_medicine_cache, domain_law_cache...
```
