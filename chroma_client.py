import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_db")
# 或者非持久化（演示用）：client = chromadb.EphemeralClient()

collection = client.get_or_create_collection(
    name="guide_textbook",
    metadata={"description": "导游考试教材知识库"}
)

# 添加文档（手动给 ID）
collection.add(
    documents=["地陪导游需要在接团前准备好导游证和接待计划。"],
    metadatas=[{"chapter": "导游业务/第三章"}],
    ids=["doc_001"]
)

texts = ["全陪导游负责全程陪同...", "地陪服务规范包括..."]
metadatas = [{"chapter": "导游业务/第二章"}, {"chapter": "导游业务/第三章"}]
ids = [f"doc_{i}" for i in range(len(texts))]


collection.add(documents=texts, metadatas=metadatas, ids=ids)

results = collection.query(
    query_texts=["地陪导游接团前准备什么？"],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(results['documents'][0],
                           results['metadatas'][0],
                           results['distances'][0]):
    print(f"内容: {doc}\n章节: {meta['chapter']}\n距离: {dist}\n")

    results = collection.query(
        query_texts=["导游接团流程"],
        n_results=3,
        where={"chapter": "导游业务/第三章"},  # 只搜第三章
        include=["documents"]
    )

    collection.update(ids=["doc_001"], documents=["新的文本内容"])
    collection.delete(ids=["doc_001"])
    # 按条件删除
    collection.delete(where={"chapter": "已废弃"})

