import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# 删除指定 Collection
client.delete_collection("guide_summary")  # 替换成你要删除的集合名