from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma

from tools import get_vectorstore
from dotenv import load_dotenv

load_dotenv()  # 自动查找并加载项目根目录下的 .env 文件
from tools import get_embeddings
embeddings = get_embeddings()
client = PersistentClient(path="./../chroma_db")
collections = client.list_collections()
# try:
#     client.delete_collection("guide_textbook")
#     client.delete_collection("guide_parent")
#     print("已删除旧集合")
# except ValueError:
#     pass
for c in collections:
    print(f"{c.name}: {c.count()} 条文档")
try:
    vec = embeddings.embed_query("测试文本")
    print("Embedding 正常，向量长度:", len(vec))
except Exception as e:
    print("Embedding 调用失败:", str(e))
child = Chroma(
        persist_directory="./../chroma_db",
        embedding_function=get_embeddings(),
        collection_name="guide_textbook",
    )
print("子切片数:", child._collection.count())
parent = Chroma(
        persist_directory="./../chroma_db",
        embedding_function=get_embeddings(),
        collection_name="guide_parent",
    )
print("父切片数:", parent._collection.count())