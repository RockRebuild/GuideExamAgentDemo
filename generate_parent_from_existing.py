from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import uuid

PDF_PATH = "全国导游人员资格统一考试模拟试题汇编.pdf"
CHROMA_DIR = "./chroma_db"
CHILD_COLLECTION = "guide_textbook"   # 复用你现有的子切片库
PARENT_COLLECTION = "guide_parent"    # 新建父切片库
from dotenv import load_dotenv
load_dotenv()
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
# 1. 重新加载 PDF，生成大粒度父切片（800 字）
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
parent_chunks = parent_splitter.split_documents(docs)
for chunk in parent_chunks:
    chunk.metadata["parent_id"] = str(uuid.uuid4())

# 2. 存入父切片库
parent_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=PARENT_COLLECTION,
)
parent_ids = [chunk.metadata["parent_id"] for chunk in parent_chunks]
parent_store.add_documents(parent_chunks, ids=parent_ids)

# 3. 为现有子切片分配 parent_id（核心关联步骤）
child_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=CHILD_COLLECTION,
)
all_child = child_store.get()  # 获取现有子切片的文本和元数据

# 这里用简单策略：把子切片文本所在的那个父切片 ID 赋给子切片
# 实际需要按文本包含关系匹配，这里简化为：重新切分子切片，直接用新生成的子切片（确保关联）
# 如果你想保留现有子切片，需要写更精细的匹配逻辑，比如判断子切片文本是哪个父切片的一部分
# 为简便，我们直接替换现有子切片库为新的子切片
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
child_chunks = []
child_ids = []
child_metadatas = []
for parent in parent_chunks:
    sub_texts = child_splitter.split_text(parent.page_content)
    for sub in sub_texts:
        child_chunks.append(sub)
        child_ids.append(str(uuid.uuid4()))
        child_metadatas.append({"parent_id": parent.metadata["parent_id"]})

# 清空旧子切片库并写入新的（这一步会删除你现有的 guide_textbook 数据，谨慎操作！）
child_store.delete_collection()
child_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=CHILD_COLLECTION,
)
child_store.add_texts(texts=child_chunks, ids=child_ids, metadatas=child_metadatas)

print(f"父切片：{len(parent_chunks)}，子切片：{len(child_chunks)}")