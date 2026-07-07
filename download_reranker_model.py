"""下载完整的 BGE-Reranker-base 模型到 bge_reranker_cache 目录。

当前缓存目录只有 model.safetensors，缺少 tokenizer 和 config 文件导致加载失败。
"""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "BAAI/bge-reranker-base"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "bge_reranker_cache", "BAAI", "bge-reranker-base")

print(f"开始下载 {MODEL_ID} → {SAVE_DIR}")
os.makedirs(SAVE_DIR, exist_ok=True)

# 下载所有文件（tokenizer.json, config.json, tokenizer_config.json, vocab.txt 等）
# 已有文件会跳过，只下载缺失的文件
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot", "*.onnx"],
    resume_download=True,
    max_workers=4,
)

print("下载完成！")
print("文件列表:")
for f in sorted(os.listdir(SAVE_DIR)):
    size_mb = os.path.getsize(os.path.join(SAVE_DIR, f)) / (1024 * 1024)
    print(f"  {f} ({size_mb:.1f} MB)")
