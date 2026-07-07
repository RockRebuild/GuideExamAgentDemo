#!/bin/bash
set -e

MODEL_DIR="/app/bge_reranker_cache/BAAI/bge-reranker-base"
MODEL_ID="BAAI/bge-reranker-base"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 检查 BGE 模型是否存在（至少要有 tokenizer 和 config）
if [ ! -f "$MODEL_DIR/tokenizer.json" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "⏳ BGE-Reranker 模型未找到，正在从 $HF_ENDPOINT 下载..."
    export HF_ENDPOINT
    pdm run python -c "
import os
os.environ['HF_ENDPOINT'] = '$HF_ENDPOINT'
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.bin', '*.msgpack', '*.h5', 'onnx/*'])
"
    echo "✅ BGE-Reranker 模型下载完成"
else
    echo "✅ BGE-Reranker 模型已存在，跳过下载"
fi

exec pdm run uvicorn server.main:app --host 0.0.0.0 --port 8080
