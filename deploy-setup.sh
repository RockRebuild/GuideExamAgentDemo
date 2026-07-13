#!/bin/bash
# deploy-setup.sh — ECS 部署前置准备脚本
# 在 ECS 上 git clone/pull 之后、docker compose up 之前运行
# 确保所有 volume 挂载所需的宿主机目录和文件都存在
set -e

echo "=== 导游考试 Agent 部署环境检查 ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. 检查关键应用文件 ──
echo ""
echo "[1/4] 检查关键应用文件..."

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
    else
        echo "  ❌ $1 缺失！"
        MISSING_FILES=1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "  ✅ $1/"
    else
        echo "  ⚠️ $1/ 不存在，将创建"
        mkdir -p "$1"
    fi
}

MISSING_FILES=0
check_file "Dockerfile"
check_file "Dockerfile.mcp"
check_file "docker-compose.yml"
check_file ".env"
check_file "entrypoint.sh"
check_file "pyproject.toml"
check_file "pdm.lock"
check_file "prompts/system_prompt.md"
check_file "question_bank.json"
check_dir  "static"
check_dir  "server"

if [ "$MISSING_FILES" = "1" ]; then
    echo ""
    echo "❌ 关键文件缺失！请确保完整上传项目文件到 ECS。"
    echo "   如果用 git clone，请运行：git clone <repo_url> && cd <dir>"
    echo "   如果用 zip 上传，请确保包含上述所有文件。"
    exit 1
fi

# ── 2. 创建运行时需要的空目录/文件 ──
echo ""
echo "[2/4] 创建运行时目录..."

# 这些目录会被 docker-compose volumes 挂载到容器中
# Docker 会自动创建不存在的目录，但权限可能是 root
# 先手动创建确保存在且权限正确
mkdir -p chroma_db
mkdir -p chat_logs
mkdir -p hf_cache
mkdir -p bge_reranker_cache
touch eval_log.jsonl

echo "  ✅ 运行时目录已就绪"

# ── 3. 检查 ChromaDB 向量库 ──
echo ""
echo "[3/4] 检查 ChromaDB 向量库..."

if [ -f "chroma_db/chroma.sqlite3" ]; then
    CHROMA_SIZE=$(du -sh chroma_db 2>/dev/null | cut -f1)
    echo "  ✅ ChromaDB 已存在（大小: $CHROMA_SIZE）"
else
    echo "  ⚠️ ChromaDB 向量库不存在。"
    echo "    需要在本地构建后上传，或首次启动后在容器内构建。"
    echo "    本地构建命令：python scripts/chroma_client.py --import-pdf <your_pdf>"
fi

# ── 4. 检查 BGE-Reranker 模型 ──
echo ""
echo "[4/4] 检查 BGE-Reranker 模型..."

MODEL_DIR="bge_reranker_cache/BAAI/bge-reranker-base"
if [ -f "$MODEL_DIR/tokenizer.json" ] && [ -f "$MODEL_DIR/config.json" ]; then
    MODEL_SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "  ✅ BGE-Reranker 模型已缓存（大小: $MODEL_SIZE）"
else
    echo "  ⚠️ BGE-Reranker 模型未缓存。"
    echo "    容器首次启动时会自动从 HF Mirror 下载（约 1.1GB）。"
    echo "    也可手动预下载：python scripts/download_reranker_model.py"
fi

# ── 完成 ──
echo ""
echo "=== 环境检查完成 ==="
echo ""
echo "下一步：docker compose up -d --build"
