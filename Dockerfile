FROM python:3.11-slim

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /app

# apt 用阿里云公网镜像（https 有证书），pypi 同理
RUN sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PDM
RUN pip install pdm -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com --no-cache-dir
RUN pdm config pypi.url "https://mirrors.aliyun.com/pypi/simple/"

# 依赖文件
COPY pyproject.toml pdm.lock ./

# CPU torch + 全部依赖
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && PIP_TRUSTED_HOST=mirrors.aliyun.com pdm install --prod --no-self --without dev \
    && rm -rf /root/.cache/pip /root/.cache/pdm /tmp/*

# 预下载 BGE-Reranker 模型到 /app/bge_reranker_cache（build 阶段一次搞定，运行时不用下载）
RUN pip install huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com --no-cache-dir \
    && mkdir -p /app/bge_reranker_cache/BAAI/bge-reranker-base \
    && HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-reranker-base \
    --local-dir /app/bge_reranker_cache/BAAI/bge-reranker-base \
    --local-dir-use-symlinks False \
    && rm -rf /root/.cache/pip /root/.cache/huggingface /tmp/*

# 应用代码
COPY . .

EXPOSE 8080
CMD ["pdm", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
