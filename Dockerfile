FROM python:3.11-slim

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /app

# 系统依赖（已移除不需要的 nodejs/npm，节省 ~945MB）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 PDM
RUN pip install pdm -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-cache-dir
RUN pdm config pypi.url "https://mirrors.aliyun.com/pypi/simple/"

# 拷贝依赖文件
COPY pyproject.toml pdm.lock ./

# 先装 CPU 版 torch（比 CUDA 版小 ~2GB，BGE-Reranker 推理用 CPU 足够）
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 安装全部依赖（torch 已满足 → 跳过）
RUN pdm install --prod --no-self \
    && rm -rf /root/.cache/pip /root/.cache/pdm /tmp/*

# bge_reranker_cache 不再 COPY 到镜像（首次请求时自动从 hf-mirror.com 下载）
# 节省 ~1.1GB，模型下载后缓存在容器层或可挂载 volume 固化

# 拷贝应用代码
COPY . .

EXPOSE 8080
CMD ["pdm", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
