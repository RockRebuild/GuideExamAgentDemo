FROM python:3.11-slim

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /app

# 系统依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# 安装 PDM
RUN pip install pdm -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pdm config pypi.url "https://mirrors.aliyun.com/pypi/simple/"

# 拷贝依赖文件
COPY pyproject.toml pdm.lock ./

# 关键：一次性安装所有依赖（包括 torch 和 transformers）
RUN pdm install --prod --no-lock --no-self

# 拷贝模型文件夹（已下载的 BGE-Reranker）
COPY bge_reranker_cache ./bge_reranker_cache

# 拷贝应用代码
COPY . .

EXPOSE 8501
CMD ["pdm", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]