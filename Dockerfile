FROM python:3.11-slim

WORKDIR /app

# 换成阿里云镜像源加速 apt-get 下载
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# 安装 PDM
RUN pip install pdm -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pdm config pypi.url "https://mirrors.aliyun.com/pypi/simple/"

COPY pyproject.toml pdm.lock ./
RUN pdm install --prod --no-lock --no-self

COPY . .

EXPOSE 8501
CMD ["pdm", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]