FROM python:3.11-slim

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /app

RUN sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pdm -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com --no-cache-dir
RUN pdm config pypi.url "https://mirrors.aliyun.com/pypi/simple/"
RUN pdm config python.use_venv false

COPY pyproject.toml pdm.lock ./
# PDM 用 PEP 582 模式，包都在 __pypackages__/
# torch 不在 pyproject.toml 里 → PDM 不装 → 零 CUDA；事后 --target 到 __pypackages__ 即可
RUN PIP_TRUSTED_HOST=mirrors.aliyun.com pdm install --prod --no-self --without dev \
    && pip install "torch>=2.0" \
        --target /app/__pypackages__/3.11/lib \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --trusted-host download.pytorch.org --no-cache-dir \
    && rm -rf /root/.cache/pip /root/.cache/pdm /tmp/*

COPY . .
RUN rm -rf /app/chroma_db /app/chat_logs /app/*.pdf \
    && chmod +x /app/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
