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

COPY pyproject.toml pdm.lock ./

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && PIP_TRUSTED_HOST=mirrors.aliyun.com pdm install --prod --no-self --without dev \
    && rm -rf /root/.cache/pip /root/.cache/pdm /tmp/*

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
