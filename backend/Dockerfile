FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev git ca-certificates libmagic1 \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

ENV PATH="/root/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

COPY . .

RUN uv sync --frozen --no-cache --no-dev

EXPOSE 8000

CMD ["uv", "run", "--", "uvicorn", "docint.app:app", "--host", "0.0.0.0", "--port", "8000"]
