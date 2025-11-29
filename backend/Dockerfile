FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/root/.cache/uv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libmagic1 \
 && rm -rf /var/lib/apt/lists/*

# Install system dependencies
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-cache --no-dev --no-install-project

# Copy the rest of the application code
COPY . .

# Install the project
RUN uv sync --frozen --no-cache --no-dev

# Expose the application port and define the default command
EXPOSE 8000
CMD ["uv", "run", "--", "uvicorn", "docint.app:app", "--host", "0.0.0.0", "--port", "8000"]
