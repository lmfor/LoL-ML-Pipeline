FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

# Install dependencies into the container
# --frozen ensures we use the exact lockfile versions
RUN uv sync --frozen

# Copy the rest of the project
COPY . .

RUN mkdir -p /app/data /app/models

# ENV PATH="/app/.venv/bin:$PATH"

# Entrypoint for AWS Batch
CMD ["uv", "run", "python", "src/training/models/train.py"]