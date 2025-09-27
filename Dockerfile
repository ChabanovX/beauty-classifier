FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency files first
COPY pyproject.toml /app/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Copy application code
COPY . /app

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "api"]