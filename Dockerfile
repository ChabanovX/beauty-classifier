FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml .

RUN uv venv && . .venv/bin/activate && uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "-m", "src.interfaces.cli"]