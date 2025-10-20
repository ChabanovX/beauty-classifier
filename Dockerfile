FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

RUN apt update && apt install -y make

COPY pyproject.toml .

COPY uv.lock .

RUN uv venv && . .venv/bin/activate && uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["make", "setup-prod", "run-prod"]
