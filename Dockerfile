FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

RUN sudo apt update && sudo apt install build-essential

COPY pyproject.toml .

RUN uv venv && . .venv/bin/activate && uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["make", "setup-prod", "run-prod"]
