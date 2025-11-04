# BUILD
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# dependency files
COPY pyproject.toml uv.lock Makefile ./

# install dependencies
RUN make sync-prod

COPY . .

RUN make run-migrations pull-data

# RUNTIME
FROM python:3.13-slim

WORKDIR /app

RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# copy venv and modules
COPY --from=builder --chown=appuser:appuser /app/.venv .venv

# copy src files
COPY --from=builder --chown=appuser:appuser /app/src src
COPY --from=builder --chown=appuser:appuser /app/models models
COPY --from=builder --chown=appuser:appuser /app/static static
COPY --from=builder --chown=appuser:appuser /app/logs logs
COPY --from=builder --chown=appuser:appuser /app/src src
COPY --from=builder --chown=appuser:appuser /app/.env .env

EXPOSE 8000

CMD [".venv/bin/python", "-m", "src.interfaces.api.v1"]
