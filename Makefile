ML_MODULE ?= src.infrastructure.ml_models

help:
	@echo Avaliable commands:
	@echo =====================
	@echo setup                 setup project (sync, run-migrations, pull-data)
	@echo run-dev               run project in dev mode (debug messages, sqlite db, reload on code changes)
	@echo runp-rod              run project in production mode (info messages, postgres db, reload off)
	@echo test                  run tests
	@echo compose-up            launch docker compose
	@echo train-attractiveness  train attractiveness model
	@echo eval-attractiveness   evaluate attractiveness model
	@echo help                  show this message
	@echo sync                  sync dependencies
	@echo run-migrations        run migrations
	@echo pull-data             pull data from remote DVC repository

sync:
	uv sync

run-migrations:
	uv run alembic upgrade head

pull-data:
	uv run dvc pull

setup: sync run-migrations pull-data

run:
	uvicorn src.interfaces.api.app:app --reload

migrate:
	uv run alembic revision --autogenerate -m "$(m)"
	@echo "Please edit the generated migration file and then run 'uv run alembic upgrade head'"

lint:
	uv run ruff check --fix && uv run ruff format

test:
	uv run pytest

compose-up:
	docker compose up -d

mlflow-compose-up:
	docker compose -f docker-compose-mlflow.yml up -d

train-attractiveness:
	uv run python -c "from $(ML_MODULE) import attractiveness_model; attractiveness_model.train()"

eval-attractiveness:
	uv run python -c "from $(ML_MODULE) import attractiveness_model; attractiveness_model.evaluate()"
