ATTRACTIVENESS-MODULE := api.models
DISABLE-WARNINGS := import warnings; warnings.filterwarnings("ignore")

train-attractiveness:
	uv run python -c \
		"$(DISABLE-WARNINGS); from $(ATTRACTIVENESS-MODULE) import attractiveness_model; \
		attractiveness_model.train()"

eval-attractiveness:
	uv run python -c \
		"$(DISABLE-WARNINGS); from $(ATTRACTIVENESS-MODULE) import attractiveness_model; \
		attractiveness_model.evaluate()"
	
compose-up:
	docker-compose up

pull-data:
	dvc pull

test:
	uv run python -m tests