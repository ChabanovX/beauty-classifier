# PMLDL Project Backend by Okurki Team

This repo contains a fastapi backend with ml models for the project.

# Repo Structure

```
├── api -- FastAPI app
│   ├── models/ -- ML models in pytorch
│   ├── services/
│   ├── v1/
│   └── __main__.py -- FastAPI app entry point
├── dataset -- Data-related files for dataset analysis and the dataset itself
│    ├── data/ -- Dataset files
│    └── analyze.ipynb -- Jupyter notebook for data analysis
├── models -- Trained ML models in .pt format
├── tests/
```

# Development

## Setup

1. Install [make](https://www.gnu.org/software/make/), [uv](https://www.uvproject.xyz/), [dvc](https://dvc.org/), [docker](https://docs.docker.com/get-docker/), [docker-compose](https://docs.docker.com/compose/install/)

2. Run `make pull-data` to pull the dataset and models from the remote repository

3. Run `make compose-up` to start the backend

To run locally, download dependencies locally `uv sync` as well, and start the app `uv run python -m api`

## Testing

Run `make test` to run the tests
