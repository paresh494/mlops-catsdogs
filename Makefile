.PHONY: help setup synth-data preprocess train test lint serve docker-build docker-run \
        compose-up compose-down smoke monitor dvc-init mlflow-ui clean

IMAGE ?= mlops-catsdogs:local
PORT  ?= 8000
PY    ?= python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:            ## Install dev dependencies + package (editable)
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install --no-deps -e .

synth-data:       ## Generate a tiny synthetic dataset (no Kaggle needed)
	$(PY) scripts/generate_synthetic_data.py --per-class 150 --out data/raw

preprocess:       ## Resize to 224x224 RGB and split 80/10/10
	$(PY) -m mlops_catsdogs.data.preprocess

train:            ## Train baseline CNN + log to MLflow
	$(PY) -m mlops_catsdogs.train --epochs 5 --batch-size 32

test:             ## Run unit tests
	pytest

lint:             ## Ruff lint
	ruff check .

serve:            ## Run the API locally with reload
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:     ## Build the inference image
	docker build -t $(IMAGE) .

docker-run:       ## Run the inference image
	docker run --rm -p $(PORT):8000 --name catsdogs $(IMAGE)

compose-up:       ## Bring up inference + Prometheus
	IMAGE=$(IMAGE) docker compose -f deploy/docker-compose.yml up -d --build

compose-down:     ## Tear down the compose stack
	docker compose -f deploy/docker-compose.yml down

smoke:            ## Run the post-deploy smoke test
	bash scripts/smoke_test.sh http://localhost:$(PORT)

monitor:          ## Post-deployment performance report against the running service
	$(PY) scripts/monitor_performance.py --base-url http://localhost:$(PORT) --data-dir data/processed/test

dvc-init:         ## Initialise DVC and track the dataset
	dvc init && dvc add data/raw && git add data/raw.dvc data/.gitignore .dvc

mlflow-ui:        ## Open the MLflow tracking UI
	mlflow ui --backend-store-uri ./mlruns

clean:            ## Remove generated artifacts
	rm -rf data/processed/* models/*.pt metrics mlruns .pytest_cache .ruff_cache
