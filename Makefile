.PHONY: help install test train serve clean docker-up docker-down seed

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run all tests
	pytest feature_engine/tests/ training/tests/ inference/tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	pytest --cov=feature_engine --cov=training --cov=inference -v --tb=short

generate-data:  ## Generate synthetic training data
	python training/scripts/generate_synthetic_data.py

train:  ## Train model locally (no MLflow)
	python training/scripts/run_training.py --export-native

train-onnx:  ## Train and export ONNX model
	python training/scripts/run_training.py --export-onnx

serve:  ## Run inference service locally
	uvicorn inference.app:app --reload --port 8000

docker-up:  ## Start all infrastructure
	docker compose up -d

docker-down:  ## Stop all infrastructure
	docker compose down

seed:  ## Seed Redis with sample features
	python scripts/seed_features.py

all: install generate-data train docker-up seed serve  ## Full local setup

clean:  ## Remove generated files
	rm -rf data/ model/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
