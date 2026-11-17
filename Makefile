.PHONY: install demo-artifacts train run-api run-mlflow test docker-up docker-down

install:
	pip install -r requirements.txt

demo-artifacts:
	PYTHONPATH=. python scripts/generate_demo_artifacts.py

train:
	PYTHONPATH=. python -m src.models.train --config config/config.yaml

run-api:
	PYTHONPATH=. uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-mlflow:
	mlflow ui --port 5000

test:
	PYTHONPATH=. pytest tests/ -v

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
