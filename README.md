# Real-Time Fraud Detection & MLOps Platform

An end-to-end machine learning engineering platform designed to detect fraudulent credit card transactions on highly imbalanced financial datasets. The system encompasses automated spatial-temporal feature engineering, multi-algorithm benchmarking with **MLflow** experiment tracking, low-latency REST inference via **FastAPI**, and containerized deployment using **Docker**.

---

## System Architecture

```text
+-----------------------+     +----------------------------+     +----------------------------+
| Raw Transaction Data  | --> | Spatial & Temporal Feature | --> | Stratified Split, Ordinal  |
| (Historical / Stream) |     | Engineering (Haversine/Age)|     | Encoding & StandardScaler  |
+-----------------------+     +----------------------------+     +----------------------------+
                                                                               |
                                                                               v
+-----------------------+     +----------------------------+     +----------------------------+
| Dockerized FastAPI    | <-- | Model Registry & Artifacts | <-- | Multi-Model Training       |
| Real-Time Scoring API |     | (.pkl + Metadata Schema)   |     | (LogReg, XGBoost, LightGBM)|
+-----------------------+     +----------------------------+     +----------------------------+
```

---

## Key Capabilities

* **Spatial & Temporal Feature Engineering:** Computes great-circle distance (`distance_km`) between cardholder coordinates and merchant locations via the Haversine formula, demographic age segmentation (`age_group`), distance bucketing (`distance_group`), and cyclical transaction timestamps.
* **Imbalanced Classification Pipeline:** Handles severe class imbalance (<1% positive fraud rate) using cost-sensitive learning (`scale_pos_weight` in XGBoost and `class_weight='balanced'` in LightGBM and Logistic Regression) combined with stratified partitioning.
* **Experiment Tracking & Versioning:** Logs hyperparameters, evaluation metrics (`Precision`, `Recall`, `F1-Score`, `ROC-AUC`), and serialized model artifacts into **MLflow**.
* **Real-Time Inference Service:** Exposes `/predict`, `/predict/batch`, `/model/info`, and `/health` REST endpoints powered by **FastAPI** and **Pydantic v2** schema validation.
* **Production Containerization:** Packaged with a multi-layer `Dockerfile` and `docker-compose.yml` orchestrating both the inference service and an MLflow tracking server.

---

## Repository Structure

```text
fraud-detection-mlops-platform/
├── .github/workflows/
│   └── ci.yml                              # Automated CI pipeline (Artifact build + Pytest)
├── config/
│   └── config.yaml                         # Centralized pipeline, feature & model configuration
├── data/
│   ├── raw/                                # Raw transaction datasets
│   └── processed/                          # Processed feature matrices
├── models/
│   ├── ordinal_encoder.pkl                 # Fitted OrdinalEncoder with unknown category handling
│   ├── standard_scaler.pkl                 # Fitted StandardScaler for linear baselines
│   ├── best_fraud_model.pkl                # Champion gradient boosted model
│   └── model_metadata.json                 # Active model metadata and feature contract
├── notebooks/
│   └── fraud_detection_experiments.ipynb   # Research & benchmarking notebook
├── scripts/
│   └── generate_demo_artifacts.py          # Synthetic data generator & initial artifact builder
├── src/
│   ├── api/
│   │   ├── app.py                          # FastAPI application & routing
│   │   └── schemas.py                      # Pydantic request/response contracts
│   ├── data/
│   │   └── dataset.py                      # Data loading & stratified splitting utilities
│   ├── features/
│   │   └── engineering.py                  # Haversine distance, age & temporal transformers
│   ├── models/
│   │   └── train.py                        # Training orchestrator with MLflow integration
│   └── utils/
│       ├── config_loader.py                # YAML configuration parser
│       └── logger.py                       # Structured application logger
├── tests/
│   └── test_platform.py                    # Unit & integration test suite
├── Dockerfile                              # Production container image definition
├── docker-compose.yml                      # Multi-container orchestration (API + MLflow UI)
├── Makefile                                # Developer automation commands
└── requirements.txt                        # Pinned Python dependencies
```

---

## Quickstart Guide

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Training & MLflow Tracking
To train on the full Kaggle Fraud Detection dataset, place `fraudTrain.csv` inside `data/raw/` and execute:
```bash
PYTHONPATH=. python -m src.models.train --config config/config.yaml --data data/raw/fraudTrain.csv
```
*(Note: Pre-trained artifacts generated from `scripts/generate_demo_artifacts.py` are already included in `models/` so the API can be launched immediately).*

To inspect experiment runs and compare metrics visually:
```bash
mlflow ui --port 5000
```

### 3. Launch the Real-Time Inference API
```bash
PYTHONPATH=. uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Swagger UI:** `http://localhost:8000/docs`
* **Health Check Endpoint:** `http://localhost:8000/health`

### 4. Sample API Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2025-07-28 02:15:30",
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "amt": 845.50,
    "gender": "F",
    "city_pop": 3495,
    "job": "Psychologist, counselling",
    "dob": "1988-03-09",
    "lat": 36.0788,
    "long": -81.1781,
    "merch_lat": 36.9112,
    "merch_long": -82.9483
  }'
```

**Sample JSON Response:**
```json
{
  "fraud_probability": 0.9412,
  "is_fraud": true,
  "risk_tier": "HIGH",
  "recommended_action": "BLOCK_TRANSACTION",
  "computed_distance_km": 182.46,
  "customer_age": 37,
  "model_version": "LightGBM"
}
```

---

## Docker Deployment

Build and start both the FastAPI inference container and the MLflow tracking server:
```bash
docker compose up --build -d
```
Run the automated test suite:
```bash
PYTHONPATH=. pytest tests/ -v
```
