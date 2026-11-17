import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.features.engineering import transform_single_record
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS = {
    "encoder": None,
    "scaler": None,
    "model": None,
    "metadata": {},
    "config": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config("config/config.yaml")
    ARTIFACTS["config"] = config

    encoder_path = Path(config["artifacts"]["encoder_path"])
    scaler_path = Path(config["artifacts"]["scaler_path"])
    model_path = Path(config["artifacts"]["model_path"])
    metadata_path = Path(config["artifacts"]["metadata_path"])

    if encoder_path.exists() and model_path.exists():
        ARTIFACTS["encoder"] = joblib.load(encoder_path)
        ARTIFACTS["model"] = joblib.load(model_path)
        if scaler_path.exists():
            ARTIFACTS["scaler"] = joblib.load(scaler_path)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                ARTIFACTS["metadata"] = json.load(f)
        logger.info("Successfully loaded serialized model and preprocessing artifacts.")
    else:
        logger.warning(
            "Model artifacts not found in 'models/'. Run `python scripts/generate_demo_artifacts.py` "
            "or `python -m src.models.train` before invoking inference endpoints."
        )
    yield
    ARTIFACTS.clear()


app = FastAPI(
    title="Real-Time Fraud Detection & MLOps Platform",
    description="Production-grade REST API for low-latency transaction fraud risk scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _score_single(tx: TransactionRequest) -> PredictionResponse:
    if ARTIFACTS["model"] is None or ARTIFACTS["encoder"] is None:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not loaded. Ensure artifacts exist in the 'models/' directory.",
        )

    payload = tx.model_dump()
    requires_scaling = ARTIFACTS["metadata"].get("requires_scaling", False)

    features_df = transform_single_record(
        payload=payload,
        encoder=ARTIFACTS["encoder"],
        scaler=ARTIFACTS["scaler"],
        scale_numeric=requires_scaling,
    )

    prob = float(ARTIFACTS["model"].predict_proba(features_df)[0, 1])

    thresholds = ARTIFACTS["config"].get("model_training", {})
    decision_thresh = thresholds.get("decision_threshold", 0.50)
    high_thresh = thresholds.get("high_risk_threshold", 0.75)
    med_thresh = thresholds.get("medium_risk_threshold", 0.35)

    is_fraud = prob >= decision_thresh
    if prob >= high_thresh:
        risk_tier = "HIGH"
        action = "BLOCK_TRANSACTION"
    elif prob >= med_thresh:
        risk_tier = "MEDIUM"
        action = "STEP_UP_AUTHENTICATION"
    else:
        risk_tier = "LOW"
        action = "APPROVE_TRANSACTION"

    return PredictionResponse(
        fraud_probability=round(prob, 4),
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        recommended_action=action,
        computed_distance_km=round(float(features_df["distance_km"].iloc[0]), 2),
        customer_age=int(features_df["age"].iloc[0]),
        model_version=ARTIFACTS["metadata"].get("best_model_name", "LightGBM_v1"),
    )


@app.get("/health", tags=["System"])
def health_check():
    model_ready = ARTIFACTS.get("model") is not None
    return {
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "active_model": ARTIFACTS.get("metadata", {}).get("best_model_name", "uninitialized"),
    }


@app.get("/model/info", tags=["System"])
def model_metadata():
    if not ARTIFACTS.get("metadata"):
        raise HTTPException(status_code=404, detail="Model metadata not available.")
    return ARTIFACTS["metadata"]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(tx: TransactionRequest):
    start_time = time.perf_counter()
    response = _score_single(tx)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        f"Scored transaction | prob={response.fraud_probability:.4f} | "
        f"tier={response.risk_tier} | latency={elapsed_ms:.2f}ms"
    )
    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictionRequest):
    results = [_score_single(tx) for tx in batch.transactions]
    flagged = sum(1 for r in results if r.is_fraud)
    return BatchPredictionResponse(
        total_transactions=len(results),
        flagged_fraud_count=flagged,
        predictions=results,
    )
