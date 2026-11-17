import pandas as pd
from fastapi.testclient import TestClient
from src.api.app import app
from src.features.engineering import (
    categorize_age,
    categorize_distance,
    haversine_distance,
)

def test_haversine_distance_computation():
    dist = haversine_distance(36.0788, -81.1781, 36.011293, -82.048315)
    assert 75.0 <= dist <= 82.0

def test_age_and_distance_bucketing():
    assert categorize_age(16) == "Teen"
    assert categorize_age(28) == "Adult"
    assert categorize_age(70) == "Elderly"
    assert categorize_distance(0.2) == "Very Near"
    assert categorize_distance(10.0) == "Moderate"
    assert categorize_distance(85.0) == "Very Far"

def test_api_health_and_prediction():
    with TestClient(app) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["model_loaded"] is True

        sample_payload = {
            "trans_date_trans_time": "2025-07-28 02:15:30",
            "merchant": "fraud_Rippin, Kub and Mann",
            "category": "misc_net",
            "amt": 950.75,
            "gender": "F",
            "city_pop": 3495,
            "job": "Psychologist, counselling",
            "dob": "1988-03-09",
            "lat": 36.0788,
            "long": -81.1781,
            "merch_lat": 37.2112,
            "merch_long": -82.9483,
        }
        pred_res = client.post("/predict", json=sample_payload)
        assert pred_res.status_code == 200
        body = pred_res.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
