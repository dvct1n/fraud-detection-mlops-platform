import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "real-time-fraud-detection-platform", "version": "1.0.0", "random_state": 42},
    "data": {
        "train_path": "data/raw/fraudTrain.csv",
        "test_path": "data/raw/fraudTest.csv",
        "sample_fraction": 0.20,
        "test_size": 0.20,
        "target_column": "is_fraud",
        "drop_columns": [
            "Unnamed: 0", "first", "last", "street", "city", "state", "zip",
            "dob", "cc_num", "trans_num", "unix_time",
            "lat", "long", "merch_lat", "merch_long", "trans_date_trans_time"
        ],
    },
    "features": {
        "categorical_columns": ["merchant", "category", "gender", "job", "age_group", "distance_group"],
        "numerical_columns": ["amt", "city_pop", "age", "day", "month", "year", "hour", "minute", "distance_km"],
    },
    "model_training": {
        "mlflow_tracking_uri": "http://localhost:5000",
        "experiment_name": "Fraud_Detection_Production_Benchmark",
        "decision_threshold": 0.50,
        "high_risk_threshold": 0.75,
        "medium_risk_threshold": 0.35,
        "hyperparameters": {
            "logistic_regression": {"solver": "lbfgs", "penalty": "l2", "class_weight": "balanced", "max_iter": 1000},
            "xgboost": {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"},
            "lightgbm": {"objective": "binary", "class_weight": "balanced", "n_estimators": 250, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31},
        },
    },
    "artifacts": {
        "encoder_path": "models/ordinal_encoder.pkl",
        "scaler_path": "models/standard_scaler.pkl",
        "model_path": "models/best_fraud_model.pkl",
        "metadata_path": "models/model_metadata.json",
    },
}


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if path.exists() and YAML_AVAILABLE:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    return DEFAULT_CONFIG
