import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # Khử cảnh báo "possibly unbound" của Pylance
    MLFLOW_AVAILABLE = False

from src.features.engineering import (
    CATEGORICAL_COLS,
    FEATURE_ORDER,
    NUMERICAL_COLS,
    build_features_dataframe,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_and_sample(csv_path: str, sample_frac: float, target_col: str, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 0.0 < sample_frac < 1.0 and len(df) > 20000:
        df = (
            df.groupby(target_col, group_keys=False)
            .apply(lambda x: x.sample(frac=sample_frac, random_state=random_state))
            .reset_index(drop=True)
        )
    return df


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    data_override=None,
    train_override: Optional[str] = None,
    test_override: Optional[str] = None,
):
    config = load_config(config_path)
    random_state = config["project"]["random_state"]
    target_col = config["data"]["target_column"]
    sample_frac = config["data"]["sample_fraction"]
    drop_cols = config["data"]["drop_columns"]

    train_csv = train_override or config["data"]["train_path"]
    test_csv = test_override or config["data"]["test_path"]

    logger.info(f"Loading training dataset from: {train_csv}")
    raw_train_df = _load_and_sample(train_csv, sample_frac, target_col, random_state)
    logger.info(f"Training records loaded: {len(raw_train_df):,} (Fraud: {raw_train_df[target_col].sum():,})")

    logger.info(f"Loading testing dataset from: {test_csv}")
    raw_test_df = _load_and_sample(test_csv, sample_frac, target_col, random_state)
    logger.info(f"Testing records loaded : {len(raw_test_df):,} (Fraud: {raw_test_df[target_col].sum():,})")

    logger.info("Executing spatial-temporal feature engineering pipeline...")
    train_processed = build_features_dataframe(raw_train_df, drop_cols)
    test_processed = build_features_dataframe(raw_test_df, drop_cols)

    X_train = train_processed[FEATURE_ORDER].copy()
    y_train = train_processed[target_col].astype(int)

    X_test = test_processed[FEATURE_ORDER].copy()
    y_test = test_processed[target_col].astype(int)

    logger.info("Fitting OrdinalEncoder and StandardScaler on training partition...")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[CATEGORICAL_COLS] = encoder.fit_transform(X_train[CATEGORICAL_COLS])
    X_test[CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    neg_count = int((y_train == 0).sum())
    pos_count = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = neg_count / pos_count

    hp = config["model_training"]["hyperparameters"]
    candidate_models = {
        "Baseline_LogisticRegression": LogisticRegression(
            **hp["logistic_regression"], random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            **hp["xgboost"],
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            **hp["lightgbm"],
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    if MLFLOW_AVAILABLE and mlflow is not None:
        mlflow.set_experiment(config["model_training"]["experiment_name"])

    benchmark_rows = []
    best_f1 = -1.0
    best_model = None
    best_model_name = ""
    best_metrics = {}

    for model_name, model in candidate_models.items():
        logger.info(f"Training on fraudTrain.csv and evaluating on fraudTest.csv: {model_name}")
        run_ctx = (
            mlflow.start_run(run_name=model_name)
            if (MLFLOW_AVAILABLE and mlflow is not None)
            else nullcontext()
        )
        with run_ctx:
            requires_scaling = model_name == "Baseline_LogisticRegression"
            train_features = X_train_scaled if requires_scaling else X_train
            test_features = X_test_scaled if requires_scaling else X_test

            model.fit(train_features, y_train)
            y_pred = model.predict(test_features)
            y_prob = model.predict_proba(test_features)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
            }
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")

            benchmark_rows.append(
                {
                    "Model": model_name,
                    "Precision": round(metrics["precision"], 4),
                    "Recall": round(metrics["recall"], 4),
                    "F1-Score": round(metrics["f1_score"], 4),
                    "ROC-AUC": round(metrics["roc_auc"], 4),
                    "Accuracy": round(metrics["accuracy"], 4),
                    "TP": int(tp),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TN": int(tn),
                }
            )

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model
                best_model_name = model_name
                best_metrics = metrics

    results_df = pd.DataFrame(benchmark_rows).sort_values(by="F1-Score", ascending=False)
    logger.info("\nModel Benchmark Summary (Evaluated on fraudTest.csv):\n" + results_df.to_string(index=False))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoder, config["artifacts"]["encoder_path"])
    joblib.dump(scaler, config["artifacts"]["scaler_path"])
    joblib.dump(best_model, config["artifacts"]["model_path"])

    metadata = {
        "best_model_name": best_model_name,
        "requires_scaling": best_model_name == "Baseline_LogisticRegression",
        "metrics": {k: round(v, 4) for k, v in best_metrics.items()},
        "feature_order": FEATURE_ORDER,
        "categorical_columns": CATEGORICAL_COLS,
        "numerical_columns": NUMERICAL_COLS,
        "benchmark_table": benchmark_rows,
    }
    with open(config["artifacts"]["metadata_path"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved champion model ({best_model_name}) and preprocessors to {models_dir.resolve()}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection Models on Train/Test CSVs")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--train", type=str, default=None, help="Optional path to fraudTrain.csv")
    parser.add_argument("--test", type=str, default=None, help="Optional path to fraudTest.csv")
    args = parser.parse_args()
    run_training_pipeline(config_path=args.config, train_override=args.train, test_override=args.test)