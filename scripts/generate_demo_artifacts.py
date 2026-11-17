"""
Utility script to synthesize a realistic credit card transaction sample and pre-build
serialized model artifacts (`ordinal_encoder.pkl`, `standard_scaler.pkl`, `best_fraud_model.pkl`)
so the FastAPI service and unit tests can run immediately out-of-the-box.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.train import run_training_pipeline

def create_synthetic_transactions(output_path: str = "data/raw/sample_transactions.csv", n_samples: int = 5000):
    rng = np.random.default_rng(42)
    merchants = [
        "fraud_Rippin, Kub and Mann",
        "fraud_Heller, Gutmann and Zieme",
        "fraud_Lind-Buckridge",
        "fraud_Kutch, Hermiston and Farrell",
        "fraud_Keeling-Crist",
    ]
    categories = ["misc_net", "grocery_pos", "entertainment", "gas_transport", "shopping_net", "misc_pos"]
    jobs = [
        "Psychologist, counselling",
        "Special educational needs teacher",
        "Nature conservation officer",
        "Patent attorney",
        "Software engineer",
    ]
    genders = ["F", "M"]

    is_fraud = rng.choice([0, 1], size=n_samples, p=[0.96, 0.04])
    amt = np.where(
        is_fraud == 1,
        rng.uniform(350.0, 1800.0, size=n_samples),
        rng.exponential(scale=65.0, size=n_samples) + 1.5,
    )

    lat = rng.uniform(28.0, 48.0, size=n_samples)
    long_ = rng.uniform(-120.0, -72.0, size=n_samples)
    merch_lat = lat + np.where(is_fraud == 1, rng.uniform(0.5, 1.5, size=n_samples), rng.uniform(-0.2, 0.2, size=n_samples))
    merch_long = long_ + np.where(is_fraud == 1, rng.uniform(0.5, 1.5, size=n_samples), rng.uniform(-0.2, 0.2, size=n_samples))

    hours = np.where(is_fraud == 1, rng.choice([0, 1, 2, 3, 23], size=n_samples), rng.integers(6, 22, size=n_samples))
    days = rng.integers(1, 28, size=n_samples)
    months = rng.integers(1, 13, size=n_samples)
    years = rng.choice([2019, 2020], size=n_samples)
    minutes = rng.integers(0, 60, size=n_samples)

    timestamps = [
        f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{mi:02d}:00"
        for y, m, d, h, mi in zip(years, months, days, hours, minutes)
    ]
    birth_years = rng.integers(1950, 2003, size=n_samples)
    dobs = [f"{by:04d}-05-15" for by in birth_years]

    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n_samples),
            "trans_date_trans_time": timestamps,
            "cc_num": rng.integers(10**15, 9 * 10**15, size=n_samples),
            "merchant": rng.choice(merchants, size=n_samples),
            "category": rng.choice(categories, size=n_samples),
            "amt": np.round(amt, 2),
            "first": "Alex",
            "last": "Morgan",
            "gender": rng.choice(genders, size=n_samples),
            "street": "100 Main St",
            "city": "Metro",
            "state": "NY",
            "zip": 10001,
            "lat": np.round(lat, 4),
            "long": np.round(long_, 4),
            "city_pop": rng.integers(150, 250000, size=n_samples),
            "job": rng.choice(jobs, size=n_samples),
            "dob": dobs,
            "trans_num": [f"tx_{i:08d}" for i in range(n_samples)],
            "unix_time": 1325376000 + np.arange(n_samples) * 60,
            "merch_lat": np.round(merch_lat, 6),
            "merch_long": np.round(merch_long, 6),
            "is_fraud": is_fraud,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)

if __name__ == "__main__":
    sample_csv = create_synthetic_transactions()
    run_training_pipeline(
    config_path="config/config.yaml",
    train_override=sample_csv,
    test_override=sample_csv,
)
