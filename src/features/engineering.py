from typing import Dict, Any, List
import numpy as np
import pandas as pd

FEATURE_ORDER: List[str] = [
    "merchant",
    "category",
    "amt",
    "gender",
    "city_pop",
    "job",
    "age",
    "age_group",
    "day",
    "month",
    "year",
    "hour",
    "minute",
    "distance_km",
    "distance_group",
]

CATEGORICAL_COLS: List[str] = [
    "merchant",
    "category",
    "gender",
    "job",
    "age_group",
    "distance_group",
]

NUMERICAL_COLS: List[str] = [
    "amt",
    "city_pop",
    "age",
    "day",
    "month",
    "year",
    "hour",
    "minute",
    "distance_km",
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in kilometers between customer and merchant locations."""
    earth_radius_km = 6371.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return earth_radius_km * c


def categorize_age(age: int) -> str:
    """Assign demographic age bracket."""
    if age < 18:
        return "Teen"
    elif age < 25:
        return "Young Adult"
    elif age < 35:
        return "Adult"
    elif age < 50:
        return "Middle-aged"
    elif age < 65:
        return "Senior"
    return "Elderly"


def categorize_distance(dist_km: float) -> str:
    """Bucket spatial distance between cardholder and merchant."""
    if dist_km < 0.5:
        return "Very Near"
    elif dist_km < 3.0:
        return "Near"
    elif dist_km < 15.0:
        return "Moderate"
    elif dist_km < 50.0:
        return "Far"
    return "Very Far"


def build_features_dataframe(df: pd.DataFrame, drop_cols: List[str] = None) -> pd.DataFrame:
    """Apply end-to-end feature engineering pipeline on batch transaction dataframe."""
    data = df.copy()

    dob_dt = pd.to_datetime(data["dob"])
    trans_dt = pd.to_datetime(data["trans_date_trans_time"])

    data["age"] = (trans_dt.dt.year - dob_dt.dt.year).astype(int)
    data["age_group"] = data["age"].apply(categorize_age)

    data["day"] = trans_dt.dt.day.astype(int)
    data["month"] = trans_dt.dt.month.astype(int)
    data["year"] = trans_dt.dt.year.astype(int)
    data["hour"] = trans_dt.dt.hour.astype(int)
    data["minute"] = trans_dt.dt.minute.astype(int)

    data["distance_km"] = haversine_distance(
        data["lat"], data["long"], data["merch_lat"], data["merch_long"]
    )
    data["distance_group"] = data["distance_km"].apply(categorize_distance)

    if drop_cols is None:
        drop_cols = [
            "Unnamed: 0", "first", "last", "street", "city", "state", "zip",
            "dob", "cc_num", "trans_num", "unix_time",
            "lat", "long", "merch_lat", "merch_long", "trans_date_trans_time",
        ]
    existing_drop_cols = [c for c in drop_cols if c in data.columns]
    data.drop(columns=existing_drop_cols, inplace=True)

    return data


def transform_single_record(payload: Dict[str, Any], encoder, scaler=None, scale_numeric: bool = False) -> pd.DataFrame:
    """Transform a single real-time transaction payload into model-ready feature vector."""
    trans_dt = pd.to_datetime(payload["trans_date_trans_time"])
    dob_dt = pd.to_datetime(payload["dob"])
    age = int(trans_dt.year - dob_dt.year)
    dist_km = float(
        haversine_distance(
            payload["lat"], payload["long"], payload["merch_lat"], payload["merch_long"]
        )
    )

    row = {
        "merchant": payload["merchant"],
        "category": payload["category"],
        "amt": float(payload["amt"]),
        "gender": payload["gender"],
        "city_pop": int(payload["city_pop"]),
        "job": payload["job"],
        "age": age,
        "age_group": categorize_age(age),
        "day": int(trans_dt.day),
        "month": int(trans_dt.month),
        "year": int(trans_dt.year),
        "hour": int(trans_dt.hour),
        "minute": int(trans_dt.minute),
        "distance_km": dist_km,
        "distance_group": categorize_distance(dist_km),
    }

    df = pd.DataFrame([row])[FEATURE_ORDER]
    df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])

    if scale_numeric and scaler is not None:
        df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

    return df
