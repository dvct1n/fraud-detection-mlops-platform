from typing import List
from pydantic import BaseModel, Field

class TransactionRequest(BaseModel):
    trans_date_trans_time: str = Field(
        ...,
        description="Transaction timestamp in YYYY-MM-DD HH:MM:SS format",
        examples=["2025-07-28 02:15:30"],
    )
    merchant: str = Field(
        ...,
        description="Merchant name",
        examples=["fraud_Rippin, Kub and Mann"],
    )
    category: str = Field(
        ...,
        description="Merchant category code",
        examples=["misc_net"],
    )
    amt: float = Field(
        ...,
        gt=0,
        description="Transaction amount in USD",
        examples=[845.50],
    )
    gender: str = Field(
        ...,
        pattern="^(M|F)$",
        description="Cardholder gender (M or F)",
        examples=["F"],
    )
    city_pop: int = Field(
        ...,
        ge=0,
        description="Cardholder city population",
        examples=[3495],
    )
    job: str = Field(
        ...,
        description="Cardholder occupation",
        examples=["Psychologist, counselling"],
    )
    dob: str = Field(
        ...,
        description="Cardholder date of birth in YYYY-MM-DD format",
        examples=["1988-03-09"],
    )
    lat: float = Field(..., ge=-90.0, le=90.0, examples=[36.0788])
    long: float = Field(..., ge=-180.0, le=180.0, examples=[-81.1781])
    merch_lat: float = Field(..., ge=-90.0, le=90.0, examples=[36.9112])
    merch_long: float = Field(..., ge=-180.0, le=180.0, examples=[-82.9483])


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    recommended_action: str
    computed_distance_km: float
    customer_age: int
    model_version: str


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchPredictionResponse(BaseModel):
    total_transactions: int
    flagged_fraud_count: int
    predictions: List[PredictionResponse]
