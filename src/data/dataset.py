from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_and_split_data(
    csv_path: str,
    sample_frac: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw dataset, optionally sample, and perform stratified train/test split."""
    logger.info(f"Loading raw transaction data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} total rows.")

    if 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        logger.info(f"Sampled {sample_frac*100:.1f}% of dataset: {len(df):,} rows.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["is_fraud"],
    )
    logger.info(
        f"Split complete -> Train: {len(train_df):,} rows (Fraud: {train_df['is_fraud'].sum():,}) | "
        f"Test: {len(test_df):,} rows (Fraud: {test_df['is_fraud'].sum():,})"
    )
    return train_df, test_df
