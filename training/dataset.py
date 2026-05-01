"""Dataset construction with point-in-time correct feature joins."""

import pandas as pd
import numpy as np
from pathlib import Path

from training.labels import build_relevance_label


FEATURE_COLS = [
    "driver_eta_seconds",
    "driver_idle_duration_s",
    "driver_acceptance_rate_7d",
    "driver_avg_rating_30d",
    "driver_cancel_rate_7d",
    "rider_cancel_rate_30d",
    "rider_avg_tip_pct",
    "surge_multiplier",
    "hour_sin",
    "hour_cos",
]


def load_training_data(path: str = "data/training_set.parquet") -> pd.DataFrame:
    """Load and prepare training data."""
    df = pd.read_parquet(path)
    df["label"] = build_relevance_label(df)
    return df


def get_groups(df: pd.DataFrame, group_col: str = "request_id") -> np.ndarray:
    """Get group sizes for LTR training (candidates per request)."""
    return df.groupby(group_col).size().values


def train_test_split_by_request(
    df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by request_id to avoid data leakage."""
    rng = np.random.default_rng(seed)
    request_ids = df["request_id"].unique()
    rng.shuffle(request_ids)
    split_idx = int(len(request_ids) * (1 - test_fraction))
    train_ids = set(request_ids[:split_idx])
    train_df = df[df["request_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[~df["request_id"].isin(train_ids)].reset_index(drop=True)
    return train_df, test_df
