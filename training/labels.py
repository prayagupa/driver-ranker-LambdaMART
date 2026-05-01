"""Composite label construction for the ranking model."""

import pandas as pd


def build_relevance_label(df: pd.DataFrame) -> pd.Series:
    """Build composite relevance label from outcome signals.

    Weights:
      - accepted (40%): driver accepted the trip offer
      - completed (35%): trip completed without cancellation
      - high_rating (25%): rider gave ≥ 4.5 stars

    Returns a continuous score in [0, 1].
    """
    return (
        0.4 * df["accepted"].astype(float)
        + 0.35 * df["completed"].astype(float)
        + 0.25 * df["high_rating"].astype(float)
    )


def build_high_rating_flag(df: pd.DataFrame, threshold: float = 4.5) -> pd.Series:
    """Derive binary high-rating flag from raw rating column."""
    return (df["rating"] >= threshold).astype(int)
