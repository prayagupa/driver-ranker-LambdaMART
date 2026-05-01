"""Batch feature computation from historical trip data."""

import pandas as pd
from sqlalchemy import create_engine

from feature_engine.config import POSTGRES_URL


def compute_driver_features(pg_url: str = POSTGRES_URL) -> pd.DataFrame:
    """Compute historical driver features from trips table."""
    engine = create_engine(pg_url)
    query = """
    SELECT
        driver_id,
        AVG(CASE WHEN accepted THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE created_at > NOW() - INTERVAL '7 days')
            AS driver_acceptance_rate_7d,
        AVG(rating)
            FILTER (WHERE created_at > NOW() - INTERVAL '30 days')
            AS driver_avg_rating_30d,
        AVG(CASE WHEN cancelled_by_driver THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE created_at > NOW() - INTERVAL '7 days')
            AS driver_cancel_rate_7d,
        COUNT(*) AS driver_trips_lifetime
    FROM trips
    GROUP BY driver_id
    """
    return pd.read_sql(query, engine)


def compute_rider_features(pg_url: str = POSTGRES_URL) -> pd.DataFrame:
    """Compute historical rider features from trips table."""
    engine = create_engine(pg_url)
    query = """
    SELECT
        rider_id,
        AVG(rider_rating_given) AS rider_avg_rating_given,
        AVG(CASE WHEN cancelled_by_rider THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE created_at > NOW() - INTERVAL '30 days')
            AS rider_cancel_rate_30d,
        AVG(tip_pct) AS rider_avg_tip_pct
    FROM trips
    GROUP BY rider_id
    """
    return pd.read_sql(query, engine)


def compute_pair_features(pg_url: str = POSTGRES_URL) -> pd.DataFrame:
    """Compute rider-driver pair interaction features."""
    engine = create_engine(pg_url)
    query = """
    SELECT
        rider_id,
        driver_id,
        COUNT(*) AS pair_trip_count
    FROM trips
    WHERE completed = true
    GROUP BY rider_id, driver_id
    """
    return pd.read_sql(query, engine)


def compute_driver_features_from_df(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Compute driver features from an in-memory DataFrame (for testing)."""
    grouped = trips_df.groupby("driver_id").agg(
        driver_acceptance_rate_7d=("accepted", "mean"),
        driver_avg_rating_30d=("rating", "mean"),
        driver_cancel_rate_7d=("cancelled_by_driver", "mean"),
        driver_trips_lifetime=("trip_id", "count"),
    ).reset_index()
    return grouped


def compute_rider_features_from_df(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rider features from an in-memory DataFrame (for testing)."""
    grouped = trips_df.groupby("rider_id").agg(
        rider_avg_rating_given=("rider_rating_given", "mean"),
        rider_cancel_rate_30d=("cancelled_by_rider", "mean"),
        rider_avg_tip_pct=("tip_pct", "mean"),
    ).reset_index()
    return grouped
