"""Offline feature store backed by Postgres for training data and batch features."""

import pandas as pd
from sqlalchemy import create_engine, text

from feature_engine.config import POSTGRES_URL


class OfflineFeatureStore:
    """Postgres-backed offline store for historical feature snapshots."""

    def __init__(self, pg_url: str = POSTGRES_URL):
        self.engine = create_engine(pg_url)

    def save_driver_features(self, df: pd.DataFrame):
        """Persist computed driver features."""
        df.to_sql("driver_features", self.engine, if_exists="replace", index=False)

    def save_rider_features(self, df: pd.DataFrame):
        """Persist computed rider features."""
        df.to_sql("rider_features", self.engine, if_exists="replace", index=False)

    def save_pair_features(self, df: pd.DataFrame):
        """Persist rider-driver pair features."""
        df.to_sql("pair_features", self.engine, if_exists="replace", index=False)

    def get_driver_features(self, driver_id: str) -> dict:
        """Fetch features for a single driver."""
        query = text("SELECT * FROM driver_features WHERE driver_id = :driver_id")
        df = pd.read_sql(query, self.engine, params={"driver_id": driver_id})
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    def get_training_dataset(self, days_back: int = 90) -> pd.DataFrame:
        """Build a point-in-time correct training dataset."""
        query = text("""
            SELECT
                t.trip_id, t.request_id, t.driver_id, t.rider_id,
                t.accepted, t.completed, t.cancelled_by_driver, t.cancelled_by_rider,
                t.rating, t.rider_rating_given, t.tip_pct,
                t.created_at,
                df.driver_acceptance_rate_7d, df.driver_avg_rating_30d,
                df.driver_cancel_rate_7d, df.driver_trips_lifetime,
                rf.rider_avg_rating_given, rf.rider_cancel_rate_30d, rf.rider_avg_tip_pct
            FROM trips t
            LEFT JOIN driver_features df ON t.driver_id = df.driver_id
            LEFT JOIN rider_features rf ON t.rider_id = rf.rider_id
            WHERE t.created_at > NOW() - MAKE_INTERVAL(days => :days_back)
        """)
        return pd.read_sql(query, self.engine, params={"days_back": days_back})
