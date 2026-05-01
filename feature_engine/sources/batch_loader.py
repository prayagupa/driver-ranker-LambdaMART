"""Batch data loader from Postgres for historical feature computation."""

import pandas as pd
from sqlalchemy import create_engine, text

from feature_engine.config import POSTGRES_URL


class BatchLoader:
    """Loads historical trip data from Postgres for batch feature computation."""

    def __init__(self, pg_url: str = POSTGRES_URL):
        self.engine = create_engine(pg_url)

    def load_trips(self, days_back: int = 30) -> pd.DataFrame:
        """Load trip records for the last N days."""
        query = text("""
            SELECT
                trip_id, driver_id, rider_id,
                accepted, completed,
                cancelled_by_driver, cancelled_by_rider,
                rating, rider_rating_given, tip_pct,
                created_at
            FROM trips
            WHERE created_at > NOW() - MAKE_INTERVAL(days => :days_back)
            ORDER BY created_at
        """)
        return pd.read_sql(query, self.engine, params={"days_back": days_back})

    def load_driver_ids(self) -> list[str]:
        """Get all active driver IDs."""
        query = text("SELECT DISTINCT driver_id FROM trips WHERE created_at > NOW() - INTERVAL '30 days'")
        df = pd.read_sql(query, self.engine)
        return df["driver_id"].tolist()

    def load_rider_ids(self) -> list[str]:
        """Get all active rider IDs."""
        query = text("SELECT DISTINCT rider_id FROM trips WHERE created_at > NOW() - INTERVAL '30 days'")
        df = pd.read_sql(query, self.engine)
        return df["rider_id"].tolist()
