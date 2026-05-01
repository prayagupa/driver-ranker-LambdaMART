"""Seed Redis with sample feature data for local testing."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from feature_engine.store.online import OnlineFeatureStore


def seed():
    store = OnlineFeatureStore()
    rng = np.random.default_rng(42)

    print("Seeding driver features...")
    for i in range(100):
        driver_id = f"d_{i}"
        features = {
            "driver_idle_duration_s": float(rng.exponential(120)),
            "driver_acceptance_rate_7d": float(rng.beta(8, 2)),
            "driver_avg_rating_30d": float(3.5 + rng.beta(5, 2) * 1.5),
            "driver_cancel_rate_7d": float(rng.beta(1, 9)),
            "driver_trips_lifetime": int(rng.integers(10, 5000)),
        }
        store.set_features("driver", driver_id, features, ttl=7200)

    print("Seeding rider features...")
    for i in range(200):
        rider_id = f"r_{i}"
        features = {
            "rider_avg_rating_given": float(3.5 + rng.beta(5, 2) * 1.5),
            "rider_cancel_rate_30d": float(rng.beta(1, 9)),
            "rider_avg_tip_pct": float(rng.exponential(0.12)),
        }
        store.set_features("rider", rider_id, features, ttl=7200)

    print(f"Done! Seeded 100 drivers + 200 riders into Redis.")


if __name__ == "__main__":
    seed()
