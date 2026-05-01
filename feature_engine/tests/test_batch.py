"""Tests for batch feature computation (in-memory, no DB required)."""

import pandas as pd
import pytest

from feature_engine.transforms.batch import (
    compute_driver_features_from_df,
    compute_rider_features_from_df,
)


@pytest.fixture
def sample_trips():
    return pd.DataFrame([
        {"trip_id": "t1", "driver_id": "d1", "rider_id": "r1", "accepted": True,
         "completed": True, "cancelled_by_driver": False, "cancelled_by_rider": False,
         "rating": 4.8, "rider_rating_given": 5.0, "tip_pct": 0.15},
        {"trip_id": "t2", "driver_id": "d1", "rider_id": "r2", "accepted": True,
         "completed": True, "cancelled_by_driver": False, "cancelled_by_rider": False,
         "rating": 4.5, "rider_rating_given": 4.0, "tip_pct": 0.10},
        {"trip_id": "t3", "driver_id": "d1", "rider_id": "r1", "accepted": False,
         "completed": False, "cancelled_by_driver": True, "cancelled_by_rider": False,
         "rating": None, "rider_rating_given": None, "tip_pct": 0.0},
        {"trip_id": "t4", "driver_id": "d2", "rider_id": "r1", "accepted": True,
         "completed": True, "cancelled_by_driver": False, "cancelled_by_rider": False,
         "rating": 4.9, "rider_rating_given": 5.0, "tip_pct": 0.20},
    ])


class TestDriverFeatures:
    def test_acceptance_rate(self, sample_trips):
        df = compute_driver_features_from_df(sample_trips)
        d1 = df[df["driver_id"] == "d1"].iloc[0]
        # d1: 2 accepted out of 3
        assert abs(d1["driver_acceptance_rate_7d"] - 2 / 3) < 0.01

    def test_cancel_rate(self, sample_trips):
        df = compute_driver_features_from_df(sample_trips)
        d1 = df[df["driver_id"] == "d1"].iloc[0]
        assert abs(d1["driver_cancel_rate_7d"] - 1 / 3) < 0.01

    def test_trips_lifetime(self, sample_trips):
        df = compute_driver_features_from_df(sample_trips)
        d1 = df[df["driver_id"] == "d1"].iloc[0]
        assert d1["driver_trips_lifetime"] == 3

    def test_multiple_drivers(self, sample_trips):
        df = compute_driver_features_from_df(sample_trips)
        assert len(df) == 2  # d1 and d2


class TestRiderFeatures:
    def test_avg_rating_given(self, sample_trips):
        df = compute_rider_features_from_df(sample_trips)
        r1 = df[df["rider_id"] == "r1"].iloc[0]
        # r1 gave ratings: 5.0, None, 5.0 → mean of non-null handled by pandas
        assert r1["rider_avg_rating_given"] is not None

    def test_cancel_rate(self, sample_trips):
        df = compute_rider_features_from_df(sample_trips)
        r1 = df[df["rider_id"] == "r1"].iloc[0]
        assert r1["rider_cancel_rate_30d"] == 0.0  # r1 never cancelled

    def test_tip_pct(self, sample_trips):
        df = compute_rider_features_from_df(sample_trips)
        r2 = df[df["rider_id"] == "r2"].iloc[0]
        assert abs(r2["rider_avg_tip_pct"] - 0.10) < 0.01
