"""Tests for contextual feature computation."""

import math
from datetime import datetime

from feature_engine.transforms.contextual import compute_time_features


class TestTimeFeatures:
    def test_midnight(self):
        dt = datetime(2026, 1, 1, 0, 0)
        features = compute_time_features(dt)
        assert abs(features["hour_sin"] - 0.0) < 1e-6
        assert abs(features["hour_cos"] - 1.0) < 1e-6
        assert features["day_of_week"] == 3  # Thursday

    def test_noon(self):
        dt = datetime(2026, 1, 1, 12, 0)
        features = compute_time_features(dt)
        assert abs(features["hour_sin"] - 0.0) < 1e-6
        assert abs(features["hour_cos"] - (-1.0)) < 1e-6

    def test_6am(self):
        dt = datetime(2026, 1, 1, 6, 0)
        features = compute_time_features(dt)
        assert abs(features["hour_sin"] - 1.0) < 1e-6
        assert abs(features["hour_cos"] - 0.0) < 1e-6

    def test_returns_all_keys(self):
        features = compute_time_features()
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "day_of_week" in features
