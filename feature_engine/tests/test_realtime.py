"""Tests for real-time feature computation."""

import time

import pytest

from feature_engine.transforms.realtime import DriverRealtimeAggregator


class TestDriverRealtimeAggregator:
    def setup_method(self):
        self.agg = DriverRealtimeAggregator(window_seconds=60)

    def test_ingest_updates_state(self):
        self.agg.ingest("d1", {"lat": 37.77, "lng": -122.42, "speed_kmh": 30.0, "status": "active"})
        loc = self.agg.get_location("d1")
        assert loc == (37.77, -122.42)

    def test_speed_returned_correctly(self):
        self.agg.ingest("d1", {"speed_kmh": 45.0, "status": "active"})
        assert self.agg.get_speed("d1") == 45.0

    def test_idle_duration_when_idle(self):
        self.agg.ingest("d1", {"status": "idle"})
        time.sleep(0.05)
        idle = self.agg.get_idle_duration("d1")
        assert idle >= 0.04  # at least 40ms

    def test_idle_duration_when_active(self):
        self.agg.ingest("d1", {"status": "active"})
        assert self.agg.get_idle_duration("d1") == 0.0

    def test_unknown_driver_returns_defaults(self):
        assert self.agg.get_speed("unknown") == 0.0
        assert self.agg.get_idle_duration("unknown") == float("inf")

    def test_get_features_returns_dict(self):
        self.agg.ingest("d1", {"speed_kmh": 20.0, "status": "idle"})
        features = self.agg.get_features("d1")
        assert "driver_idle_duration_s" in features
        assert "driver_speed_kmh" in features
        assert features["driver_speed_kmh"] == 20.0

    def test_window_eviction(self):
        agg = DriverRealtimeAggregator(window_seconds=0)  # immediate eviction
        agg.ingest("d1", {"speed_kmh": 10.0, "status": "idle"})
        time.sleep(0.01)
        agg.ingest("d1", {"speed_kmh": 20.0, "status": "idle"})
        # Old events should be evicted, only latest remains
        assert len(agg.events["d1"]) == 1
