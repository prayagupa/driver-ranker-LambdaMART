"""Tests for the inference API."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    # Mock Redis to avoid needing a real connection
    with patch("feature_engine.store.online.redis") as mock_redis:
        mock_r = MagicMock()
        mock_redis.from_url.return_value = mock_r
        mock_r.get.return_value = None
        mock_r.pipeline.return_value = mock_r
        mock_r.execute.return_value = [None, None, None]
        mock_r.ping.return_value = True

        from inference.app import app
        yield TestClient(app)


SAMPLE_PAYLOAD = {
    "ride": {
        "request_id": "test_1",
        "rider_id": "r_1",
        "pickup_lat": 37.77,
        "pickup_lng": -122.42,
        "destination_lat": 37.80,
        "destination_lng": -122.27,
        "surge_multiplier": 1.2,
    },
    "candidates": [
        {"driver_id": "d_1", "eta_seconds": 120, "lat": 37.77, "lng": -122.42},
        {"driver_id": "d_2", "eta_seconds": 60, "lat": 37.775, "lng": -122.418},
        {"driver_id": "d_3", "eta_seconds": 240, "lat": 37.78, "lng": -122.41},
    ],
}


class TestMatchEndpoint:
    def test_returns_200(self, client):
        resp = client.post("/match", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200

    def test_returns_all_candidates_ranked(self, client):
        resp = client.post("/match", json=SAMPLE_PAYLOAD)
        data = resp.json()
        assert len(data["ranked_drivers"]) == 3

    def test_ranks_are_sequential(self, client):
        resp = client.post("/match", json=SAMPLE_PAYLOAD)
        data = resp.json()
        ranks = [d["rank"] for d in data["ranked_drivers"]]
        assert ranks == [1, 2, 3]

    def test_response_has_required_fields(self, client):
        resp = client.post("/match", json=SAMPLE_PAYLOAD)
        data = resp.json()
        assert "request_id" in data
        assert "ranked_drivers" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_latency_is_reasonable(self, client):
        resp = client.post("/match", json=SAMPLE_PAYLOAD)
        data = resp.json()
        assert data["latency_ms"] < 500  # generous for test env

    def test_heuristic_prefers_lower_eta(self, client):
        """With fallback heuristic, lower ETA should rank higher."""
        resp = client.post("/match", json=SAMPLE_PAYLOAD)
        data = resp.json()
        # d_2 has lowest ETA (60s), should rank first in heuristic
        assert data["ranked_drivers"][0]["driver_id"] == "d_2"


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestValidation:
    def test_missing_ride_field(self, client):
        payload = {"candidates": SAMPLE_PAYLOAD["candidates"]}
        resp = client.post("/match", json=payload)
        assert resp.status_code == 422

    def test_empty_candidates(self, client):
        payload = {
            "ride": SAMPLE_PAYLOAD["ride"],
            "candidates": [],
        }
        resp = client.post("/match", json=payload)
        assert resp.status_code == 200
        assert resp.json()["ranked_drivers"] == []
