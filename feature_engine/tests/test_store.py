"""Tests for online feature store (uses fakeredis for local testing)."""

import json

import pytest

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from feature_engine.store.online import OnlineFeatureStore


@pytest.fixture
def store():
    """Create store with fakeredis backend for testing without real Redis."""
    if not HAS_FAKEREDIS:
        pytest.skip("fakeredis not installed")

    s = OnlineFeatureStore.__new__(OnlineFeatureStore)
    s.r = fakeredis.FakeRedis(decode_responses=True)
    return s


class TestOnlineFeatureStore:
    def test_set_and_get(self, store):
        features = {"driver_acceptance_rate_7d": 0.85, "driver_avg_rating_30d": 4.7}
        store.set_features("driver", "d1", features)
        result = store.get_features("driver", "d1")
        assert result == features

    def test_get_missing_returns_empty(self, store):
        result = store.get_features("driver", "nonexistent")
        assert result == {}

    def test_get_multi(self, store):
        store.set_features("driver", "d1", {"rating": 4.5})
        store.set_features("driver", "d2", {"rating": 4.8})
        results = store.get_multi("driver", ["d1", "d2", "d3"])
        assert results[0] == {"rating": 4.5}
        assert results[1] == {"rating": 4.8}
        assert results[2] == {}

    def test_delete(self, store):
        store.set_features("driver", "d1", {"rating": 4.5})
        store.delete_features("driver", "d1")
        assert store.get_features("driver", "d1") == {}

    def test_overwrite(self, store):
        store.set_features("driver", "d1", {"rating": 4.0})
        store.set_features("driver", "d1", {"rating": 4.9})
        result = store.get_features("driver", "d1")
        assert result["rating"] == 4.9
