"""Online feature store backed by Redis for low-latency serving."""

import json
import logging

import redis

from feature_engine.config import REDIS_URL

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """Redis-backed online feature store with p99 < 5ms read latency."""

    def __init__(self, redis_url: str = REDIS_URL):
        self.r = redis.from_url(redis_url, decode_responses=True)

    def set_features(self, entity_type: str, entity_id: str, features: dict, ttl: int = 3600):
        """Write features for an entity with TTL."""
        key = f"features:{entity_type}:{entity_id}"
        self.r.set(key, json.dumps(features), ex=ttl)

    def get_features(self, entity_type: str, entity_id: str) -> dict:
        """Read features for a single entity."""
        key = f"features:{entity_type}:{entity_id}"
        raw = self.r.get(key)
        if raw is None:
            return {}
        return json.loads(raw)

    def get_multi(self, entity_type: str, entity_ids: list[str]) -> list[dict]:
        """Batch-read features for multiple entities using pipeline."""
        if not entity_ids:
            return []
        pipe = self.r.pipeline()
        for eid in entity_ids:
            pipe.get(f"features:{entity_type}:{eid}")
        results = pipe.execute()
        return [json.loads(r) if r else {} for r in results]

    def delete_features(self, entity_type: str, entity_id: str):
        """Remove features for an entity."""
        key = f"features:{entity_type}:{entity_id}"
        self.r.delete(key)

    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self.r.ping()
        except redis.ConnectionError:
            return False
