"""Feature hydration for inference - fetches features from online store."""

import math
import numpy as np
from datetime import datetime

from feature_engine.store.online import OnlineFeatureStore
from feature_engine.config import MODEL_FEATURE_ORDER
from inference.schemas import MatchRequest


class FeatureHydrator:
    """Fetches and assembles feature vectors for model scoring."""

    def __init__(self, store: OnlineFeatureStore):
        self.store = store

    def hydrate(self, request: MatchRequest) -> np.ndarray:
        """Build feature matrix (N_candidates x N_features) for scoring.

        Fetches driver features in batch, rider features once,
        and computes contextual features inline.
        """
        driver_ids = [c.driver_id for c in request.candidates]
        driver_features = self.store.get_multi("driver", driver_ids)
        rider_features = self.store.get_features("rider", request.ride.rider_id)

        # Contextual (compute once)
        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)

        rows = []
        for i, cand in enumerate(request.candidates):
            df = driver_features[i]
            row = [
                cand.eta_seconds,
                df.get("driver_idle_duration_s", 0.0),
                df.get("driver_acceptance_rate_7d", 0.5),
                df.get("driver_avg_rating_30d", 4.0),
                df.get("driver_cancel_rate_7d", 0.05),
                rider_features.get("rider_cancel_rate_30d", 0.05),
                rider_features.get("rider_avg_tip_pct", 0.1),
                request.ride.surge_multiplier,
                hour_sin,
                hour_cos,
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float32)
