"""Heuristic fallback scorer when ML model is unavailable or times out."""

import numpy as np

from inference.schemas import MatchRequest


def heuristic_score(req: MatchRequest) -> np.ndarray:
    """Simple inverse-ETA scoring as fallback.

    Lower ETA = higher score. Used when:
    - Model file not found
    - Model inference exceeds latency SLA
    - Feature store is down
    """
    etas = np.array([c.eta_seconds for c in req.candidates], dtype=np.float32)
    # Inverse ETA with smoothing to avoid div-by-zero
    return 1.0 / (etas + 1.0)
