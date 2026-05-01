"""FastAPI application for rider-driver matching inference."""

import time
import logging

from fastapi import FastAPI

from inference.schemas import MatchRequest, MatchResponse, ScoredDriver
from inference.fallback import heuristic_score
from inference.feature_hydrator import FeatureHydrator
from inference import config
from feature_engine.store.online import OnlineFeatureStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rider-Driver Matching Service",
    description="ML-powered ranking of candidate drivers for ride requests",
    version="1.0.0",
)

# Initialize components
store = OnlineFeatureStore(redis_url=config.REDIS_URL)
hydrator = FeatureHydrator(store)

# Load model (with graceful fallback)
ranker = None
try:
    if config.MODEL_FORMAT == "onnx":
        from inference.ranker import ONNXRanker
        ranker = ONNXRanker(config.MODEL_PATH)
    else:
        from inference.ranker import LightGBMRanker
        ranker = LightGBMRanker(config.MODEL_PATH)
    logger.info(f"Loaded model from {config.MODEL_PATH}")
except FileNotFoundError:
    logger.warning(f"Model not found at {config.MODEL_PATH}, using heuristic fallback")
except Exception as e:
    logger.warning(f"Failed to load model: {e}, using heuristic fallback")


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    """Rank candidate drivers for a ride request."""
    start = time.perf_counter()

    try:
        if ranker is not None:
            features = hydrator.hydrate(req)
            scores = ranker.score(features)
        else:
            scores = heuristic_score(req)
    except Exception as e:
        logger.error(f"Scoring failed: {e}, falling back to heuristic")
        scores = heuristic_score(req)

    # Rank by score descending
    order = scores.argsort()[::-1]
    ranked = [
        ScoredDriver(
            driver_id=req.candidates[i].driver_id,
            score=float(scores[i]),
            rank=rank + 1,
        )
        for rank, i in enumerate(order)
    ]

    latency = (time.perf_counter() - start) * 1000
    return MatchResponse(
        request_id=req.ride.request_id,
        ranked_drivers=ranked,
        model_version=config.MODEL_VERSION,
        latency_ms=round(latency, 2),
    )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_version": config.MODEL_VERSION,
        "model_loaded": ranker is not None,
        "feature_store_connected": store.health_check(),
    }
