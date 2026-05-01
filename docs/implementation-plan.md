# Implementation Plan: Rider-Driver Matching ML System

> All components runnable locally via Docker Compose + local Python environment.

---

## Architecture Overview (Local Dev)

```
docker-compose.yml
├── kafka (+ zookeeper)        — event streaming
├── redis                      — online feature store
├── postgres                   — offline feature store / labels
├── flink-jobmanager/taskmanager — stream processing
├── mlflow                     — experiment tracking & model registry
└── inference-service          — model serving (FastAPI)
```

---

## Phase 1: Feature Engineering

### 1.1 Project Structure

```
feature_engine/
├── __init__.py
├── config.py                  # feature definitions & metadata
├── sources/
│   ├── kafka_consumer.py      # consume GPS, trip events
│   └── batch_loader.py        # load from Postgres (historical)
├── transforms/
│   ├── realtime.py            # streaming window aggregates
│   ├── batch.py               # historical feature computation
│   └── contextual.py          # time encoding, weather stub
├── store/
│   ├── online.py              # Redis read/write
│   └── offline.py             # Postgres read/write
├── registry.py                # feature catalog & schema validation
└── tests/
    ├── test_realtime.py
    ├── test_batch.py
    └── test_store.py
```

### 1.2 Feature Definitions (config.py)

```python
FEATURES = {
    # Real-time (computed from streams, freshness < 2s)
    "driver_eta_seconds": {"type": "float", "source": "realtime", "window": None},
    "driver_idle_duration_s": {"type": "float", "source": "realtime", "window": "5m"},
    "driver_speed_kmh": {"type": "float", "source": "realtime", "window": None},

    # Batch (materialized hourly)
    "driver_acceptance_rate_7d": {"type": "float", "source": "batch", "window": "7d"},
    "driver_avg_rating_30d": {"type": "float", "source": "batch", "window": "30d"},
    "driver_cancel_rate_7d": {"type": "float", "source": "batch", "window": "7d"},
    "driver_trips_lifetime": {"type": "int", "source": "batch", "window": None},
    "rider_avg_rating_given": {"type": "float", "source": "batch", "window": None},
    "rider_cancel_rate_30d": {"type": "float", "source": "batch", "window": "30d"},
    "rider_avg_tip_pct": {"type": "float", "source": "batch", "window": None},
    "pair_trip_count": {"type": "int", "source": "batch", "window": None},

    # Contextual
    "hour_sin": {"type": "float", "source": "contextual"},
    "hour_cos": {"type": "float", "source": "contextual"},
    "day_of_week": {"type": "int", "source": "contextual"},
    "surge_multiplier": {"type": "float", "source": "request"},
}
```

### 1.3 Real-Time Feature Computation

```python
# feature_engine/transforms/realtime.py
import time
from collections import defaultdict, deque

class DriverRealtimeAggregator:
    """In-memory sliding-window aggregator (local replacement for Flink)."""

    def __init__(self, window_seconds=300):
        self.window = window_seconds
        self.events = defaultdict(deque)  # driver_id -> deque of (ts, payload)

    def ingest(self, driver_id: str, event: dict):
        now = time.time()
        self.events[driver_id].append((now, event))
        self._evict(driver_id, now)

    def get_idle_duration(self, driver_id: str) -> float:
        records = self.events.get(driver_id, deque())
        if not records:
            return float("inf")
        last_ts, last_event = records[-1]
        if last_event.get("status") == "idle":
            return time.time() - last_ts
        return 0.0

    def get_speed(self, driver_id: str) -> float:
        records = self.events.get(driver_id, deque())
        if not records:
            return 0.0
        return records[-1][1].get("speed_kmh", 0.0)

    def _evict(self, driver_id, now):
        q = self.events[driver_id]
        while q and (now - q[0][0]) > self.window:
            q.popleft()
```

### 1.4 Batch Feature Computation

```python
# feature_engine/transforms/batch.py
import pandas as pd
from sqlalchemy import create_engine

def compute_driver_features(pg_url: str) -> pd.DataFrame:
    engine = create_engine(pg_url)
    query = """
    SELECT
        driver_id,
        AVG(CASE WHEN accepted THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE created_at > NOW() - INTERVAL '7 days') AS driver_acceptance_rate_7d,
        AVG(rating)
            FILTER (WHERE created_at > NOW() - INTERVAL '30 days') AS driver_avg_rating_30d,
        AVG(CASE WHEN cancelled_by_driver THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE created_at > NOW() - INTERVAL '7 days') AS driver_cancel_rate_7d,
        COUNT(*) AS driver_trips_lifetime
    FROM trips
    GROUP BY driver_id
    """
    return pd.read_sql(query, engine)

def compute_rider_features(pg_url: str) -> pd.DataFrame:
    engine = create_engine(pg_url)
    query = """
    SELECT
        rider_id,
        AVG(rider_rating_given) AS rider_avg_rating_given,
        AVG(CASE WHEN cancelled_by_rider THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE created_at > NOW() - INTERVAL '30 days') AS rider_cancel_rate_30d,
        AVG(tip_pct) AS rider_avg_tip_pct
    FROM trips
    GROUP BY rider_id
    """
    return pd.read_sql(query, engine)
```

### 1.5 Online Store Interface

```python
# feature_engine/store/online.py
import json, redis

class OnlineFeatureStore:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.r = redis.from_url(redis_url)

    def set_features(self, entity_type: str, entity_id: str, features: dict, ttl=3600):
        key = f"features:{entity_type}:{entity_id}"
        self.r.set(key, json.dumps(features), ex=ttl)

    def get_features(self, entity_type: str, entity_id: str) -> dict:
        key = f"features:{entity_type}:{entity_id}"
        raw = self.r.get(key)
        return json.loads(raw) if raw else {}

    def get_multi(self, entity_type: str, entity_ids: list[str]) -> list[dict]:
        pipe = self.r.pipeline()
        for eid in entity_ids:
            pipe.get(f"features:{entity_type}:{eid}")
        results = pipe.execute()
        return [json.loads(r) if r else {} for r in results]
```

### 1.6 Local Testing

```bash
# Run unit tests (no infra needed — uses mocks & in-memory stores)
pytest feature_engine/tests/ -v

# Integration test with local Redis + Postgres
docker compose up -d redis postgres
pytest feature_engine/tests/ -v --integration
```

---

## Phase 2: Model Training

### 2.1 Project Structure

```
training/
├── __init__.py
├── dataset.py            # build training dataset with point-in-time joins
├── labels.py             # composite label construction
├── train.py              # LambdaMART training via LightGBM
├── evaluate.py           # NDCG, acceptance-rate lift, fairness checks
├── export.py             # export to ONNX
├── config.yaml           # hyperparams, paths
├── scripts/
│   ├── generate_synthetic_data.py   # for local dev
│   └── run_training.py              # CLI entrypoint
└── tests/
    ├── test_dataset.py
    ├── test_labels.py
    └── test_train.py
```

### 2.2 Synthetic Data Generator (Local Dev)

```python
# training/scripts/generate_synthetic_data.py
import numpy as np, pandas as pd

def generate(n_requests=10_000, max_candidates=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for req_id in range(n_requests):
        n_cands = rng.integers(5, max_candidates)
        for rank in range(n_cands):
            rows.append({
                "request_id": req_id,
                "driver_id": f"d_{rng.integers(0, 500)}",
                "rider_id": f"r_{rng.integers(0, 2000)}",
                "driver_eta_seconds": rng.exponential(180),
                "driver_idle_duration_s": rng.exponential(120),
                "driver_acceptance_rate_7d": rng.beta(8, 2),
                "driver_avg_rating_30d": 3.5 + rng.beta(5, 2) * 1.5,
                "driver_cancel_rate_7d": rng.beta(1, 9),
                "rider_cancel_rate_30d": rng.beta(1, 9),
                "rider_avg_tip_pct": rng.exponential(0.1),
                "surge_multiplier": 1.0 + rng.exponential(0.3),
                "hour_sin": np.sin(2 * np.pi * rng.integers(0, 24) / 24),
                "hour_cos": np.cos(2 * np.pi * rng.integers(0, 24) / 24),
                # Labels
                "accepted": int(rng.random() < 0.75),
                "completed": int(rng.random() < 0.90),
                "high_rating": int(rng.random() < 0.60),
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate()
    df.to_parquet("data/training_set.parquet", index=False)
    print(f"Generated {len(df)} rows")
```

### 2.3 Label Construction

```python
# training/labels.py
import pandas as pd

def build_relevance_label(df: pd.DataFrame) -> pd.Series:
    """Composite relevance: weighted sum of outcome signals."""
    return (
        0.4 * df["accepted"]
        + 0.35 * df["completed"]
        + 0.25 * df["high_rating"]
    )
```

### 2.4 Training Script

```python
# training/train.py
import lightgbm as lgb
import mlflow, mlflow.lightgbm
import pandas as pd
from training.labels import build_relevance_label

FEATURE_COLS = [
    "driver_eta_seconds", "driver_idle_duration_s",
    "driver_acceptance_rate_7d", "driver_avg_rating_30d",
    "driver_cancel_rate_7d", "rider_cancel_rate_30d",
    "rider_avg_tip_pct", "surge_multiplier",
    "hour_sin", "hour_cos",
]

def train(data_path: str = "data/training_set.parquet"):
    df = pd.read_parquet(data_path)
    df["label"] = build_relevance_label(df)

    # Group by request for LTR
    groups = df.groupby("request_id").size().values

    train_data = lgb.Dataset(
        df[FEATURE_COLS], label=df["label"], group=groups
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "n_estimators": 300,
        "verbose": -1,
    }

    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(run_name="lambdamart_v1"):
        model = lgb.train(params, train_data, num_boost_round=300)
        mlflow.lightgbm.log_model(model, "model")
        mlflow.log_params(params)

        # Log feature importance
        importance = dict(zip(FEATURE_COLS, model.feature_importance().tolist()))
        mlflow.log_dict(importance, "feature_importance.json")

    return model
```

### 2.5 Export to ONNX

```python
# training/export.py
import onnxmltools
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from onnxconverter_common import FloatTensorType

def export_to_onnx(model, n_features: int, output_path: str = "model/ranker.onnx"):
    initial_type = [("features", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model, initial_types=initial_type, target_opset=12
    )
    onnxmltools.utils.save_model(onnx_model, output_path)
    print(f"Exported ONNX model to {output_path}")
```

### 2.6 Local Testing

```bash
# Generate data & train (no infra except optional MLflow)
python training/scripts/generate_synthetic_data.py
python training/scripts/run_training.py

# With MLflow UI
docker compose up -d mlflow
mlflow ui  # or visit http://localhost:5000

# Unit tests
pytest training/tests/ -v
```

---

## Phase 3: Inference Service

### 3.1 Project Structure

```
inference/
├── __init__.py
├── app.py                # FastAPI application
├── ranker.py             # ONNX model loader & scorer
├── feature_hydrator.py   # fetch features from online store
├── schemas.py            # request/response Pydantic models
├── fallback.py           # heuristic fallback scorer
├── config.py             # env-based config
├── Dockerfile
└── tests/
    ├── test_app.py       # API integration tests
    ├── test_ranker.py
    └── test_fallback.py
```

### 3.2 Pydantic Schemas

```python
# inference/schemas.py
from pydantic import BaseModel

class RideRequest(BaseModel):
    request_id: str
    rider_id: str
    pickup_lat: float
    pickup_lng: float
    destination_lat: float
    destination_lng: float
    surge_multiplier: float = 1.0

class CandidateDriver(BaseModel):
    driver_id: str
    eta_seconds: float
    lat: float
    lng: float

class MatchRequest(BaseModel):
    ride: RideRequest
    candidates: list[CandidateDriver]

class ScoredDriver(BaseModel):
    driver_id: str
    score: float
    rank: int

class MatchResponse(BaseModel):
    request_id: str
    ranked_drivers: list[ScoredDriver]
    model_version: str
    latency_ms: float
```

### 3.3 ONNX Ranker

```python
# inference/ranker.py
import numpy as np
import onnxruntime as ort

class ONNXRanker:
    def __init__(self, model_path: str = "model/ranker.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score N candidates. features shape: (N, n_features)."""
        result = self.session.run(None, {self.input_name: features.astype(np.float32)})
        return result[0].flatten()
```

### 3.4 Feature Hydrator

```python
# inference/feature_hydrator.py
import numpy as np
from feature_engine.store.online import OnlineFeatureStore
from inference.schemas import MatchRequest

FEATURE_ORDER = [
    "driver_eta_seconds", "driver_idle_duration_s",
    "driver_acceptance_rate_7d", "driver_avg_rating_30d",
    "driver_cancel_rate_7d", "rider_cancel_rate_30d",
    "rider_avg_tip_pct", "surge_multiplier",
    "hour_sin", "hour_cos",
]

class FeatureHydrator:
    def __init__(self, store: OnlineFeatureStore):
        self.store = store

    def hydrate(self, request: MatchRequest) -> np.ndarray:
        driver_ids = [c.driver_id for c in request.candidates]
        driver_features = self.store.get_multi("driver", driver_ids)
        rider_features = self.store.get_features("rider", request.ride.rider_id)

        rows = []
        for i, cand in enumerate(request.candidates):
            df = driver_features[i]
            row = [
                cand.eta_seconds,
                df.get("driver_idle_duration_s", 0),
                df.get("driver_acceptance_rate_7d", 0.5),
                df.get("driver_avg_rating_30d", 4.0),
                df.get("driver_cancel_rate_7d", 0.05),
                rider_features.get("rider_cancel_rate_30d", 0.05),
                rider_features.get("rider_avg_tip_pct", 0.1),
                request.ride.surge_multiplier,
                df.get("hour_sin", 0),
                df.get("hour_cos", 1),
            ]
            rows.append(row)
        return np.array(rows, dtype=np.float32)
```

### 3.5 FastAPI Application

```python
# inference/app.py
import time
from fastapi import FastAPI, HTTPException
from inference.schemas import MatchRequest, MatchResponse, ScoredDriver
from inference.ranker import ONNXRanker
from inference.feature_hydrator import FeatureHydrator
from inference.fallback import heuristic_score
from feature_engine.store.online import OnlineFeatureStore

app = FastAPI(title="Rider-Driver Matching Service")

# Init on startup
ranker = ONNXRanker("model/ranker.onnx")
store = OnlineFeatureStore()
hydrator = FeatureHydrator(store)
MODEL_VERSION = "lambdamart_v1"

@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    start = time.perf_counter()

    try:
        features = hydrator.hydrate(req)
        scores = ranker.score(features)
    except Exception:
        # Fallback to heuristic
        scores = heuristic_score(req)

    # Rank
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
        model_version=MODEL_VERSION,
        latency_ms=round(latency, 2),
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}
```

### 3.6 Heuristic Fallback

```python
# inference/fallback.py
import numpy as np
from inference.schemas import MatchRequest

def heuristic_score(req: MatchRequest) -> np.ndarray:
    """Simple inverse-ETA scoring as fallback."""
    etas = np.array([c.eta_seconds for c in req.candidates])
    # Lower ETA = higher score; avoid div-by-zero
    return 1.0 / (etas + 1.0)
```

### 3.7 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.8 Local Testing

```bash
# Run inference service locally
uvicorn inference.app:app --reload --port 8000

# Or via Docker
docker compose up inference-service

# Hit the endpoint
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "ride": {
      "request_id": "r123",
      "rider_id": "rider_42",
      "pickup_lat": 37.7749,
      "pickup_lng": -122.4194,
      "destination_lat": 37.8044,
      "destination_lng": -122.2712,
      "surge_multiplier": 1.2
    },
    "candidates": [
      {"driver_id": "d_1", "eta_seconds": 120, "lat": 37.77, "lng": -122.42},
      {"driver_id": "d_2", "eta_seconds": 240, "lat": 37.78, "lng": -122.41},
      {"driver_id": "d_3", "eta_seconds": 60, "lat": 37.775, "lng": -122.418}
    ]
  }'

# Unit + integration tests
pytest inference/tests/ -v
```

---

## Phase 4: End-to-End Local Stack

### 4.1 Docker Compose

```yaml
# docker-compose.yml
version: "3.9"
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: featurestore
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    ports: ["5432:5432"]
    volumes:
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      CLUSTER_ID: "local-dev-cluster"
    ports: ["9092:9092"]

  mlflow:
    image: ghcr.io/mlflow/mlflow:2.12.1
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0

  inference-service:
    build: .
    ports: ["8000:8000"]
    environment:
      REDIS_URL: redis://redis:6379
      MODEL_PATH: /app/model/ranker.onnx
    depends_on: [redis]
```

### 4.2 One-Command Local Run

```bash
# Start everything
docker compose up -d

# Seed features into Redis (for testing)
python scripts/seed_features.py

# Generate synthetic data & train model
python training/scripts/generate_synthetic_data.py
python training/scripts/run_training.py

# Export model
python -c "from training.export import export_to_onnx; from training.train import train; m=train(); export_to_onnx(m, 10)"

# Start inference (or use docker compose service)
uvicorn inference.app:app --reload --port 8000

# Run full test suite
pytest --cov=feature_engine --cov=training --cov=inference -v
```

---

## Phase 5: Testing Strategy

| Layer | What | How | Local? |
|-------|------|-----|--------|
| Unit | Individual transforms, label logic, ranker | pytest + mocks | ✅ |
| Integration | Feature store read/write, model load | pytest + testcontainers (Redis, PG) | ✅ |
| Contract | API schema validation | pytest + FastAPI TestClient | ✅ |
| End-to-end | Request → score → ranked response | docker compose + curl / httpx | ✅ |
| Performance | Latency p99 < 20ms for 50 candidates | locust / pytest-benchmark | ✅ |
| Data quality | Feature distributions, null rates | Great Expectations | ✅ |

### Sample Test

```python
# inference/tests/test_app.py
from fastapi.testclient import TestClient
from inference.app import app

client = TestClient(app)

def test_match_returns_ranked_drivers():
    payload = {
        "ride": {
            "request_id": "test_1",
            "rider_id": "r_1",
            "pickup_lat": 37.77, "pickup_lng": -122.42,
            "destination_lat": 37.80, "destination_lng": -122.27,
        },
        "candidates": [
            {"driver_id": "d_1", "eta_seconds": 120, "lat": 37.77, "lng": -122.42},
            {"driver_id": "d_2", "eta_seconds": 60, "lat": 37.775, "lng": -122.418},
        ],
    }
    resp = client.post("/match", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ranked_drivers"]) == 2
    assert data["ranked_drivers"][0]["rank"] == 1
    assert data["latency_ms"] < 100  # generous for test
```

---

## Requirements (requirements.txt)

```
# Feature Engineering
redis>=5.0
sqlalchemy>=2.0
psycopg2-binary>=2.9
pandas>=2.1
confluent-kafka>=2.3

# Training
lightgbm>=4.3
mlflow>=2.12
onnxmltools>=1.12
onnxconverter-common>=1.14
scikit-learn>=1.4
pyarrow>=15.0

# Inference
fastapi>=0.111
uvicorn>=0.29
onnxruntime>=1.17
pydantic>=2.7
numpy>=1.26

# Testing
pytest>=8.1
pytest-cov>=5.0
httpx>=0.27
pytest-benchmark>=4.0
```

---

## Summary

| Phase | Deliverable | Local Test Command |
|-------|------------|-------------------|
| Feature Eng | Streaming + batch features in Redis | `pytest feature_engine/tests/ -v` |
| Training | LambdaMART model + ONNX export | `python training/scripts/run_training.py && pytest training/tests/` |
| Inference | FastAPI service scoring candidates | `uvicorn inference.app:app --reload` + `pytest inference/tests/` |
| Full Stack | All services running together | `docker compose up -d && pytest --cov` |
