"""Feature definitions and metadata registry."""

import os

# Infrastructure
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://dev:dev@localhost:5432/featurestore")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# Feature definitions
FEATURES = {
    # Real-time (computed from streams, freshness < 2s)
    "driver_eta_seconds": {"type": "float", "source": "realtime", "window": None, "default": 300.0},
    "driver_idle_duration_s": {"type": "float", "source": "realtime", "window": "5m", "default": 0.0},
    "driver_speed_kmh": {"type": "float", "source": "realtime", "window": None, "default": 0.0},

    # Batch (materialized hourly)
    "driver_acceptance_rate_7d": {"type": "float", "source": "batch", "window": "7d", "default": 0.5},
    "driver_avg_rating_30d": {"type": "float", "source": "batch", "window": "30d", "default": 4.0},
    "driver_cancel_rate_7d": {"type": "float", "source": "batch", "window": "7d", "default": 0.05},
    "driver_trips_lifetime": {"type": "int", "source": "batch", "window": None, "default": 0},
    "rider_avg_rating_given": {"type": "float", "source": "batch", "window": None, "default": 4.0},
    "rider_cancel_rate_30d": {"type": "float", "source": "batch", "window": "30d", "default": 0.05},
    "rider_avg_tip_pct": {"type": "float", "source": "batch", "window": None, "default": 0.1},
    "pair_trip_count": {"type": "int", "source": "batch", "window": None, "default": 0},

    # Contextual
    "hour_sin": {"type": "float", "source": "contextual", "default": 0.0},
    "hour_cos": {"type": "float", "source": "contextual", "default": 1.0},
    "day_of_week": {"type": "int", "source": "contextual", "default": 0},
    "surge_multiplier": {"type": "float", "source": "request", "default": 1.0},
}

# Feature vector order for model input
MODEL_FEATURE_ORDER = [
    "driver_eta_seconds",
    "driver_idle_duration_s",
    "driver_acceptance_rate_7d",
    "driver_avg_rating_30d",
    "driver_cancel_rate_7d",
    "rider_cancel_rate_30d",
    "rider_avg_tip_pct",
    "surge_multiplier",
    "hour_sin",
    "hour_cos",
]
