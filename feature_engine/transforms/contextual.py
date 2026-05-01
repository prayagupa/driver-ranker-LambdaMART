"""Contextual feature computation (time encoding, weather, etc.)."""

import math
from datetime import datetime


def compute_time_features(dt: datetime | None = None) -> dict:
    """Compute cyclical time-of-day and day-of-week features."""
    if dt is None:
        dt = datetime.now()

    hour = dt.hour + dt.minute / 60.0
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "day_of_week": dt.weekday(),
    }


def compute_weather_features(lat: float, lng: float) -> dict:
    """Stub for weather feature lookup. Replace with real API in production."""
    # In production: call weather API and cache result for 5 min
    return {
        "weather_condition": "clear",  # clear, rain, snow, fog
        "temperature_c": 20.0,
    }
