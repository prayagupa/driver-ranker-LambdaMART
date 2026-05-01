"""Real-time feature computation using sliding window aggregation."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class DriverState:
    lat: float = 0.0
    lng: float = 0.0
    speed_kmh: float = 0.0
    status: str = "idle"
    last_update: float = 0.0


class DriverRealtimeAggregator:
    """In-memory sliding-window aggregator for driver real-time features.

    This serves as a local replacement for Flink. In production,
    this logic runs as a Flink stateful function.
    """

    def __init__(self, window_seconds: int = 300):
        self.window = window_seconds
        self.events: dict[str, deque] = defaultdict(deque)
        self.state: dict[str, DriverState] = {}

    def ingest(self, driver_id: str, event: dict):
        """Ingest a GPS/status event for a driver."""
        now = time.time()
        self.events[driver_id].append((now, event))
        self._evict(driver_id, now)

        # Update latest state
        state = self.state.get(driver_id, DriverState())
        state.lat = event.get("lat", state.lat)
        state.lng = event.get("lng", state.lng)
        state.speed_kmh = event.get("speed_kmh", state.speed_kmh)
        state.status = event.get("status", state.status)
        state.last_update = now
        self.state[driver_id] = state

    def get_features(self, driver_id: str) -> dict:
        """Get all real-time features for a driver."""
        return {
            "driver_idle_duration_s": self.get_idle_duration(driver_id),
            "driver_speed_kmh": self.get_speed(driver_id),
        }

    def get_idle_duration(self, driver_id: str) -> float:
        """Seconds since driver became idle. 0 if not idle."""
        state = self.state.get(driver_id)
        if not state:
            return float("inf")
        if state.status == "idle":
            return time.time() - state.last_update
        return 0.0

    def get_speed(self, driver_id: str) -> float:
        """Current speed in km/h."""
        state = self.state.get(driver_id)
        if not state:
            return 0.0
        return state.speed_kmh

    def get_location(self, driver_id: str) -> tuple[float, float]:
        """Current (lat, lng)."""
        state = self.state.get(driver_id)
        if not state:
            return (0.0, 0.0)
        return (state.lat, state.lng)

    def _evict(self, driver_id: str, now: float):
        q = self.events[driver_id]
        while q and (now - q[0][0]) > self.window:
            q.popleft()
