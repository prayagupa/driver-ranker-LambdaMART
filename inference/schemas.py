"""Request/response schemas for the matching API."""

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
