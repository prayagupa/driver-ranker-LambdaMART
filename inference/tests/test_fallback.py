"""Tests for the fallback scorer."""

import numpy as np
from inference.fallback import heuristic_score
from inference.schemas import MatchRequest, RideRequest, CandidateDriver


class TestHeuristicScore:
    def _make_request(self, etas: list[float]) -> MatchRequest:
        return MatchRequest(
            ride=RideRequest(
                request_id="r1", rider_id="rider_1",
                pickup_lat=0, pickup_lng=0,
                destination_lat=1, destination_lng=1,
            ),
            candidates=[
                CandidateDriver(driver_id=f"d_{i}", eta_seconds=eta, lat=0, lng=0)
                for i, eta in enumerate(etas)
            ],
        )

    def test_lower_eta_gets_higher_score(self):
        req = self._make_request([300, 100, 200])
        scores = heuristic_score(req)
        assert scores[1] > scores[2] > scores[0]

    def test_zero_eta_handled(self):
        req = self._make_request([0, 100])
        scores = heuristic_score(req)
        assert scores[0] > scores[1]
        assert np.isfinite(scores[0])

    def test_returns_correct_shape(self):
        req = self._make_request([60, 120, 180, 240])
        scores = heuristic_score(req)
        assert scores.shape == (4,)
