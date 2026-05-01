"""Generate synthetic training data for local development and testing."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate(n_requests: int = 10_000, max_candidates: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ride-request data with candidate drivers.

    Each row = one (request, candidate_driver) pair with features and outcome labels.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for req_id in range(n_requests):
        n_cands = rng.integers(5, max_candidates)
        surge = 1.0 + rng.exponential(0.3)
        hour = rng.integers(0, 24)

        for _ in range(n_cands):
            eta = rng.exponential(180)
            acceptance_rate = rng.beta(8, 2)
            driver_rating = 3.5 + rng.beta(5, 2) * 1.5

            # Simulate outcomes correlated with features
            accept_prob = 0.5 + 0.3 * acceptance_rate - 0.001 * eta
            accept_prob = np.clip(accept_prob, 0.1, 0.95)
            accepted = int(rng.random() < accept_prob)

            complete_prob = 0.85 + 0.1 * (driver_rating - 4.0) if accepted else 0.0
            completed = int(rng.random() < np.clip(complete_prob, 0, 1))

            high_rating_prob = 0.4 + 0.2 * (driver_rating - 4.0) if completed else 0.0
            high_rating = int(rng.random() < np.clip(high_rating_prob, 0, 1))

            rows.append({
                "request_id": req_id,
                "driver_id": f"d_{rng.integers(0, 500)}",
                "rider_id": f"r_{rng.integers(0, 2000)}",
                # Features
                "driver_eta_seconds": eta,
                "driver_idle_duration_s": rng.exponential(120),
                "driver_acceptance_rate_7d": acceptance_rate,
                "driver_avg_rating_30d": driver_rating,
                "driver_cancel_rate_7d": rng.beta(1, 9),
                "rider_cancel_rate_30d": rng.beta(1, 9),
                "rider_avg_tip_pct": rng.exponential(0.1),
                "surge_multiplier": surge,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                # Labels
                "accepted": accepted,
                "completed": completed,
                "high_rating": high_rating,
                "rating": (4.0 + rng.random()) if completed else 0.0,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    df = generate()
    output_path = output_dir / "training_set.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Generated {len(df)} rows ({df['request_id'].nunique()} requests) → {output_path}")
