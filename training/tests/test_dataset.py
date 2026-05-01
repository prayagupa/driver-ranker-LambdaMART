"""Tests for dataset utilities."""

import pandas as pd
import numpy as np
import pytest

from training.dataset import get_groups, train_test_split_by_request, FEATURE_COLS


@pytest.fixture
def sample_df():
    """Minimal dataset with 3 requests."""
    rows = []
    for req_id in range(3):
        n_cands = req_id + 2  # 2, 3, 4 candidates
        for i in range(n_cands):
            row = {col: float(i) for col in FEATURE_COLS}
            row["request_id"] = req_id
            row["driver_id"] = f"d_{i}"
            row["rider_id"] = f"r_{req_id}"
            row["accepted"] = 1
            row["completed"] = 1
            row["high_rating"] = 1
            row["label"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


class TestGetGroups:
    def test_group_sizes(self, sample_df):
        groups = get_groups(sample_df)
        assert list(groups) == [2, 3, 4]

    def test_sum_equals_total(self, sample_df):
        groups = get_groups(sample_df)
        assert groups.sum() == len(sample_df)


class TestTrainTestSplit:
    def test_no_request_leakage(self, sample_df):
        train, test = train_test_split_by_request(sample_df, test_fraction=0.5)
        train_ids = set(train["request_id"].unique())
        test_ids = set(test["request_id"].unique())
        assert train_ids.isdisjoint(test_ids)

    def test_all_data_covered(self, sample_df):
        train, test = train_test_split_by_request(sample_df, test_fraction=0.3)
        assert len(train) + len(test) == len(sample_df)

    def test_deterministic(self, sample_df):
        t1, _ = train_test_split_by_request(sample_df, seed=42)
        t2, _ = train_test_split_by_request(sample_df, seed=42)
        assert t1["request_id"].tolist() == t2["request_id"].tolist()
