"""Tests for label construction."""

import pandas as pd
import numpy as np

from training.labels import build_relevance_label, build_high_rating_flag


class TestBuildRelevanceLabel:
    def test_all_positive(self):
        df = pd.DataFrame({"accepted": [1], "completed": [1], "high_rating": [1]})
        label = build_relevance_label(df)
        assert abs(label.iloc[0] - 1.0) < 1e-6

    def test_all_negative(self):
        df = pd.DataFrame({"accepted": [0], "completed": [0], "high_rating": [0]})
        label = build_relevance_label(df)
        assert abs(label.iloc[0] - 0.0) < 1e-6

    def test_partial(self):
        df = pd.DataFrame({"accepted": [1], "completed": [1], "high_rating": [0]})
        label = build_relevance_label(df)
        expected = 0.4 + 0.35
        assert abs(label.iloc[0] - expected) < 1e-6

    def test_weights_sum_to_one(self):
        # When all signals are 1, label should be 1.0
        df = pd.DataFrame({"accepted": [1], "completed": [1], "high_rating": [1]})
        label = build_relevance_label(df)
        assert abs(label.iloc[0] - 1.0) < 1e-6


class TestHighRatingFlag:
    def test_above_threshold(self):
        df = pd.DataFrame({"rating": [4.8, 5.0, 4.5]})
        flags = build_high_rating_flag(df)
        assert flags.tolist() == [1, 1, 1]

    def test_below_threshold(self):
        df = pd.DataFrame({"rating": [3.0, 4.4, 4.0]})
        flags = build_high_rating_flag(df)
        assert flags.tolist() == [0, 0, 0]

    def test_custom_threshold(self):
        df = pd.DataFrame({"rating": [4.0, 4.1]})
        flags = build_high_rating_flag(df, threshold=4.0)
        assert flags.tolist() == [1, 1]
