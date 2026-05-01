"""Tests for model training (lightweight, fast)."""

import numpy as np
import pandas as pd
import pytest

from training.dataset import FEATURE_COLS
from training.evaluate import ndcg_at_k


class TestNDCG:
    def test_perfect_ranking(self):
        relevance = np.array([3.0, 2.0, 1.0, 0.0])
        assert ndcg_at_k(relevance, k=4) == 1.0

    def test_worst_ranking(self):
        relevance = np.array([0.0, 0.0, 0.0, 3.0])
        score = ndcg_at_k(relevance, k=4)
        assert score < 1.0

    def test_empty(self):
        assert ndcg_at_k(np.array([]), k=5) == 0.0

    def test_single_element(self):
        assert ndcg_at_k(np.array([1.0]), k=5) == 1.0

    def test_k_truncation(self):
        relevance = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 5.0])
        score_k2 = ndcg_at_k(relevance, k=2)
        score_k6 = ndcg_at_k(relevance, k=6)
        # The 5.0 at position 6 only affects k=6
        assert score_k2 != score_k6
