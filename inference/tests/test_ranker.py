"""Tests for the ranker module."""

import numpy as np
import pytest


class TestLightGBMRanker:
    def test_missing_model_raises(self):
        from inference.ranker import LightGBMRanker
        with pytest.raises(FileNotFoundError):
            LightGBMRanker("/nonexistent/model.txt")


class TestONNXRanker:
    def test_missing_model_raises(self):
        from inference.ranker import ONNXRanker
        with pytest.raises(FileNotFoundError):
            ONNXRanker("/nonexistent/model.onnx")
