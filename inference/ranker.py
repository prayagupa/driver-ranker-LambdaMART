"""ONNX-based ranking model for inference."""

import numpy as np
from pathlib import Path


class ONNXRanker:
    """Loads and runs the ONNX ranking model."""

    def __init__(self, model_path: str = "model/ranker.onnx"):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score N candidates. features shape: (N, n_features). Returns (N,) scores."""
        result = self.session.run(None, {self.input_name: features.astype(np.float32)})
        return result[0].flatten()


class LightGBMRanker:
    """Loads and runs a native LightGBM model (alternative to ONNX)."""

    def __init__(self, model_path: str = "model/ranker.txt"):
        import lightgbm as lgb

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = lgb.Booster(model_file=model_path)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score N candidates. features shape: (N, n_features). Returns (N,) scores."""
        return self.model.predict(features).flatten()
