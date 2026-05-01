"""Configuration for inference service."""

import os

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "model/ranker.txt")
MODEL_FORMAT = os.getenv("MODEL_FORMAT", "lightgbm")  # "onnx" or "lightgbm"
MODEL_VERSION = os.getenv("MODEL_VERSION", "lambdamart_v1")

# Feature Store
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Service
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Latency SLA (ms) - fallback to heuristic if exceeded
LATENCY_SLA_MS = float(os.getenv("LATENCY_SLA_MS", "20.0"))
