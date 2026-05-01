"""Model evaluation metrics for ranking quality."""

import numpy as np
import pandas as pd


def ndcg_at_k(relevance: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@k for a single query's ranked results."""
    relevance = relevance[:k]
    if len(relevance) == 0:
        return 0.0

    dcg = np.sum((2**relevance - 1) / np.log2(np.arange(2, len(relevance) + 2)))
    ideal = np.sort(relevance)[::-1]
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def evaluate_model(model, df: pd.DataFrame, feature_cols: list[str], k: int = 5) -> dict:
    """Evaluate ranking model on a dataset.

    Returns dict with:
      - ndcg@k (averaged over all requests)
      - mean_reciprocal_rank
      - top1_acceptance_rate
    """
    scores = model.predict(df[feature_cols])
    df = df.copy()
    df["score"] = scores

    ndcg_scores = []
    mrr_scores = []
    top1_accepted = []

    for _, group in df.groupby("request_id"):
        group_sorted = group.sort_values("score", ascending=False)
        relevance = group_sorted["label"].values
        ndcg_scores.append(ndcg_at_k(relevance, k))

        # MRR: rank of first accepted driver
        accepted_ranks = np.where(group_sorted["accepted"].values == 1)[0]
        if len(accepted_ranks) > 0:
            mrr_scores.append(1.0 / (accepted_ranks[0] + 1))
        else:
            mrr_scores.append(0.0)

        # Top-1 acceptance
        top1_accepted.append(float(group_sorted.iloc[0]["accepted"]))

    return {
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "top1_acceptance_rate": float(np.mean(top1_accepted)),
    }
