"""Reusable retrieval metrics with explicit zero-qrels handling."""

from __future__ import annotations

from math import log2
from typing import Iterable


def retrieval_metrics_at_k(
    ranked_review_ids: Iterable[str], qrels: dict[str, int], k: int
) -> dict[str, float | None]:
    ranked = list(ranked_review_ids)[:k]
    relevant = {review_id: grade for review_id, grade in qrels.items() if grade > 0}
    if not relevant:
        return {"recall": None, "precision": None, "mrr": None, "ndcg": None}
    binary_hits = [1 if review_id in relevant else 0 for review_id in ranked]
    recall = sum(binary_hits) / len(relevant)
    precision = sum(binary_hits) / k
    first = next((index for index, hit in enumerate(binary_hits, start=1) if hit), None)
    mrr = 0.0 if first is None else 1.0 / first
    dcg = sum(
        ((2**qrels.get(review_id, 0) - 1) / log2(index + 1))
        for index, review_id in enumerate(ranked, start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / log2(index + 1) for index, grade in enumerate(ideal, start=1))
    return {"recall": recall, "precision": precision, "mrr": mrr, "ndcg": 0.0 if idcg == 0 else dcg / idcg}
