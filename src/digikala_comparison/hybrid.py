"""Product-scoped Reciprocal Rank Fusion over BM25 and dense evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from .bm25 import ProductScopedBM25, RetrievedReview
from .config import HybridSettings, Settings
from .dense_index import DenseFaissRetriever
from .retrieval_contract import ProductReviewRetriever


class HybridRetrievedReview(RetrievedReview):
    """One evidence item with every available component rank/score preserved."""

    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    fused_score: float
    fused_rank: int


class _MergedCandidate(BaseModel):
    review: RetrievedReview
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None


def reciprocal_rank(rank: int, rrf_k: int) -> float:
    if rank < 1:
        raise ValueError("retrieval ranks must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    return 1.0 / (rrf_k + rank)


def fuse_rrf(
    *,
    product_id: str | int,
    bm25_candidates: Iterable[RetrievedReview],
    dense_candidates: Iterable[RetrievedReview],
    rrf_k: int,
    final_top_k: int,
) -> list[HybridRetrievedReview]:
    """Fuse exactly by stable review ID, never by text coincidence."""
    expected_product_id = str(product_id)
    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive")
    merged: dict[str, _MergedCandidate] = {}

    def add(source: Literal["bm25", "dense"], candidates: Iterable[RetrievedReview]) -> None:
        seen_in_source: set[str] = set()
        for candidate in candidates:
            if str(candidate.product_id) != expected_product_id:
                raise ValueError("cross-product candidate rejected before hybrid fusion")
            review_id = str(candidate.review_id)
            if review_id in seen_in_source:
                raise ValueError(f"duplicate {source} candidate review_id: {review_id}")
            seen_in_source.add(review_id)
            existing = merged.get(review_id)
            if existing is None:
                existing = _MergedCandidate(review=candidate)
                merged[review_id] = existing
            if source == "bm25":
                existing.bm25_score = candidate.score
                existing.bm25_rank = candidate.rank
            else:
                existing.dense_score = candidate.score
                existing.dense_rank = candidate.rank

    add("bm25", bm25_candidates)
    add("dense", dense_candidates)
    ranked: list[tuple[_MergedCandidate, float]] = []
    for candidate in merged.values():
        score = 0.0
        if candidate.bm25_rank is not None:
            score += reciprocal_rank(candidate.bm25_rank, rrf_k)
        if candidate.dense_rank is not None:
            score += reciprocal_rank(candidate.dense_rank, rrf_k)
        ranked.append((candidate, score))
    # Stable lexical review-ID tie break keeps output identical across runs.
    ranked.sort(key=lambda item: (-item[1], str(item[0].review.review_id)))
    output: list[HybridRetrievedReview] = []
    for fused_rank, (candidate, fused_score) in enumerate(ranked[:final_top_k], start=1):
        source = candidate.review
        payload = source.model_dump()
        payload.update(
            {
                "score": fused_score,
                "rank": fused_rank,
                "bm25_score": candidate.bm25_score,
                "bm25_rank": candidate.bm25_rank,
                "dense_score": candidate.dense_score,
                "dense_rank": candidate.dense_rank,
                "fused_score": fused_score,
                "fused_rank": fused_rank,
            }
        )
        output.append(
            HybridRetrievedReview(**payload)
        )
    if any(result.product_id != expected_product_id for result in output):
        raise RuntimeError("cross-product candidate leaked through hybrid fusion")
    return output


class HybridRRFRetriever:
    """Dense + lexical product-scoped retrieval with configured rank fusion."""

    def __init__(
        self,
        bm25: ProductReviewRetriever,
        dense: ProductReviewRetriever,
        settings: HybridSettings,
    ):
        if settings.bm25_candidate_depth <= 0 or settings.dense_candidate_depth <= 0:
            raise ValueError("hybrid candidate depths must be positive")
        if settings.final_top_k <= 0:
            raise ValueError("hybrid final_top_k must be positive")
        if settings.rrf_k < 0:
            raise ValueError("hybrid rrf_k must be non-negative")
        self.bm25 = bm25
        self.dense = dense
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridRRFRetriever":
        if settings.paths.retrieval_corpus is None:
            raise ValueError("retrieval_corpus must be configured")
        return cls(
            ProductScopedBM25(settings.paths.retrieval_corpus, settings.bm25),
            DenseFaissRetriever.from_settings(settings),
            settings.hybrid,
        )

    def retrieve(
        self, product_id: str | int, query: str, top_k: int | None = None
    ) -> list[HybridRetrievedReview]:
        bm25 = self.bm25.retrieve(product_id, query, self.settings.bm25_candidate_depth)
        dense = self.dense.retrieve(product_id, query, self.settings.dense_candidate_depth)
        return fuse_rrf(
            product_id=product_id,
            bm25_candidates=bm25,
            dense_candidates=dense,
            rrf_k=self.settings.rrf_k,
            final_top_k=top_k or self.settings.final_top_k,
        )
