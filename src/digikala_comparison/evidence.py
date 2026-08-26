"""Production-facing, provenance-preserving retrieval evidence API.

This module intentionally keeps Top-K evidence distinct from full-population
product statistics.  Neither type can be passed to the other's API.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from .bm25 import ProductScopedBM25, RetrievedReview
from .config import Settings
from .dense_index import DenseFaissRetriever
from .hybrid import HybridRRFRetriever
from .reranker import HybridBgeReranker
from .retrieval_freeze import load_frozen_retrieval_experiment


EvidenceStatus = Literal["sufficient_candidates", "limited_candidates", "no_evidence"]


class EvidenceAudit(BaseModel):
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    reranker_score: float | None = None
    final_rank: int | None = None


class EvidenceItem(BaseModel):
    review_id: str
    product_id: str
    rank: int
    final_score: float
    raw_evidence_text: str | None
    is_buyer: bool | None
    recommendation_status: str | None
    likes: float | None
    dislikes: float | None
    audit: EvidenceAudit


class ScoreDistributionSummary(BaseModel):
    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    note: str = "Retrieval scores are ranking signals, not calibrated probabilities or confidence values."


class EvidenceSet(BaseModel):
    """Top-K review evidence only; it contains no product-population percentages."""

    product_id: str
    criterion: str
    query: str
    retrieval_method: str
    retrieval_method_version: str
    experiment_manifest_sha256: str
    requested_top_k: int
    retrieved_count: int
    eligible_product_review_count: int
    retrieval_status: EvidenceStatus
    score_distribution: ScoreDistributionSummary
    evidence_items: list[EvidenceItem] = Field(default_factory=list)


class FullProductStatistics(BaseModel):
    """Full-population values sourced only from product_statistics.parquet."""

    product_id: str
    total_review_count: int
    recommendation_known_count: int
    recommended_count: int
    not_recommended_count: int
    no_idea_count: int
    recommended_percentage: float | None
    not_recommended_percentage: float | None
    no_idea_percentage: float | None
    opinionated_recommend_percentage: float | None


class GlobalRecommendationSummary(BaseModel):
    product_id: str
    population_review_count: int
    recommendation_known_count: int
    recommended_percentage: float | None
    not_recommended_percentage: float | None
    no_idea_percentage: float | None
    opinionated_recommend_percentage: float | None


def global_recommendation_summary(statistics: FullProductStatistics) -> GlobalRecommendationSummary:
    """Expose population statistics only from the separately typed Phase 2 artifact."""
    if not isinstance(statistics, FullProductStatistics):
        raise TypeError("global recommendation summaries require FullProductStatistics, never EvidenceSet")
    return GlobalRecommendationSummary(
        product_id=statistics.product_id,
        population_review_count=statistics.total_review_count,
        recommendation_known_count=statistics.recommendation_known_count,
        recommended_percentage=statistics.recommended_percentage,
        not_recommended_percentage=statistics.not_recommended_percentage,
        no_idea_percentage=statistics.no_idea_percentage,
        opinionated_recommend_percentage=statistics.opinionated_recommend_percentage,
    )


class ProductStatisticsStore:
    """Lazy typed access to Phase 2 full-population statistics; not retrieval data."""

    def __init__(self, statistics_path: Path):
        if not statistics_path.is_file():
            raise FileNotFoundError("product_statistics.parquet is required for full-population statistics")
        self.statistics_path = statistics_path

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProductStatisticsStore":
        if settings.paths.product_statistics is None:
            raise ValueError("product_statistics must be configured")
        return cls(settings.paths.product_statistics)

    def get(self, product_id: str | int) -> FullProductStatistics | None:
        result = (
            pl.scan_parquet(self.statistics_path)
            .filter(pl.col("product_id").cast(pl.String) == str(product_id))
            .select(
                "product_id",
                "total_review_count",
                "recommendation_known_count",
                "recommended_count",
                "not_recommended_count",
                "no_idea_count",
                "recommended_percentage",
                "not_recommended_percentage",
                "no_idea_percentage",
                "opinionated_recommend_percentage",
            )
            .collect()
        )
        if result.height == 0:
            return None
        row = result.row(0, named=True)
        return FullProductStatistics(**row)


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evidence_item(result: RetrievedReview) -> EvidenceItem:
    return EvidenceItem(
        review_id=str(result.review_id),
        product_id=str(result.product_id),
        rank=int(result.rank),
        final_score=float(result.score),
        raw_evidence_text=result.review_text_raw,
        is_buyer=result.is_buyer,
        recommendation_status=result.recommendation_status,
        likes=result.likes,
        dislikes=result.dislikes,
        audit=EvidenceAudit(
            bm25_score=getattr(result, "bm25_score", None),
            bm25_rank=getattr(result, "bm25_rank", None),
            dense_score=getattr(result, "dense_score", None),
            dense_rank=getattr(result, "dense_rank", None),
            fused_score=getattr(result, "fused_score", None),
            fused_rank=getattr(result, "fused_rank", None),
            reranker_score=getattr(result, "reranker_score", None),
            final_rank=getattr(result, "final_rank", None),
        ),
    )


class ProductionEvidenceRetriever:
    """Uses only the manifest-frozen production method to fetch review evidence."""

    def __init__(self, settings: Settings, manifest: dict[str, Any], retriever: Any | None = None):
        self.settings = settings
        self.manifest = manifest
        self.method = str(manifest["selected_production_retriever"]["selected_method"])
        self._validate_frozen_configuration()
        self.retriever = retriever if retriever is not None else self._build_retriever()

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProductionEvidenceRetriever":
        return cls(settings, load_frozen_retrieval_experiment(settings))

    def _validate_frozen_configuration(self) -> None:
        frozen = self.manifest["configuration"]
        current = {
            "bm25": asdict(self.settings.bm25),
            "dense": asdict(self.settings.dense),
            "hybrid": asdict(self.settings.hybrid),
            "reranker": asdict(self.settings.reranker),
        }
        # The frozen configuration is JSON, so canonicalize tuples (for
        # example reranker candidate_depths) before comparing it to dataclasses.
        current = json.loads(json.dumps(current))
        if frozen != current:
            raise ValueError("current retrieval settings differ from the frozen experiment manifest")
        corpus = self.settings.paths.retrieval_corpus
        if corpus is None or not corpus.is_file():
            raise FileNotFoundError("retrieval corpus is missing")
        if corpus.stat().st_size != int(self.manifest["corpus"]["size_bytes"]):
            raise ValueError("retrieval corpus size differs from the frozen experiment manifest")

    def _build_retriever(self) -> Any:
        if self.settings.paths.retrieval_corpus is None:
            raise ValueError("retrieval_corpus must be configured")
        if self.method == "bm25":
            return ProductScopedBM25(self.settings.paths.retrieval_corpus, self.settings.bm25)
        if self.method == "bge_m3_dense":
            return DenseFaissRetriever.from_settings(self.settings)
        if self.method == "hybrid_rrf":
            return HybridRRFRetriever.from_settings(self.settings)
        if self.method == "hybrid_bge_reranker":
            return HybridBgeReranker.from_settings(self.settings)
        raise ValueError(f"unsupported frozen production retriever: {self.method}")

    def _eligible_review_count(self, product_id: str) -> int:
        corpus = self.settings.paths.retrieval_corpus
        if corpus is None:
            raise ValueError("retrieval_corpus must be configured")
        return int(
            pl.scan_parquet(corpus)
            .filter(pl.col("product_id").cast(pl.String) == product_id)
            .select(pl.len())
            .collect()[0, 0]
        )

    def retrieve_evidence(
        self, product_id: str | int, criterion: str, query: str, top_k: int = 10
    ) -> EvidenceSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        expected_product_id = str(product_id)
        results = list(self.retriever.retrieve(expected_product_id, query, top_k))
        if len(results) > top_k:
            raise RuntimeError("production retriever returned more than requested top_k")
        ids = [str(result.review_id) for result in results]
        if len(ids) != len(set(ids)):
            raise RuntimeError("production retriever returned duplicate review_id values")
        if any(str(result.product_id) != expected_product_id for result in results):
            raise RuntimeError("production retriever returned a cross-product evidence item")
        items = [_evidence_item(result) for result in results]
        scores = [item.final_score for item in items]
        if not items:
            status: EvidenceStatus = "no_evidence"
        elif len(items) < top_k:
            status = "limited_candidates"
        else:
            status = "sufficient_candidates"
        return EvidenceSet(
            product_id=expected_product_id,
            criterion=criterion,
            query=query,
            retrieval_method=self.method,
            retrieval_method_version=(
                f"{self.manifest['schema_version']}:{self.manifest['corpus']['sha256'][:12]}"
            ),
            experiment_manifest_sha256=_manifest_sha256(self.manifest),
            requested_top_k=top_k,
            retrieved_count=len(items),
            eligible_product_review_count=self._eligible_review_count(expected_product_id),
            retrieval_status=status,
            score_distribution=ScoreDistributionSummary(
                count=len(scores),
                minimum=min(scores) if scores else None,
                maximum=max(scores) if scores else None,
                mean=fmean(scores) if scores else None,
            ),
            evidence_items=items,
        )
