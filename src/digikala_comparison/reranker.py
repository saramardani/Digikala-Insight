"""Pinned BGE cross-encoder reranking after product-scoped hybrid retrieval."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np

from .config import RerankerSettings, Settings
from .dense_embedding import runtime_resources
from .hybrid import HybridRetrievedReview, HybridRRFRetriever
from .retrieval_contract import ProductReviewRetriever


RERANKER_INPUT_FIELDS = (
    "indexed_text_normalized: normalized title, normalized review body, then normalized "
    "advantages/disadvantages as composed in the frozen Phase 4 retrieval corpus"
)


class RerankerUnavailableError(RuntimeError):
    """The exact configured reranker cannot safely run on this host."""


class RerankerScorer(Protocol):
    """Seam that makes deterministic rerank unit tests model-free."""

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray: ...

    def metadata(self) -> dict[str, object]: ...


class RerankedReview(HybridRetrievedReview):
    """Final evidence preserving source ranks/scores and cross-encoder score."""

    reranker_score: float
    final_rank: int


def _required(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} must be configured")
    return path


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RerankerUnavailableError("torch is required for BGE reranking") from exc
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RerankerUnavailableError("reranker.device=cuda was requested but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("reranker.device must be auto, cpu, or cuda")
    return requested


def reranker_preflight(settings: RerankerSettings) -> dict[str, object]:
    """Check a conservative, recorded resource guard before model download/load."""
    device = _resolve_device(settings.device)
    resources = runtime_resources()
    report: dict[str, object] = {
        "schema_version": "bge-reranker-preflight-v1",
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "model_weights_bytes": settings.model_weights_bytes,
        "device": device,
        "dtype": "float16" if settings.use_fp16 else "float32",
        "batch_size": settings.batch_size,
        "max_length": settings.max_length,
        "query_max_length": settings.query_max_length,
        "truncation_policy": "tokenizer truncates each pair to max_length; query is capped at query_max_length and the passage is truncated to fit",
        "reranker_input_fields": RERANKER_INPUT_FIELDS,
        "resources": resources,
    }
    if settings.use_fp16 and device == "cpu":
        report.update({"status": "unavailable", "reason": "reranker.use_fp16=true is unsupported on CPU"})
        return report
    available_ram = int(resources["available_ram_bytes"])
    if device == "cpu" and available_ram < settings.minimum_available_ram_bytes:
        report.update(
            {
                "status": "unavailable",
                "reason": (
                    "CPU preflight rejected model load: available RAM "
                    f"({available_ram} bytes) is below the configured safety floor "
                    f"({settings.minimum_available_ram_bytes} bytes) for a "
                    f"{settings.model_weights_bytes}-byte pinned model."
                ),
                "minimum_available_ram_bytes": settings.minimum_available_ram_bytes,
            }
        )
        return report
    report["status"] = "ready"
    return report


def write_reranker_resource_report(settings: Settings) -> dict[str, object]:
    report = reranker_preflight(settings.reranker)
    _write_json(_required(settings.paths.reranker_resource_report, "reranker_resource_report"), report)
    return report


class BgeRerankerV2M3:
    """Official FlagEmbedding adapter for the exact pinned BGE reranker."""

    def __init__(self, settings: RerankerSettings, cache_dir: Path):
        self.settings = settings
        preflight = reranker_preflight(settings)
        if preflight["status"] != "ready":
            raise RerankerUnavailableError(str(preflight["reason"]))
        self.device = str(preflight["device"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        from FlagEmbedding import FlagReranker
        from huggingface_hub import snapshot_download

        started = perf_counter()
        snapshot = snapshot_download(
            repo_id=settings.model_id,
            revision=settings.model_revision,
            cache_dir=str(cache_dir),
        )
        self.snapshot_path = Path(snapshot)
        self.model = FlagReranker(
            str(self.snapshot_path),
            use_fp16=settings.use_fp16,
            devices=self.device,
            batch_size=settings.batch_size,
            query_max_length=settings.query_max_length,
            max_length=settings.max_length,
            normalize=False,
        )
        self.load_seconds = perf_counter() - started

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        if len(queries) != len(passages):
            raise ValueError("reranker query/passage counts must match")
        if not queries:
            return np.empty(0, dtype=np.float64)
        pairs = list(zip(queries, passages, strict=True))
        scores = self.model.compute_score(
            pairs,
            batch_size=self.settings.batch_size,
            max_length=self.settings.max_length,
        )
        values = np.asarray(scores if isinstance(scores, list) else [scores], dtype=np.float64).reshape(-1)
        if values.shape[0] != len(pairs) or not np.isfinite(values).all():
            raise RuntimeError("BGE reranker returned invalid scores")
        return values

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.settings.model_id,
            "model_revision": self.settings.model_revision,
            "adapter": "FlagEmbedding.FlagReranker",
            "flag_embedding_version": importlib.metadata.version("FlagEmbedding"),
            "device": self.device,
            "dtype": "float16" if self.settings.use_fp16 else "float32",
            "batch_size": self.settings.batch_size,
            "max_length": self.settings.max_length,
            "query_max_length": self.settings.query_max_length,
            "truncation_policy": "tokenizer truncates each pair to max_length; query is capped at query_max_length and the passage is truncated to fit",
            "reranker_input_fields": RERANKER_INPUT_FIELDS,
            "snapshot_path": str(self.snapshot_path),
            "model_load_seconds": self.load_seconds,
        }


class HybridBgeReranker:
    """Product-safe hybrid candidate retrieval followed by BGE cross-encoding."""

    def __init__(
        self,
        hybrid: ProductReviewRetriever,
        scorer: RerankerScorer,
        settings: RerankerSettings,
    ):
        if settings.candidate_depths and any(depth <= 0 for depth in settings.candidate_depths):
            raise ValueError("reranker candidate depths must be positive")
        if settings.candidate_depth <= 0:
            raise ValueError("reranker candidate_depth must be positive")
        if settings.final_top_k <= 0:
            raise ValueError("reranker final_top_k must be positive")
        self.hybrid = hybrid
        self.scorer = scorer
        self.settings = settings
        self.last_reranker_latency_ms: float | None = None
        self.last_candidate_review_ids: list[str] = []
        self.last_candidate_count = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridBgeReranker":
        cache = _required(settings.paths.reranker_cache_root, "reranker_cache_root")
        return cls(
            HybridRRFRetriever.from_settings(settings),
            BgeRerankerV2M3(settings.reranker, cache),
            settings.reranker,
        )

    def retrieve(
        self, product_id: str | int, query: str, top_k: int | None = None
    ) -> list[RerankedReview]:
        expected_product_id = str(product_id)
        candidate_depth = self.settings.candidate_depth
        candidates = list(self.hybrid.retrieve(product_id, query, candidate_depth))[:candidate_depth]
        if any(str(candidate.product_id) != expected_product_id for candidate in candidates):
            raise ValueError("cross-product candidate rejected before BGE reranking")
        self.last_candidate_review_ids = [str(candidate.review_id) for candidate in candidates]
        self.last_candidate_count = len(candidates)
        if not candidates:
            self.last_reranker_latency_ms = 0.0
            return []
        started = perf_counter()
        scores = self.scorer.score_pairs(
            [query] * len(candidates),
            [candidate.indexed_text_normalized for candidate in candidates],
        )
        self.last_reranker_latency_ms = (perf_counter() - started) * 1000
        if scores.shape != (len(candidates),):
            raise RuntimeError("reranker score count does not match hybrid candidate count")
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), str(item[0].review_id)),
        )
        final_top_k = top_k or self.settings.final_top_k
        if final_top_k <= 0:
            raise ValueError("reranker top_k must be positive")
        output: list[RerankedReview] = []
        for final_rank, (candidate, reranker_score) in enumerate(ranked[:final_top_k], start=1):
            payload = candidate.model_dump()
            payload.update(
                {
                    "score": float(reranker_score),
                    "rank": final_rank,
                    "reranker_score": float(reranker_score),
                    "final_rank": final_rank,
                }
            )
            output.append(RerankedReview(**payload))
        if any(str(result.product_id) != expected_product_id for result in output):
            raise RuntimeError("cross-product candidate leaked through BGE reranking")
        return output
