from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from digikala_comparison.config import RerankerSettings, Settings
from digikala_comparison.bm25 import RetrievedReview
from digikala_comparison.hybrid import HybridRetrievedReview
from digikala_comparison import reranker_evaluation
from digikala_comparison.reranker import (
    BgeRerankerV2M3,
    HybridBgeReranker,
    RerankerUnavailableError,
    reranker_preflight,
)


def _candidate(
    review_id: str,
    *,
    product_id: str = "p1",
    fused_rank: int = 1,
    text: str | None = None,
) -> HybridRetrievedReview:
    return HybridRetrievedReview(
        review_id=review_id,
        product_id=product_id,
        score=1 / (60 + fused_rank),
        rank=fused_rank,
        indexed_text_normalized=text or f"evidence {review_id}",
        review_text_raw=text or f"raw {review_id}",
        title_raw=None,
        advantages_items=None,
        disadvantages_items=None,
        is_buyer=True,
        recommendation_status="recommended",
        review_rate=5.0,
        likes=0.0,
        dislikes=0.0,
        bm25_score=10.0,
        bm25_rank=fused_rank,
        dense_score=0.5,
        dense_rank=fused_rank,
        fused_score=1 / (60 + fused_rank),
        fused_rank=fused_rank,
    )


class _HybridStub:
    def __init__(self, candidates: list[HybridRetrievedReview]):
        self.candidates = candidates
        self.calls: list[tuple[str, str, int | None]] = []

    def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[HybridRetrievedReview]:
        self.calls.append((str(product_id), query, top_k))
        return self.candidates[:top_k]


class _ScoreStub:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[tuple[list[str], list[str]]] = []

    def score_pairs(self, queries: list[str], passages: list[str]) -> np.ndarray:
        self.calls.append((queries, passages))
        return np.asarray([self.scores[passage.rsplit(" ", 1)[-1]] for passage in passages])

    def metadata(self) -> dict[str, object]:
        return {"adapter": "tests._ScoreStub"}


def test_reranker_mapping_id_preservation_truncation_and_determinism() -> None:
    hybrid = _HybridStub([_candidate("r1", fused_rank=1), _candidate("r2", fused_rank=2), _candidate("r3", fused_rank=3)])
    scorer = _ScoreStub({"r1": 0.1, "r2": 0.9, "r3": 0.9})
    reranker = HybridBgeReranker(
        hybrid, scorer, RerankerSettings(candidate_depth=2, candidate_depths=(2,), final_top_k=2)
    )

    first = reranker.retrieve("p1", "query")
    second = reranker.retrieve("p1", "query")

    assert [item.review_id for item in first] == ["r2", "r1"]
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert first[0].review_id == "r2"
    assert first[0].bm25_rank == 2
    assert first[0].dense_rank == 2
    assert first[0].fused_rank == 2
    assert first[0].final_rank == 1
    assert first[0].reranker_score == pytest.approx(0.9)
    assert reranker.last_candidate_review_ids == ["r1", "r2"]
    assert hybrid.calls[0][2] == 2
    assert len(scorer.calls[0][0]) == 2


def test_reranker_tie_break_and_product_filtering() -> None:
    scorer = _ScoreStub({"a": 0.5, "b": 0.5})
    stable = HybridBgeReranker(
        _HybridStub([_candidate("b"), _candidate("a", fused_rank=2)]),
        scorer,
        RerankerSettings(candidate_depth=2, candidate_depths=(2,), final_top_k=2),
    )
    assert [item.review_id for item in stable.retrieve("p1", "query")] == ["a", "b"]

    leaking = HybridBgeReranker(
        _HybridStub([_candidate("bad", product_id="p2")]),
        _ScoreStub({"bad": 1.0}),
        RerankerSettings(candidate_depth=1, candidate_depths=(1,), final_top_k=1),
    )
    with pytest.raises(ValueError, match="cross-product"):
        leaking.retrieve("p1", "query")


def test_flag_embedding_adapter_uses_configured_batching_without_model_download() -> None:
    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

        def compute_score(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            self.calls.append((pairs, kwargs))
            return [0.1, 0.2, 0.3]

    adapter = object.__new__(BgeRerankerV2M3)
    adapter.settings = RerankerSettings(batch_size=2, max_length=128)
    adapter.model = _Recorder()
    values = adapter.score_pairs(["q"] * 3, ["a", "b", "c"])

    assert values.tolist() == [0.1, 0.2, 0.3]
    assert adapter.model.calls == [
        ([("q", "a"), ("q", "b"), ("q", "c")], {"batch_size": 2, "max_length": 128})
    ]


def _tuning_settings(tmp_path: Path) -> Settings:
    base = Settings.from_toml(Path(__file__).parents[1] / "config" / "default.toml")
    identifiers = [
        "bm25_seed_01", "bm25_seed_02", "bm25_seed_03", "bm25_seed_04", "bm25_seed_05", "bm25_seed_low_evidence"
    ]
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "\n".join(json.dumps({"query_id": key, "product_id": "p1", "query": key}) for key in identifiers) + "\n",
        encoding="utf-8",
    )
    qrels_path = tmp_path / "qrels.csv"
    pl.DataFrame({"query_id": identifiers, "review_id": [f"r{index}" for index in range(6)], "relevance_grade": [1] * 6}).write_csv(qrels_path)
    return replace(
        base,
        paths=replace(
            base.paths,
            retrieval_queries=query_path,
            retrieval_qrels=qrels_path,
            hybrid_splits=tmp_path / "splits.json",
            reranker_tuning_report=tmp_path / "tuning.json",
        ),
        reranker=replace(base.reranker, candidate_depths=(20, 50), candidate_depth=20),
    )


def test_depth_tuning_uses_only_development_queries(tmp_path: Path) -> None:
    settings = _tuning_settings(tmp_path)
    from digikala_comparison.hybrid_evaluation import freeze_hybrid_splits

    split = freeze_hybrid_splits(settings)
    seen_queries: list[str] = []

    class _QueryHybrid:
        def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[HybridRetrievedReview]:
            seen_queries.append(query)
            ordered = [
                "bm25_seed_01", "bm25_seed_02", "bm25_seed_03", "bm25_seed_04", "bm25_seed_05", "bm25_seed_low_evidence"
            ]
            return [_candidate(f"r{ordered.index(query)}")]

    scorer = _ScoreStub({f"r{index}": 1.0 for index in range(6)})
    report = reranker_evaluation.tune_reranker_candidate_depth(settings, split, _QueryHybrid(), scorer)

    assert set(seen_queries) == set(split["development_query_ids"])
    assert set(seen_queries).isdisjoint(split["test_query_ids"])
    assert report["selected_candidate_depth"] == 20


def test_configuration_and_graceful_resource_failure_without_model_load() -> None:
    settings = Settings.from_toml(Path(__file__).parents[1] / "config" / "default.toml")
    assert settings.reranker.model_id == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker.model_revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert settings.reranker.candidate_depths == (20, 50, 100)
    too_large = replace(settings.reranker, minimum_available_ram_bytes=2**63)
    assert reranker_preflight(too_large)["status"] == "unavailable"
    with pytest.raises(RerankerUnavailableError, match="CPU preflight rejected"):
        BgeRerankerV2M3(too_large, Path("unused-test-cache"))
    import torch

    if not torch.cuda.is_available():
        with pytest.raises(RerankerUnavailableError, match="CUDA is unavailable"):
            BgeRerankerV2M3(replace(settings.reranker, device="cuda"), Path("unused-test-cache"))


def test_four_way_benchmark_available_path_uses_reranked_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _tuning_settings(tmp_path)
    corpus = tmp_path / "corpus.parquet"
    corpus.write_bytes(b"synthetic")
    settings = replace(
        settings,
        paths=replace(
            settings.paths,
            retrieval_corpus=corpus,
            hybrid_ranked_results=tmp_path / "hybrid-results.parquet",
            reranker_cache_root=tmp_path / "reranker-cache",
            reranker_resource_report=tmp_path / "resource.json",
            reranker_ranked_results=tmp_path / "ranked.parquet",
            reranker_evaluation_report=tmp_path / "evaluation.json",
            reranker_analysis_report=tmp_path / "analysis.json",
            reranker_failure_analysis=tmp_path / "failures.json",
            reranker_selection_report=tmp_path / "selection.json",
        ),
    )
    ordered = [
        "bm25_seed_01", "bm25_seed_02", "bm25_seed_03", "bm25_seed_04", "bm25_seed_05", "bm25_seed_low_evidence"
    ]

    def retrieved(product_id: str | int, query: str, score: float) -> list[RetrievedReview]:
        review_id = f"r{ordered.index(query)}"
        return [
            RetrievedReview(
                review_id=review_id,
                product_id=str(product_id),
                score=score,
                rank=1,
                indexed_text_normalized=f"evidence {review_id}",
                review_text_raw=f"raw {review_id}",
                title_raw=None,
                advantages_items=None,
                disadvantages_items=None,
                is_buyer=True,
                recommendation_status="recommended",
                review_rate=5.0,
                likes=0.0,
                dislikes=0.0,
            )
        ]

    class _Bm25:
        def __init__(self, *args: object, **kwargs: object):
            pass

        def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[RetrievedReview]:
            return retrieved(product_id, query, 2.0)

    class _Dense:
        manifest = {"index_storage_bytes": 11}

        @classmethod
        def from_settings(cls, settings: Settings) -> "_Dense":
            return cls()

        def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[RetrievedReview]:
            return retrieved(product_id, query, 0.9)

    class _Model:
        def __init__(self, *args: object, **kwargs: object):
            pass

        def score_pairs(self, queries: list[str], passages: list[str]) -> np.ndarray:
            return np.ones(len(passages))

        def metadata(self) -> dict[str, object]:
            return {"adapter": "tests._Model"}

    split = {
        "development_query_ids": ["bm25_seed_01", "bm25_seed_03", "bm25_seed_04"],
        "test_query_ids": ["bm25_seed_02", "bm25_seed_05", "bm25_seed_low_evidence"],
        "queries_sha256": "queries", "qrels_sha256": "qrels",
    }
    base = {
        "corpus_version": "test-corpus",
        "configuration": {"rrf_k": 60},
        "tuning": {"selected_rrf_k": 60},
        "methods": {
            "bm25": {"status": "available", "metrics_at_k": {"recall": 0.1, "precision": 0.1, "mrr": 0.1, "ndcg": 0.1}, "latency_ms": {"warm_p95": 1.0}, "per_query": []},
            "bge_m3_dense": {"status": "available", "metrics_at_k": {"recall": 0.1, "precision": 0.1, "mrr": 0.1, "ndcg": 0.1}, "latency_ms": {"warm_p95": 1.0}, "per_query": []},
            "hybrid_rrf": {"status": "available", "metrics_at_k": {"recall": 0.1, "precision": 0.1, "mrr": 0.1, "ndcg": 0.1}, "latency_ms": {"warm_p95": 1.0}, "storage": {"total_storage_bytes": 5}, "per_query": []},
        },
    }
    monkeypatch.setattr(reranker_evaluation, "evaluate_hybrid", lambda settings: base)
    monkeypatch.setattr(reranker_evaluation, "freeze_hybrid_splits", lambda settings: split)
    monkeypatch.setattr(reranker_evaluation, "write_reranker_resource_report", lambda settings: {"status": "ready", "resources": {}})
    monkeypatch.setattr(reranker_evaluation, "ProductScopedBM25", _Bm25)
    monkeypatch.setattr(reranker_evaluation, "DenseFaissRetriever", _Dense)
    monkeypatch.setattr(reranker_evaluation, "BgeRerankerV2M3", _Model)

    report = reranker_evaluation.evaluate_reranker(settings)

    assert report["status"] == "available"
    assert report["methods"]["hybrid_bge_reranker"]["status"] == "available"
    assert report["production_selection"]["selected_method"] == "hybrid_bge_reranker"
    rows = pl.read_parquet(settings.paths.reranker_ranked_results)
    assert rows.filter(pl.col("method") == "hybrid_bge_reranker").height == 3
    assert rows.filter(pl.col("method") == "hybrid_bge_reranker").select("reranker_score").to_series().null_count() == 0
