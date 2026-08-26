from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import polars as pl
import pytest

from digikala_comparison.bm25 import RetrievedReview
from digikala_comparison.config import Settings
from digikala_comparison.evidence import (
    FullProductStatistics,
    ProductStatisticsStore,
    ProductionEvidenceRetriever,
    global_recommendation_summary,
)
from digikala_comparison.retrieval_freeze import (
    freeze_retrieval_experiment,
    load_frozen_retrieval_experiment,
)


_IDS = [
    "bm25_seed_01", "bm25_seed_02", "bm25_seed_03", "bm25_seed_04", "bm25_seed_05", "bm25_seed_low_evidence"
]


def _settings(tmp_path: Path) -> Settings:
    base = Settings.from_toml(Path(__file__).parents[1] / "config" / "default.toml")
    corpus = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "product_id": ["p1", "p1", "p2"],
            "indexed_text_normalized": ["first", "second", "third"],
            "review_text_raw": ["raw first", "raw second", "raw third"],
            "title_raw": [None, None, None],
            "advantages_items": [None, None, None],
            "disadvantages_items": [None, None, None],
            "is_buyer_bool": [True, False, True],
            "recommendation_status": ["recommended", "not_recommended", "recommended"],
            "review_rate_numeric": [5.0, 2.0, 4.0],
            "likes_numeric": [2.0, 0.0, 1.0],
            "dislikes_numeric": [0.0, 1.0, 0.0],
        }
    ).write_parquet(corpus)
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "\n".join(json.dumps({"query_id": key, "product_id": "p1", "query": key}) for key in _IDS) + "\n",
        encoding="utf-8",
    )
    qrels_path = tmp_path / "qrels.csv"
    pl.DataFrame({"query_id": _IDS, "review_id": ["r1"] * len(_IDS), "relevance_grade": [1] * len(_IDS)}).write_csv(qrels_path)
    corpus_report = tmp_path / "corpus-report.json"
    corpus_report.write_text(
        json.dumps(
            {
                "corpus_version": f"{base.dataset.revision}:bm25-corpus-v1",
                "document_unit": "one eligible canonical review",
                "eligible_retrieval_reviews": 3,
                "eligibility_policy": {"valid_review_id": True},
                "field_composition_policy": "test composition",
                "indexed_fields": ["review_text_normalized"],
                "tokenization": "test tokenizer",
            }
        ),
        encoding="utf-8",
    )
    statistics = tmp_path / "statistics.parquet"
    pl.DataFrame(
        {
            "product_id": ["p1"],
            "total_review_count": [99],
            "recommendation_known_count": [90],
            "recommended_count": [70],
            "not_recommended_count": [10],
            "no_idea_count": [10],
            "recommended_percentage": [70 / 90],
            "not_recommended_percentage": [10 / 90],
            "no_idea_percentage": [10 / 90],
            "opinionated_recommend_percentage": [70 / 80],
        }
    ).write_parquet(statistics)
    paths = replace(
        base.paths,
        retrieval_corpus=corpus,
        retrieval_corpus_report=corpus_report,
        product_statistics=statistics,
        retrieval_queries=query_path,
        retrieval_qrels=qrels_path,
        hybrid_splits=tmp_path / "split.json",
        frozen_development_queries=tmp_path / "frozen" / "development.jsonl",
        frozen_development_qrels=tmp_path / "frozen" / "development.csv",
        frozen_test_queries=tmp_path / "frozen" / "test.jsonl",
        frozen_test_qrels=tmp_path / "frozen" / "test.csv",
        retrieval_experiment_manifest=tmp_path / "manifest.json",
        retrieval_benchmark_markdown=tmp_path / "summary.md",
        reranker_evaluation_report=tmp_path / "benchmark.json",
    )
    return replace(base, paths=paths)


def _benchmark() -> dict[str, object]:
    method = {
        "status": "available",
        "metrics_at_k": {"recall": 1.0, "precision": 0.1, "mrr": 1.0, "ndcg": 1.0},
        "latency_ms": {"warm_p50": 1.0, "warm_p95": 2.0},
        "storage_bytes": 3,
        "peak_process_memory_bytes": 4,
    }
    unavailable = {
        "status": "unavailable",
        "metrics_at_k": {"recall": None, "precision": None, "mrr": None, "ndcg": None},
        "latency_ms": None,
        "storage_bytes": None,
        "peak_process_memory_bytes": None,
    }
    return {
        "status": "reranker_unavailable",
        "methods": {"bm25": method, "bge_m3_dense": unavailable, "hybrid_rrf": unavailable, "hybrid_bge_reranker": unavailable},
        "production_selection": {"selected_method": "bm25", "status": "baseline_retained"},
        "ranked_results_path": "synthetic.parquet",
        "label_limitations": "test seed labels",
    }


def _review(review_id: str, product_id: str = "p1", score: float = 1.0, rank: int = 1) -> RetrievedReview:
    return RetrievedReview(
        review_id=review_id,
        product_id=product_id,
        score=score,
        rank=rank,
        indexed_text_normalized=f"normalized {review_id}",
        review_text_raw=f"raw {review_id}",
        title_raw=None,
        advantages_items=None,
        disadvantages_items=None,
        is_buyer=True,
        recommendation_status="recommended",
        review_rate=5.0,
        likes=2.0,
        dislikes=0.0,
    )


class _Retriever:
    def __init__(self, results: list[RetrievedReview]):
        self.results = results

    def retrieve(self, product_id: str, query: str, top_k: int) -> list[RetrievedReview]:
        return self.results


def test_freeze_creates_separate_partitions_and_reproducible_manifest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = freeze_retrieval_experiment(settings, _benchmark())
    second = freeze_retrieval_experiment(settings, _benchmark())

    assert first == second == load_frozen_retrieval_experiment(settings)
    assert first["partitions"]["development"]["query_count"] == 3
    assert first["partitions"]["test"]["query_count"] == 3
    assert first["selected_production_retriever"]["selected_method"] == "bm25"
    assert first["tokenizer"]["version"] == "persian-lexical-v1"
    assert settings.paths.frozen_development_queries.is_file()
    assert settings.paths.frozen_test_qrels.is_file()
    assert settings.paths.retrieval_benchmark_markdown.is_file()
    assert ProductionEvidenceRetriever.from_settings(settings).method == "bm25"


def test_evidence_set_schema_provenance_fewer_than_k_and_no_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manifest = freeze_retrieval_experiment(settings, _benchmark())
    evidence = ProductionEvidenceRetriever(
        settings, manifest, _Retriever([_review("r1", score=0.8), _review("r2", score=0.2, rank=2)])
    ).retrieve_evidence("p1", "battery", "battery life", top_k=3)

    assert evidence.product_id == "p1"
    assert evidence.criterion == "battery"
    assert evidence.retrieval_method == "bm25"
    assert evidence.retrieved_count == 2
    assert evidence.eligible_product_review_count == 2
    assert evidence.retrieval_status == "limited_candidates"
    assert [item.review_id for item in evidence.evidence_items] == ["r1", "r2"]
    assert evidence.evidence_items[0].raw_evidence_text == "raw r1"
    assert evidence.score_distribution.mean == pytest.approx(0.5)

    empty = ProductionEvidenceRetriever(settings, manifest, _Retriever([])).retrieve_evidence("p1", "battery", "battery", top_k=2)
    assert empty.retrieval_status == "no_evidence"
    assert empty.evidence_items == []


def test_evidence_api_rejects_cross_product_and_duplicate_ids(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manifest = freeze_retrieval_experiment(settings, _benchmark())
    with pytest.raises(RuntimeError, match="cross-product"):
        ProductionEvidenceRetriever(settings, manifest, _Retriever([_review("r3", product_id="p2")])).retrieve_evidence("p1", "x", "q")
    with pytest.raises(RuntimeError, match="duplicate review_id"):
        ProductionEvidenceRetriever(settings, manifest, _Retriever([_review("r1"), _review("r1", rank=2)])).retrieve_evidence("p1", "x", "q")


def test_full_statistics_are_typed_and_cannot_be_calculated_from_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    statistics = ProductStatisticsStore.from_settings(settings).get("p1")
    assert statistics is not None
    summary = global_recommendation_summary(statistics)
    assert summary.population_review_count == 99
    assert summary.recommended_percentage == pytest.approx(70 / 90)

    manifest = freeze_retrieval_experiment(settings, _benchmark())
    evidence = ProductionEvidenceRetriever(settings, manifest, _Retriever([_review("r1")])).retrieve_evidence("p1", "x", "q")
    with pytest.raises(TypeError, match="FullProductStatistics"):
        global_recommendation_summary(evidence)  # type: ignore[arg-type]
    assert not hasattr(evidence, "recommended_percentage")


def test_frozen_configuration_rejects_runtime_drift(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manifest = freeze_retrieval_experiment(settings, _benchmark())
    drifted = replace(settings, bm25=replace(settings.bm25, k1=2.0))
    with pytest.raises(ValueError, match="differ from the frozen"):
        ProductionEvidenceRetriever(drifted, manifest, _Retriever([]))
