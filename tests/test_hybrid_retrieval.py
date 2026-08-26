from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from digikala_comparison.bm25 import RetrievedReview
from digikala_comparison.config import HybridSettings, Settings
from digikala_comparison.hybrid import HybridRRFRetriever, fuse_rrf, reciprocal_rank
from digikala_comparison import hybrid_evaluation


def _review(
    review_id: str,
    product_id: str = "p1",
    *,
    score: float = 1.0,
    rank: int = 1,
    text: str | None = None,
) -> RetrievedReview:
    return RetrievedReview(
        review_id=review_id,
        product_id=product_id,
        score=score,
        rank=rank,
        indexed_text_normalized=text or f"normalized {review_id}",
        review_text_raw=text or f"raw {review_id}",
        title_raw=None,
        advantages_items=None,
        disadvantages_items=None,
        is_buyer=True,
        recommendation_status="recommended",
        review_rate=5.0,
        likes=0.0,
        dislikes=0.0,
    )


class _StubRetriever:
    def __init__(self, results: list[RetrievedReview]):
        self.results = results

    def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[RetrievedReview]:
        assert str(product_id) == "p1"
        return self.results[:top_k]


def test_rrf_formula_and_component_provenance() -> None:
    result = fuse_rrf(
        product_id="p1",
        bm25_candidates=[_review("r1", score=9.0, rank=1), _review("r2", score=8.0, rank=2)],
        dense_candidates=[_review("r2", score=0.9, rank=1), _review("r3", score=0.8, rank=2)],
        rrf_k=60,
        final_top_k=3,
    )

    assert reciprocal_rank(1, 60) == pytest.approx(1 / 61)
    assert [item.review_id for item in result] == ["r2", "r1", "r3"]
    assert result[0].fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert result[0].bm25_score == 8.0
    assert result[0].dense_score == 0.9
    assert result[0].bm25_rank == 2
    assert result[0].dense_rank == 1


def test_candidates_merge_only_on_review_id_and_support_missing_source() -> None:
    result = fuse_rrf(
        product_id="p1",
        bm25_candidates=[_review("same", score=4.0, text="lexical wording")],
        dense_candidates=[
            _review("same", score=0.8, text="different dense payload"),
            _review("dense-only", score=0.7, rank=2),
        ],
        rrf_k=10,
        final_top_k=10,
    )

    assert [item.review_id for item in result] == ["same", "dense-only"]
    assert result[0].review_text_raw == "lexical wording"
    assert result[1].bm25_rank is None
    assert result[1].dense_rank == 2


def test_stable_ties_duplicate_and_product_filtering() -> None:
    ties = fuse_rrf(
        product_id="p1",
        bm25_candidates=[_review("b", rank=1), _review("a", rank=1)],
        dense_candidates=[],
        rrf_k=60,
        final_top_k=2,
    )
    assert [item.review_id for item in ties] == ["a", "b"]
    with pytest.raises(ValueError, match="duplicate bm25 candidate"):
        fuse_rrf(
            product_id="p1",
            bm25_candidates=[_review("same"), _review("same", rank=2)],
            dense_candidates=[],
            rrf_k=60,
            final_top_k=2,
        )
    with pytest.raises(ValueError, match="cross-product"):
        fuse_rrf(
            product_id="p1",
            bm25_candidates=[_review("leak", product_id="p2")],
            dense_candidates=[],
            rrf_k=60,
            final_top_k=2,
        )


def test_hybrid_retriever_is_deterministic_and_product_scoped() -> None:
    hybrid = HybridRRFRetriever(
        _StubRetriever([_review("b", rank=1), _review("a", rank=2)]),
        _StubRetriever([_review("a", score=0.9, rank=1)]),
        HybridSettings(bm25_candidate_depth=2, dense_candidate_depth=2, rrf_k=60, final_top_k=2),
    )
    first = hybrid.retrieve("p1", "query")
    second = hybrid.retrieve("p1", "query")
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert [item.review_id for item in first] == ["a", "b"]
    assert all(item.product_id == "p1" for item in first)


def _benchmark_settings(tmp_path: Path) -> Settings:
    base = Settings.from_toml(Path(__file__).parents[1] / "config" / "default.toml")
    query_ids = [
        "bm25_seed_01",
        "bm25_seed_02",
        "bm25_seed_03",
        "bm25_seed_04",
        "bm25_seed_05",
        "bm25_seed_low_evidence",
    ]
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps({"query_id": query_id, "product_id": "p1", "query": query_id})
            for query_id in query_ids
        )
        + "\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.csv"
    pl.DataFrame(
        {"query_id": query_ids, "review_id": [f"r{index}" for index in range(6)], "relevance_grade": [1] * 6}
    ).write_csv(qrels)
    paths = replace(
        base.paths,
        retrieval_queries=queries,
        retrieval_qrels=qrels,
        retrieval_corpus=tmp_path / "corpus.parquet",
        hybrid_splits=tmp_path / "splits.json",
        hybrid_tuning_report=tmp_path / "tuning.json",
        dense_evaluation_report=tmp_path / "dense.json",
    )
    return replace(base, paths=paths, hybrid=replace(base.hybrid, tuning_rrf_k=(20, 60)))


def test_hybrid_configuration_and_frozen_development_test_separation(tmp_path: Path) -> None:
    settings = _benchmark_settings(tmp_path)
    assert settings.hybrid.bm25_candidate_depth == 100
    assert settings.hybrid.dense_candidate_depth == 100
    assert settings.hybrid.rrf_k == 60
    split = hybrid_evaluation.freeze_hybrid_splits(settings)
    assert set(split["development_query_ids"]).isdisjoint(split["test_query_ids"])
    assert set(split["development_query_ids"] + split["test_query_ids"]) == {
        "bm25_seed_01", "bm25_seed_02", "bm25_seed_03", "bm25_seed_04", "bm25_seed_05", "bm25_seed_low_evidence"
    }


def test_rrf_tuning_reads_only_development_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _benchmark_settings(tmp_path)
    split = hybrid_evaluation.freeze_hybrid_splits(settings)
    settings.paths.dense_evaluation_report.write_text(json.dumps({"status": "available"}), encoding="utf-8")
    observed_query_ids: list[str] = []

    class _AnyRetriever:
        def __init__(self, *args: object, **kwargs: object):
            pass

        @classmethod
        def from_settings(cls, settings: Settings) -> "_AnyRetriever":
            return cls()

        def retrieve(self, *args: object, **kwargs: object) -> list[RetrievedReview]:
            return []

    def fake_evaluate_retriever(**kwargs: object) -> tuple[dict[str, object], list[dict[str, object]]]:
        observed_query_ids.extend(str(query["query_id"]) for query in kwargs["queries"])  # type: ignore[index]
        rrf_k = kwargs["retriever"].settings.rrf_k  # type: ignore[index,union-attr]
        return ({"metrics_at_k": {"ndcg": float(rrf_k), "recall": 0.0}}, [])

    monkeypatch.setattr(hybrid_evaluation, "ProductScopedBM25", _AnyRetriever)
    monkeypatch.setattr(hybrid_evaluation, "DenseFaissRetriever", _AnyRetriever)
    monkeypatch.setattr(hybrid_evaluation, "_evaluate_retriever", fake_evaluate_retriever)
    report = hybrid_evaluation.tune_rrf_on_development(settings, split)

    assert set(observed_query_ids) == set(split["development_query_ids"])
    assert set(observed_query_ids).isdisjoint(split["test_query_ids"])
    assert report["selected_rrf_k"] == 60


def test_three_method_benchmark_persists_candidate_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the available benchmark path without loading the real BGE model."""
    settings = _benchmark_settings(tmp_path)
    settings.paths.retrieval_corpus.write_bytes(b"synthetic corpus")
    settings.paths.dense_evaluation_report.write_text(json.dumps({"status": "available"}), encoding="utf-8")
    ordered_ids = [
        "bm25_seed_01",
        "bm25_seed_02",
        "bm25_seed_03",
        "bm25_seed_04",
        "bm25_seed_05",
        "bm25_seed_low_evidence",
    ]

    class _SyntheticBm25:
        def __init__(self, *args: object, **kwargs: object):
            pass

        def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[RetrievedReview]:
            return [_review(f"r{ordered_ids.index(query)}", str(product_id), score=2.0, text="lexical evidence")]

    class _SyntheticDense(_SyntheticBm25):
        def __init__(self, *args: object, **kwargs: object):
            super().__init__(*args, **kwargs)
            self.manifest = {"index_storage_bytes": 23}

        @classmethod
        def from_settings(cls, settings: Settings) -> "_SyntheticDense":
            return cls()

        def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[RetrievedReview]:
            return [_review(f"r{ordered_ids.index(query)}", str(product_id), score=0.9, text="dense evidence")]

    monkeypatch.setattr(hybrid_evaluation, "ProductScopedBM25", _SyntheticBm25)
    monkeypatch.setattr(hybrid_evaluation, "DenseFaissRetriever", _SyntheticDense)
    report = hybrid_evaluation.evaluate_hybrid(settings)

    assert report["status"] == "available"
    assert set(report["methods"]) == {"bm25", "bge_m3_dense", "hybrid_rrf"}
    assert report["methods"]["hybrid_rrf"]["status"] == "available"
    rows = pl.read_parquet(settings.paths.hybrid_ranked_results)
    hybrid_rows = rows.filter(pl.col("method") == "hybrid_rrf")
    assert hybrid_rows.height == 3
    assert hybrid_rows.select("review_id").to_series().to_list() == ["r1", "r4", "r5"]
    assert hybrid_rows.select("fused_score").to_series().null_count() == 0
    assert hybrid_rows.select("review_text_raw").to_series().to_list() == ["lexical evidence"] * 3
    assert report["methods"]["hybrid_rrf"]["storage"]["additional_retrieval_artifact_bytes"] == 23
