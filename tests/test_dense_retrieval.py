from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from digikala_comparison.config import DenseSettings, Settings
from digikala_comparison.dense_index import (
    DenseFaissRetriever,
    DenseIndexPaths,
    build_dense_embeddings,
)
from digikala_comparison.retrieval_contract import ProductReviewRetriever


class FakeDenseEmbedder:
    """Tiny deterministic embedding fixture; it never downloads BGE-M3."""

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            text = text.lower()
            vectors.append([float("good" in text), float("battery" in text), 1.0])
        values = np.asarray(vectors, dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def metadata(self) -> dict[str, object]:
        return {"adapter": "tests.FakeDenseEmbedder", "device": "cpu", "dtype": "float32"}


class ExplodingEmbedder(FakeDenseEmbedder):
    def encode_passages(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - must not run
        raise AssertionError("completed checkpoints should be resumed without re-embedding")


def _review(review_id: str, product_id: str, text: str) -> dict[str, object]:
    return {
        "review_id": review_id,
        "product_id": product_id,
        "indexed_text_normalized": text,
        "review_text_raw": text,
        "title_raw": None,
        "advantages_items": None,
        "disadvantages_items": None,
        "is_buyer_bool": True,
        "recommendation_status": "recommended",
        "review_rate_numeric": 5.0,
        "likes_numeric": 0.0,
        "dislikes_numeric": 0.0,
    }


def _settings(tmp_path: Path, corpus: Path) -> Settings:
    base = Settings.from_toml(Path(__file__).parents[1] / "config" / "default.toml")
    paths = replace(
        base.paths,
        retrieval_corpus=corpus,
        dense_index_root=tmp_path / "dense",
        dense_sorted_corpus=tmp_path / "dense-source.parquet",
        dense_manifest=tmp_path / "dense" / "manifest.json",
        dense_product_ranges=tmp_path / "dense" / "product_ranges.parquet",
    )
    return replace(
        base,
        paths=paths,
        dense=replace(
            base.dense,
            batch_size=1,
            checkpoint_documents=2,
            embedding_dimension=3,
            product_index_cache_size=2,
        ),
    )


def _build(tmp_path: Path) -> tuple[Settings, DenseIndexPaths, dict[str, object]]:
    corpus = tmp_path / "reviews.parquet"
    pl.DataFrame(
        [
            _review("r1", "p1", "good build"),
            _review("r2", "p1", "weak battery"),
            _review("r3", "p2", "good battery"),
        ]
    ).write_parquet(corpus)
    settings = _settings(tmp_path, corpus)
    result = build_dense_embeddings(settings, embedder=FakeDenseEmbedder())
    return settings, result["paths"], result["manifest"]


def test_dense_vector_mapping_checkpoint_resume_and_duplicate_prevention(tmp_path: Path) -> None:
    settings, paths, manifest = _build(tmp_path)
    assert manifest["completed_documents"] == 3
    assert len(manifest["chunks"]) == 2
    metadata = pl.read_parquet(str(paths.metadata_dir / "chunk-*.parquet")).sort("vector_id")
    assert metadata.select("review_id").to_series().to_list() == ["r1", "r2", "r3"]
    assert metadata.select("product_id").to_series().to_list() == ["p1", "p1", "p2"]
    resumed = build_dense_embeddings(settings, embedder=ExplodingEmbedder())
    assert resumed["resumed"] is True
    assert pl.read_parquet(str(paths.metadata_dir / "chunk-*.parquet")).height == 3

    duplicate = tmp_path / "duplicate.parquet"
    pl.DataFrame([_review("same", "p1", "one"), _review("same", "p1", "two")]).write_parquet(duplicate)
    duplicate_settings = _settings(tmp_path / "duplicate-root", duplicate)
    with pytest.raises(ValueError, match="duplicate review_id"):
        build_dense_embeddings(duplicate_settings, embedder=FakeDenseEmbedder())


def test_dense_retrieval_product_filtering_schema_top_k_and_common_interface(tmp_path: Path) -> None:
    settings, paths, _ = _build(tmp_path)
    retriever = DenseFaissRetriever(paths, settings.dense, FakeDenseEmbedder())

    def accepts_common_interface(value: ProductReviewRetriever) -> ProductReviewRetriever:
        return value

    assert accepts_common_interface(retriever) is retriever
    results = retriever.retrieve("p1", "good", top_k=10)
    assert [item.review_id for item in results] == ["r1", "r2"]
    assert all(item.product_id == "p1" for item in results)
    assert [item.rank for item in results] == [1, 2]
    assert all(isinstance(item.score, float) and item.review_text_raw for item in results)
    assert retriever.retrieve("missing", "good", top_k=10) == []
    assert len(retriever.retrieve("p1", "battery", top_k=1)) == 1
