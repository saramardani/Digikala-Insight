from __future__ import annotations

from pathlib import Path

import polars as pl

from digikala_comparison.bm25 import ProductScopedBM25
from digikala_comparison.config import BM25Settings
from digikala_comparison.retrieval_metrics import retrieval_metrics_at_k
from digikala_comparison.retrieval_pipeline import (
    retrieval_eligibility_expression,
    unique_retrieval_documents,
)
from digikala_comparison.retrieval_text import compose_retrieval_text, tokenize_persian_lexical


def _corpus(path: Path) -> Path:
    pl.DataFrame(
        [
            {"review_id": "r1", "product_id": "p1", "indexed_text_normalized": "کیفیت ساخت عالی", "bm25_tokens": ["کیفیت", "ساخت", "عالی"], "review_text_raw": "کیفیت ساخت عالی", "title_raw": None, "advantages_items": None, "disadvantages_items": None, "is_buyer_bool": True, "recommendation_status": "recommended", "review_rate_numeric": 5.0, "likes_numeric": 1.0, "dislikes_numeric": 0.0},
            {"review_id": "r2", "product_id": "p1", "indexed_text_normalized": "باتری ضعیف", "bm25_tokens": ["باتری", "ضعیف"], "review_text_raw": "باتری ضعیف", "title_raw": None, "advantages_items": None, "disadvantages_items": None, "is_buyer_bool": False, "recommendation_status": "not_recommended", "review_rate_numeric": 1.0, "likes_numeric": 0.0, "dislikes_numeric": 1.0},
            {"review_id": "r3", "product_id": "p2", "indexed_text_normalized": "کیفیت ساخت", "bm25_tokens": ["کیفیت", "ساخت"], "review_text_raw": "کیفیت ساخت", "title_raw": None, "advantages_items": None, "disadvantages_items": None, "is_buyer_bool": True, "recommendation_status": "recommended", "review_rate_numeric": 5.0, "likes_numeric": 0.0, "dislikes_numeric": 0.0},
        ]
    ).write_parquet(path)
    return path


def test_persian_tokenization_and_composition() -> None:
    assert tokenize_persian_lexical("كیفیت‌ ساخت ۱۲۸GB") == ["کیفیت", "ساخت", "128gb"]
    text = compose_retrieval_text({"title_normalized": "عنوان", "review_text_normalized": "متن", "advantages_items": ["خوب"], "disadvantages_items": ["خوب", "بد"]})
    assert text == "عنوان\nمتن\nخوب\nبد"


def test_bm25_product_filtering_provenance_and_determinism(tmp_path: Path) -> None:
    bm25 = ProductScopedBM25(_corpus(tmp_path / "corpus.parquet"), BM25Settings())
    first = bm25.retrieve("p1", "کیفیت ساخت", 10)
    second = bm25.retrieve("p1", "کیفیت ساخت", 1)
    assert [item.review_id for item in first] == ["r1"]
    assert second[0].review_id == "r1"
    assert all(item.product_id == "p1" for item in first)
    assert bm25.retrieve("unknown", "کیفیت", 10) == []
    assert bm25.retrieve("p1", "", 10) == []


def test_retrieval_metric_formulas_and_zero_qrels() -> None:
    metrics = retrieval_metrics_at_k(["x", "r2", "r1"], {"r1": 2, "r2": 1}, 3)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 2 / 3
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg"] is not None and 0 < metrics["ndcg"] <= 1
    assert retrieval_metrics_at_k(["r1"], {}, 10)["recall"] is None


def test_corpus_keeps_one_document_per_review_id() -> None:
    documents = pl.DataFrame(
        [
            {"review_id": "r1", "product_id": "p1", "value": "first"},
            {"review_id": "r1", "product_id": "p1", "value": "duplicate"},
            {"review_id": "r2", "product_id": "p1", "value": "other"},
        ]
    ).lazy()
    result = unique_retrieval_documents(documents).collect()
    assert result.to_dicts() == [
        {"review_id": "r1", "product_id": "p1", "value": "first"},
        {"review_id": "r2", "product_id": "p1", "value": "other"},
    ]


def test_retrieval_eligibility_requires_ids_body_length_and_tokens() -> None:
    reviews = pl.DataFrame(
        [
            {"review_id": "r1", "product_id": "p1", "review_text_normalized": "متن خوب", "bm25_tokens": ["متن"]},
            {"review_id": "", "product_id": "p1", "review_text_normalized": "متن خوب", "bm25_tokens": ["متن"]},
            {"review_id": "r3", "product_id": "p1", "review_text_normalized": "کوت", "bm25_tokens": ["کوت"]},
            {"review_id": "r4", "product_id": "p1", "review_text_normalized": None, "bm25_tokens": []},
            {"review_id": "r5", "product_id": "p1", "review_text_normalized": "متن خوب", "bm25_tokens": []},
        ]
    )
    eligible = reviews.lazy().filter(retrieval_eligibility_expression(4)).select("review_id").collect()
    assert eligible.get_column("review_id").to_list() == ["r1"]
