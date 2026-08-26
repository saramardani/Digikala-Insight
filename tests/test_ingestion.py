from __future__ import annotations

import csv

import polars as pl
import pytest

from digikala_comparison.config import NormalizationSettings
from digikala_comparison.errors import RequiredColumnsError
from digikala_comparison.ingestion import clean_comments, load_comments, load_products


SETTINGS = NormalizationSettings("NFC", True, True, True, True)


def test_required_column_validation_rejects_missing_column(tmp_path) -> None:
    path = tmp_path / "broken-products.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "title_fa"])
        writer.writerow(["1", "کالا"])

    with pytest.raises(RequiredColumnsError, match="Rate"):
        load_products(path)


def test_id_values_are_preserved_and_comments_use_review_id(source_files) -> None:
    products_path, comments_path = source_files
    products = load_products(products_path).collect()
    comments = clean_comments(load_comments(comments_path), SETTINGS).collect()

    assert products[0, "id"] == "001"
    assert comments[0, "review_id"] == "007"
    assert comments[0, "product_id"] == "001"
    assert "id" not in comments.columns


def test_normalized_review_text_handles_string_nan(source_files) -> None:
    _, comments_path = source_files
    comments = load_comments(comments_path).with_columns(
        pl.col("body").replace("متن تست", "nan")
    )
    cleaned = clean_comments(comments, SETTINGS).collect()

    assert cleaned[0, "review_text_raw"] == "nan"
    assert cleaned[0, "review_text_normalized"] is None
