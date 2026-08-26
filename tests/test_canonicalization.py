from __future__ import annotations

import polars as pl

from digikala_comparison.canonicalization import (
    build_canonical_products,
    duplicate_conflict_report,
)


def _products() -> pl.LazyFrame:
    return pl.DataFrame(
        [
            {"id": "1", "title_fa": "A55", "Brand": "Samsung", "Category1": "mobile", "Category2": "phone", "sub_category": "mobile", "Price": 10, "Seller": "x", "Is_Fake": False, "Rate": 90, "Rate_cnt": 2, "min_price_last_month": 9},
            {"id": "1", "title_fa": "A55", "Brand": "Samsung", "Category1": "mobile", "Category2": "phone", "sub_category": "mobile", "Price": 10, "Seller": "x", "Is_Fake": False, "Rate": 90, "Rate_cnt": 2, "min_price_last_month": 9},
            {"id": "2", "title_fa": "A55", "Brand": "Samsung", "Category1": "mobile", "Category2": "phone", "sub_category": "mobile", "Price": 10, "Seller": "x", "Is_Fake": False, "Rate": 90, "Rate_cnt": 2, "min_price_last_month": 9},
            {"id": "2", "title_fa": "A55", "Brand": "Samsung", "Category1": "mobile", "Category2": "phone", "sub_category": "mobile", "Price": 11, "Seller": "y", "Is_Fake": False, "Rate": 90, "Rate_cnt": 2, "min_price_last_month": 9},
            {"id": "3", "title_fa": "A55", "Brand": "Samsung", "Category1": "mobile", "Category2": "phone", "sub_category": "mobile", "Price": 10, "Seller": "x", "Is_Fake": False, "Rate": 90, "Rate_cnt": 2, "min_price_last_month": 9},
            {"id": "3", "title_fa": "A35", "Brand": "Samsung", "Category1": "mobile", "Category2": "phone", "sub_category": "mobile", "Price": 10, "Seller": "x", "Is_Fake": False, "Rate": 90, "Rate_cnt": 2, "min_price_last_month": 9},
        ]
    ).lazy()


def test_canonicalization_preserves_duplicate_conflicts() -> None:
    canonical = build_canonical_products(_products())
    result = canonical.select(["product_id", "source_row_count", "canonicalization_status", "has_metadata_conflict"]).collect()
    assert result.filter(pl.col("product_id") == "1")[0, "canonicalization_status"] == "exact_duplicate_metadata"
    assert result.filter(pl.col("product_id") == "2")[0, "canonicalization_status"] == "same_identity_mutable_fields_differ"
    assert result.filter(pl.col("product_id") == "3")[0, "has_metadata_conflict"] is True
    report = duplicate_conflict_report(canonical)
    assert report["duplicate_product_id_count"] == 3
    assert report["metadata_conflict_count"] == 2
