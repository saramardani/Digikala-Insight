from __future__ import annotations

import polars as pl

from digikala_comparison.quality import analyze_data_quality


def test_duplicate_and_orphan_detection() -> None:
    products = pl.DataFrame(
        {
            "id": ["1", "1", "2"],
            "title_fa": ["الف", "الف", "ب"],
            "Rate": ["90", "90", "80"],
            "Rate_cnt": ["1", "1", "2"],
            "Category1": ["x", "x", "x"],
            "Category2": ["", "", ""],
            "Brand": ["b", "b", "b"],
            "Price": ["1", "1", "2"],
            "Seller": ["s", "s", "s"],
            "Is_Fake": ["False", "False", "False"],
            "min_price_last_month": ["0", "0", "0"],
            "sub_category": ["x", "x", "x"],
        }
    ).lazy()
    comments = pl.DataFrame(
        {
            "id": ["r1", "r1", "r2"],
            "title": ["t", "t", "t"],
            "body": ["ok", "nan", "ok"],
            "created_at": ["d", "d", "d"],
            "rate": ["1", "1", "1"],
            "recommendation_status": ["recommended", "recommended", "no_idea"],
            "is_buyer": ["True", "False", "True"],
            "product_id": ["1", "1", "missing"],
            "advantages": ["nan", "nan", "nan"],
            "disadvantages": ["nan", "nan", "nan"],
            "likes": ["0", "0", "0"],
            "dislikes": ["0", "0", "0"],
            "seller_title": ["s", "s", "s"],
            "seller_code": ["c", "c", "c"],
            "true_to_size_rate": ["nan", "nan", "nan"],
        }
    ).lazy()

    report = analyze_data_quality(products, comments)

    assert report["products"]["duplicate_ids"]["duplicate_id_values"] == 1
    assert report["comments"]["duplicate_ids"]["duplicate_rows_beyond_first"] == 1
    assert report["comments"]["string_nan_counts"]["body"] == 1
    assert report["join"]["orphan_product_id_review_rows"] == 1
