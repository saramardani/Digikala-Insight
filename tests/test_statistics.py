from __future__ import annotations

import polars as pl

from digikala_comparison.statistics import build_product_statistics


def test_full_population_statistics_keep_support_and_unknowns() -> None:
    products = pl.DataFrame({"id": ["p1", "p2"]}).lazy()
    reviews = pl.DataFrame(
        {
            "product_id": ["p1", "p1", "p1", "orphan"],
            "is_buyer_bool": [True, False, None, True],
            "review_rate_numeric": [4.0, 2.0, None, 5.0],
            "recommendation_status_normalized": [
                "recommended",
                "not_recommended",
                None,
                "recommended",
            ],
            "likes_numeric": [3.0, 1.0, None, 8.0],
            "dislikes_numeric": [0.0, 2.0, 1.0, 0.0],
        }
    ).lazy()

    statistics = build_product_statistics(products, reviews).sort("product_id").collect()
    first = statistics.row(0, named=True)
    second = statistics.row(1, named=True)

    assert first["total_review_count"] == 3
    assert first["buyer_review_count"] == 1
    assert first["non_buyer_review_count"] == 1
    assert first["unknown_buyer_review_count"] == 1
    assert first["recommended_count"] == 1
    assert first["not_recommended_count"] == 1
    assert first["recommendation_known_count"] == 2
    assert first["recommendation_unknown_count"] == 1
    assert first["opinionated_review_count"] == 2
    assert first["recommended_percentage"] == 0.5
    assert first["opinionated_recommend_percentage"] == 0.5
    assert first["total_likes"] == 4.0
    assert first["likes_valid_count"] == 2
    assert second["total_review_count"] == 0
    assert second["recommended_percentage"] is None
    assert second["opinionated_recommend_percentage"] is None


def test_orphan_reviews_do_not_create_product_statistics_rows() -> None:
    products = pl.DataFrame({"id": ["known"]}).lazy()
    reviews = pl.DataFrame(
        {
            "product_id": ["missing"],
            "is_buyer_bool": [True],
            "review_rate_numeric": [1.0],
            "recommendation_status_normalized": ["recommended"],
            "likes_numeric": [0.0],
            "dislikes_numeric": [0.0],
        }
    ).lazy()

    statistics = build_product_statistics(products, reviews).collect()

    assert statistics.height == 1
    assert statistics[0, "product_id"] == "known"
    assert statistics[0, "total_review_count"] == 0
