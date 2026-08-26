"""Deterministic product statistics from the complete canonical review population."""

from __future__ import annotations

from typing import Any

import polars as pl


INITIAL_COUNT_COLUMNS = (
    "total_review_count",
    "buyer_review_count",
    "non_buyer_review_count",
    "unknown_buyer_review_count",
    "review_rate_valid_count",
    "recommended_count",
    "not_recommended_count",
    "no_idea_count",
    "recommendation_known_count",
    "likes_valid_count",
    "dislikes_valid_count",
)


def _count_where(expression: pl.Expr) -> pl.Expr:
    return expression.fill_null(False).cast(pl.Int64).sum()


def build_product_statistics(
    products: pl.LazyFrame, reviews: pl.LazyFrame
) -> pl.LazyFrame:
    """Build one row per known product from all matching canonical reviews.

    Future retrieval evidence is intentionally not an input to this function.
    """
    product_ids = products.select(pl.col("id").alias("product_id")).unique()
    matching_reviews = reviews.join(product_ids, on="product_id", how="inner")
    known_status = pl.col("recommendation_status_normalized").is_not_null()
    valid_review_rate = pl.col("review_rate_numeric").is_not_null()
    valid_likes = pl.col("likes_numeric").is_not_null()
    valid_dislikes = pl.col("dislikes_numeric").is_not_null()

    aggregates = matching_reviews.group_by("product_id").agg(
        [
            pl.len().alias("total_review_count"),
            _count_where(pl.col("is_buyer_bool") == True).alias("buyer_review_count"),  # noqa: E712
            _count_where(pl.col("is_buyer_bool") == False).alias("non_buyer_review_count"),  # noqa: E712
            _count_where(pl.col("is_buyer_bool").is_null()).alias(
                "unknown_buyer_review_count"
            ),
            _count_where(valid_review_rate).alias("review_rate_valid_count"),
            pl.col("review_rate_numeric").mean().alias("average_review_rate"),
            pl.col("review_rate_numeric").median().alias("median_review_rate"),
            _count_where(pl.col("recommendation_status_normalized") == "recommended").alias(
                "recommended_count"
            ),
            _count_where(
                pl.col("recommendation_status_normalized") == "not_recommended"
            ).alias("not_recommended_count"),
            _count_where(pl.col("recommendation_status_normalized") == "no_idea").alias(
                "no_idea_count"
            ),
            _count_where(known_status).alias("recommendation_known_count"),
            pl.col("likes_numeric").sum().alias("total_likes"),
            _count_where(valid_likes).alias("likes_valid_count"),
            pl.col("likes_numeric").mean().alias("average_likes"),
            pl.col("dislikes_numeric").sum().alias("total_dislikes"),
            _count_where(valid_dislikes).alias("dislikes_valid_count"),
            pl.col("dislikes_numeric").mean().alias("average_dislikes"),
        ]
    )

    statistics = product_ids.join(aggregates, on="product_id", how="left").with_columns(
        [
            pl.col(column).fill_null(0).cast(pl.Int64).alias(column)
            for column in INITIAL_COUNT_COLUMNS
        ]
    )
    statistics = statistics.with_columns(
        [
            (pl.col("total_review_count") - pl.col("review_rate_valid_count")).alias(
                "review_rate_invalid_or_missing_count"
            ),
            (
                pl.col("total_review_count") - pl.col("recommendation_known_count")
            ).alias("recommendation_unknown_count"),
            (
                pl.col("recommended_count") + pl.col("not_recommended_count")
            ).alias("opinionated_review_count"),
            (pl.col("total_review_count") - pl.col("likes_valid_count")).alias(
                "likes_invalid_or_missing_count"
            ),
            (pl.col("total_review_count") - pl.col("dislikes_valid_count")).alias(
                "dislikes_invalid_or_missing_count"
            ),
            pl.col("total_likes").fill_null(0.0),
            pl.col("total_dislikes").fill_null(0.0),
        ]
    )
    return statistics.with_columns(
        [
            pl.when(pl.col("total_review_count") > 0)
            .then(pl.col("buyer_review_count") / pl.col("total_review_count"))
            .otherwise(None)
            .alias("buyer_review_percentage"),
            pl.when(pl.col("recommendation_known_count") > 0)
            .then(pl.col("recommended_count") / pl.col("recommendation_known_count"))
            .otherwise(None)
            .alias("recommended_percentage"),
            pl.when(pl.col("recommendation_known_count") > 0)
            .then(
                pl.col("not_recommended_count") / pl.col("recommendation_known_count")
            )
            .otherwise(None)
            .alias("not_recommended_percentage"),
            pl.when(pl.col("recommendation_known_count") > 0)
            .then(pl.col("no_idea_count") / pl.col("recommendation_known_count"))
            .otherwise(None)
            .alias("no_idea_percentage"),
            pl.when(pl.col("opinionated_review_count") > 0)
            .then(pl.col("recommended_count") / pl.col("opinionated_review_count"))
            .otherwise(None)
            .alias("opinionated_recommend_percentage"),
        ]
    )


def _coverage_bucket() -> pl.Expr:
    return (
        pl.when(pl.col("total_review_count") == 0)
        .then(pl.lit("0"))
        .when(pl.col("total_review_count") <= 4)
        .then(pl.lit("1-4"))
        .when(pl.col("total_review_count") <= 9)
        .then(pl.lit("5-9"))
        .when(pl.col("total_review_count") <= 49)
        .then(pl.lit("10-49"))
        .when(pl.col("total_review_count") <= 99)
        .then(pl.lit("50-99"))
        .otherwise(pl.lit("100+"))
        .alias("review_coverage_bucket")
    )


def product_coverage_analysis(
    products: pl.LazyFrame, statistics: pl.LazyFrame
) -> dict[str, Any]:
    """Describe review-evidence coverage overall and by major product group."""
    stats_with_bucket = statistics.with_columns(_coverage_bucket())
    overall = stats_with_bucket.group_by("review_coverage_bucket").len(name="count").collect()
    overall_counts = {
        row["review_coverage_bucket"]: int(row["count"])
        for row in overall.iter_rows(named=True)
    }
    product_dimensions = (
        products.select(
            [
                pl.col("id").alias("product_id"),
                pl.col("Category1"),
                pl.col("sub_category"),
            ]
        )
        .unique(subset=["product_id"], keep="first")
    )
    by_category = (
        product_dimensions.join(stats_with_bucket, on="product_id", how="left")
        .group_by(["Category1", "review_coverage_bucket"])
        .len(name="count")
        .collect()
    )
    by_sub_category = (
        product_dimensions.join(stats_with_bucket, on="product_id", how="left")
        .group_by(["sub_category", "review_coverage_bucket"])
        .len(name="count")
        .collect()
    )
    return {
        "overall": [row for row in overall.iter_rows(named=True)],
        "summary": {
            "definition": "low evidence means one to four matching reviews",
            "total_products": sum(overall_counts.values()),
            "products_with_no_review_evidence": overall_counts.get("0", 0),
            "products_with_low_review_evidence": overall_counts.get("1-4", 0),
            "products_with_zero_to_nine_reviews": sum(
                overall_counts.get(bucket, 0) for bucket in ("0", "1-4", "5-9")
            ),
        },
        "by_category1": [row for row in by_category.iter_rows(named=True)],
        "by_sub_category": [row for row in by_sub_category.iter_rows(named=True)],
    }
