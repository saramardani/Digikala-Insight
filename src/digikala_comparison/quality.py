"""Deterministic, source-oriented data-quality analysis."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from .constants import COMMENT_REQUIRED_COLUMNS, PRODUCT_REQUIRED_COLUMNS
from .config import ReviewEligibilitySettings
from .ingestion import validate_required_columns


def _scalar(frame: pl.LazyFrame, expression: pl.Expr) -> Any:
    return frame.select(expression).collect()[0, 0]


def _row_count(frame: pl.LazyFrame) -> int:
    return int(_scalar(frame, pl.len()))


def _null_counts(frame: pl.LazyFrame, columns: Iterable[str]) -> dict[str, int]:
    selected = list(columns)
    result = frame.select(
        [pl.col(column).null_count().alias(column) for column in selected]
    ).collect()
    return {column: int(result[0, column]) for column in selected}


def _string_nan_counts(frame: pl.LazyFrame, columns: Iterable[str]) -> dict[str, int]:
    schema = frame.collect_schema()
    string_columns = [
        column
        for column in columns
        if schema[column] in (pl.String, pl.Utf8)
    ]
    if not string_columns:
        return {}
    result = frame.select(
        [
            pl.col(column)
            .str.strip_chars()
            .str.to_lowercase()
            .eq("nan")
            .fill_null(False)
            .sum()
            .alias(column)
            for column in string_columns
        ]
    ).collect()
    return {column: int(result[0, column]) for column in string_columns}


def _duplicate_summary(frame: pl.LazyFrame, identifier: str) -> dict[str, int]:
    duplicates = (
        frame.group_by(identifier)
        .len(name="occurrences")
        .filter(pl.col("occurrences") > 1)
        .collect()
    )
    return {
        "duplicate_id_values": int(duplicates.height),
        "duplicate_rows_beyond_first": int(
            sum(int(value) - 1 for value in duplicates["occurrences"].to_list())
        ),
    }


def _distribution(frame: pl.LazyFrame, column: str) -> dict[str, int]:
    rows = frame.group_by(column).len(name="count").sort(column).collect()
    result: dict[str, int] = {}
    for row in rows.iter_rows(named=True):
        key = "<null>" if row[column] is None else str(row[column])
        result[key] = int(row["count"])
    return result


def _distribution_summary(
    frame: pl.LazyFrame, column: str, top_n: int = 20
) -> dict[str, Any]:
    values = (
        frame.group_by(column)
        .len(name="count")
        .sort("count", descending=True)
        .head(top_n)
        .collect()
    )
    unique_non_null = _scalar(
        frame, pl.col(column).drop_nulls().n_unique().alias("unique")
    )
    return {
        "unique_non_null_count": int(unique_non_null),
        "top_values": [
            {"value": row[column], "count": int(row["count"])}
            for row in values.iter_rows(named=True)
        ],
    }


def _numeric_summary(
    frame: pl.LazyFrame, raw_column: str, numeric_column: str
) -> dict[str, Any]:
    summary = frame.select(
        [
            pl.col(numeric_column).min().alias("minimum"),
            pl.col(numeric_column).max().alias("maximum"),
            pl.col(numeric_column).is_not_null().sum().alias("valid_count"),
            pl.col(numeric_column).is_null().sum().alias("invalid_or_missing_count"),
        ]
    ).collect()
    return {
        "minimum": summary[0, "minimum"],
        "maximum": summary[0, "maximum"],
        "valid_count": int(summary[0, "valid_count"]),
        "invalid_or_missing_count": int(summary[0, "invalid_or_missing_count"]),
        "raw_distribution_summary": _distribution_summary(frame, raw_column),
    }


def _reviews_per_product_summary(reviews: pl.LazyFrame) -> dict[str, Any]:
    grouped = reviews.group_by("product_id").len(name="review_count")
    summary = grouped.select(
        [
            pl.len().alias("products_referenced"),
            pl.col("review_count").min().alias("minimum"),
            pl.col("review_count").max().alias("maximum"),
            pl.col("review_count").median().alias("median"),
        ]
    ).collect()
    return {
        "products_referenced": int(summary[0, "products_referenced"]),
        "minimum": summary[0, "minimum"],
        "maximum": summary[0, "maximum"],
        "median": summary[0, "median"],
        "bucket_counts": _distribution(
            grouped.with_columns(
                pl.when(pl.col("review_count") == 1)
                .then(pl.lit("1"))
                .when(pl.col("review_count") <= 4)
                .then(pl.lit("2-4"))
                .when(pl.col("review_count") <= 9)
                .then(pl.lit("5-9"))
                .when(pl.col("review_count") <= 49)
                .then(pl.lit("10-49"))
                .when(pl.col("review_count") <= 99)
                .then(pl.lit("50-99"))
                .otherwise(pl.lit("100+"))
                .alias("review_count_bucket")
            ),
            "review_count_bucket",
        ),
    }


def _eligibility_analysis(
    reviews: pl.LazyFrame, settings: ReviewEligibilitySettings
) -> dict[str, int]:
    non_empty = pl.col("review_text_normalized").is_not_null()
    minimum_length = pl.col("review_text_normalized").str.len_chars() >= (
        settings.minimum_normalized_text_length
    )
    buyer = pl.col("is_buyer_bool").fill_null(False)
    valid_status = pl.col("recommendation_status_normalized").is_not_null()
    allowed_status = pl.col("recommendation_status_normalized").is_in(
        settings.allowed_recommendation_status
    )
    valid_rate = pl.col("review_rate_numeric").is_not_null()
    helpfulness = (
        pl.col("likes_numeric").fill_null(0) + pl.col("dislikes_numeric").fill_null(0)
    ) >= settings.minimum_helpfulness_votes
    result = reviews.select(
        [
            pl.len().alias("all_reviews"),
            non_empty.sum().alias("non_empty_normalized_body"),
            buyer.sum().alias("buyer_only"),
            (non_empty & minimum_length).sum().alias("minimum_text_length"),
            valid_status.sum().alias("valid_recommendation_status"),
            allowed_status.sum().alias("allowed_recommendation_status"),
            valid_rate.sum().alias("valid_rate"),
            helpfulness.sum().alias("minimum_helpfulness_votes"),
            (
                (non_empty if settings.require_nonempty_normalized_text else pl.lit(True))
                & minimum_length
                & (buyer if settings.require_buyer else pl.lit(True))
                & allowed_status
                & valid_rate
                & helpfulness
            )
            .sum()
            .alias("configured_candidate_combination"),
        ]
    ).collect()
    return {column: int(result[0, column]) for column in result.columns}


def analyze_canonical_quality(
    products: pl.LazyFrame,
    reviews: pl.LazyFrame,
    eligibility_settings: ReviewEligibilitySettings,
) -> dict[str, Any]:
    """Quality report for canonical Parquet containing semantic helper columns."""
    canonical_comment_columns = (
        "review_id",
        *[column for column in COMMENT_REQUIRED_COLUMNS if column != "id"],
    )
    validate_required_columns(products, PRODUCT_REQUIRED_COLUMNS, "canonical products")
    validate_required_columns(reviews, canonical_comment_columns, "canonical reviews")

    product_rows = _row_count(products)
    review_rows = _row_count(reviews)
    product_ids = products.select("id").unique()
    orphan_reviews = reviews.join(
        product_ids, left_on="product_id", right_on="id", how="anti"
    )
    orphan_review_rows = _row_count(orphan_reviews)
    products_with_reviews = _row_count(
        products.select("id")
        .unique()
        .join(reviews.select("product_id").unique(), left_on="id", right_on="product_id")
    )
    canonical_product_columns = [*PRODUCT_REQUIRED_COLUMNS]
    canonical_review_columns = [*canonical_comment_columns]

    return {
        "products": {
            "row_count": product_rows,
            "unique_product_id_count": int(_scalar(products, pl.col("id").n_unique())),
            "null_counts": _null_counts(products, canonical_product_columns),
            "string_nan_counts": _string_nan_counts(products, canonical_product_columns),
            "duplicate_ids": _duplicate_summary(products, "id"),
            "brand_distribution_summary": _distribution_summary(products, "Brand"),
            "category1_distribution_summary": _distribution_summary(products, "Category1"),
            "category2_distribution_summary": _distribution_summary(products, "Category2"),
            "sub_category_distribution_summary": _distribution_summary(products, "sub_category"),
            "price": _numeric_summary(products, "Price", "price_numeric"),
            "rate": _numeric_summary(products, "Rate", "product_rate_numeric"),
            "rate_count": _numeric_summary(products, "Rate_cnt", "product_rate_count_numeric"),
            "is_fake_raw_distribution": _distribution_summary(products, "Is_Fake"),
            "is_fake_normalized_distribution": _distribution(products, "is_fake_bool"),
        },
        "reviews": {
            "row_count": review_rows,
            "unique_review_id_count": int(_scalar(reviews, pl.col("review_id").n_unique())),
            "null_counts": _null_counts(reviews, canonical_review_columns),
            "string_nan_counts": _string_nan_counts(reviews, canonical_review_columns),
            "duplicate_ids": _duplicate_summary(reviews, "review_id"),
            "recommendation_status_raw_distribution": _distribution_summary(
                reviews, "recommendation_status"
            ),
            "recommendation_status_normalized_distribution": _distribution(
                reviews, "recommendation_status_normalized"
            ),
            "recommendation_status_state_distribution": _distribution(
                reviews, "recommendation_status_state"
            ),
            "buyer_raw_distribution": _distribution_summary(reviews, "is_buyer"),
            "buyer_normalized_distribution": _distribution(reviews, "is_buyer_bool"),
            "review_rate": _numeric_summary(reviews, "rate", "review_rate_numeric"),
            "likes": _numeric_summary(reviews, "likes", "likes_numeric"),
            "dislikes": _numeric_summary(reviews, "dislikes", "dislikes_numeric"),
            "reviews_per_product": _reviews_per_product_summary(reviews),
            "eligibility_analysis": _eligibility_analysis(reviews, eligibility_settings),
        },
        "join": {
            "orphan_product_id_review_rows": orphan_review_rows,
            "review_join_coverage": (
                None if review_rows == 0 else (review_rows - orphan_review_rows) / review_rows
            ),
            "products_with_at_least_one_review": products_with_reviews,
            "product_join_coverage": (
                None if product_rows == 0 else products_with_reviews / product_rows
            ),
        },
    }


def _rate_summary(products: pl.LazyFrame) -> dict[str, Any]:
    numeric_rate = pl.col("Rate").cast(pl.Float64, strict=False)
    finite_rate = pl.when(numeric_rate.is_finite()).then(numeric_rate)
    summary = products.select(
        [
            finite_rate.min().alias("minimum"),
            finite_rate.max().alias("maximum"),
            finite_rate.null_count().alias("null_or_unparseable_count"),
        ]
    ).collect()
    return {
        "minimum": summary[0, "minimum"],
        "maximum": summary[0, "maximum"],
        "null_or_unparseable_count": int(summary[0, "null_or_unparseable_count"]),
        # Values are reported, not interpreted as a rating scale.
        "raw_value_distribution": _distribution(products, "Rate"),
    }


def analyze_data_quality(
    products: pl.LazyFrame, comments: pl.LazyFrame
) -> dict[str, Any]:
    """Return reproducible facts about source data without silently filtering it."""
    validate_required_columns(products, PRODUCT_REQUIRED_COLUMNS, "products")
    validate_required_columns(comments, COMMENT_REQUIRED_COLUMNS, "comments")

    product_rows = _row_count(products)
    review_rows = _row_count(comments)
    product_ids = products.select("id").unique()
    orphan_reviews = comments.join(
        product_ids, left_on="product_id", right_on="id", how="anti"
    )
    orphan_review_rows = _row_count(orphan_reviews)
    products_with_reviews = _row_count(
        products.join(
            comments.select("product_id").unique(),
            left_on="id",
            right_on="product_id",
            how="inner",
        )
    )

    return {
        "products": {
            "row_count": product_rows,
            "null_counts": _null_counts(products, PRODUCT_REQUIRED_COLUMNS),
            "string_nan_counts": _string_nan_counts(products, PRODUCT_REQUIRED_COLUMNS),
            "duplicate_ids": _duplicate_summary(products, "id"),
            "rate": _rate_summary(products),
        },
        "comments": {
            "row_count": review_rows,
            "null_counts": _null_counts(comments, COMMENT_REQUIRED_COLUMNS),
            "string_nan_counts": _string_nan_counts(comments, COMMENT_REQUIRED_COLUMNS),
            "duplicate_ids": _duplicate_summary(comments, "id"),
            "recommendation_status_distribution": _distribution(
                comments, "recommendation_status"
            ),
            "buyer_distribution": _distribution(comments, "is_buyer"),
        },
        "join": {
            "orphan_product_id_review_rows": orphan_review_rows,
            "review_join_coverage": (
                None if review_rows == 0 else (review_rows - orphan_review_rows) / review_rows
            ),
            "products_with_at_least_one_review": products_with_reviews,
            "product_join_coverage": (
                None if product_rows == 0 else products_with_reviews / product_rows
            ),
        },
    }
