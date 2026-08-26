"""Polars ingestion and loss-aware cleaning for the source CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from .config import NormalizationSettings
from .constants import COMMENT_REQUIRED_COLUMNS, PRODUCT_REQUIRED_COLUMNS
from .errors import DatasetPathError, RequiredColumnsError
from .normalization import (
    normalize_persian_text,
    parse_optional_bool,
    parse_serialized_text_list,
)
from .semantics import (
    finite_float,
    normalized_recommendation_status,
    recommendation_status_state,
)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise DatasetPathError(
            f"Dataset file not found: {path}. Place the pinned CSV at this path "
            "or update the configured path."
        )


def _scan_csv(path: Path, identifier_columns: Iterable[str]) -> pl.LazyFrame:
    """Scan strictly and force identifiers to strings so their values are preserved."""
    _require_file(path)
    schema_overrides = {column: pl.String for column in identifier_columns}
    return pl.scan_csv(
        path,
        schema_overrides=schema_overrides,
        infer_schema_length=10_000,
        try_parse_dates=False,
        ignore_errors=False,
        truncate_ragged_lines=False,
        raise_if_empty=True,
    )


def validate_required_columns(
    frame: pl.LazyFrame, required_columns: Iterable[str], dataset_name: str
) -> None:
    """Fail before processing when a documented source column is absent."""
    available = set(frame.collect_schema().names())
    missing = sorted(set(required_columns) - available)
    if missing:
        raise RequiredColumnsError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def load_products(path: Path) -> pl.LazyFrame:
    products = _scan_csv(path, identifier_columns=("id",))
    validate_required_columns(products, PRODUCT_REQUIRED_COLUMNS, "digikala-products.csv")
    return products


def load_comments(path: Path) -> pl.LazyFrame:
    comments = _scan_csv(path, identifier_columns=("id", "product_id"))
    validate_required_columns(comments, COMMENT_REQUIRED_COLUMNS, "digikala-comments.csv")
    return comments


def clean_products(
    products: pl.LazyFrame, settings: NormalizationSettings
) -> pl.LazyFrame:
    """Keep source columns and add only explicit normalized/parsed helpers."""
    return products.with_columns(
        [
            pl.col("title_fa")
            .map_elements(
                lambda value: normalize_persian_text(value, settings),
                return_dtype=pl.String,
            )
            .alias("title_fa_normalized"),
            pl.col("Is_Fake")
            .map_elements(parse_optional_bool, return_dtype=pl.Boolean)
            .alias("is_fake_bool"),
            finite_float("Price").alias("price_numeric"),
            finite_float("Rate").alias("product_rate_numeric"),
            finite_float("Rate_cnt").alias("product_rate_count_numeric"),
        ]
    )


def clean_comments(
    comments: pl.LazyFrame, settings: NormalizationSettings
) -> pl.LazyFrame:
    """Rename the review identifier and retain both raw and normalized review text."""
    return (
        comments.rename({"id": "review_id"})
        .with_columns(
            [
                pl.col("title").alias("title_raw"),
                pl.col("body").alias("review_text_raw"),
                pl.col("title")
                .map_elements(
                    lambda value: normalize_persian_text(value, settings),
                    return_dtype=pl.String,
                )
                .alias("title_normalized"),
                pl.col("body")
                .map_elements(
                    lambda value: normalize_persian_text(value, settings),
                    return_dtype=pl.String,
                )
                .alias("review_text_normalized"),
                pl.col("is_buyer")
                .map_elements(parse_optional_bool, return_dtype=pl.Boolean)
                .alias("is_buyer_bool"),
                normalized_recommendation_status().alias(
                    "recommendation_status_normalized"
                ),
                recommendation_status_state().alias("recommendation_status_state"),
                finite_float("rate").alias("review_rate_numeric"),
                finite_float("likes").alias("likes_numeric"),
                finite_float("dislikes").alias("dislikes_numeric"),
                pl.col("advantages")
                .map_elements(
                    parse_serialized_text_list, return_dtype=pl.List(pl.String)
                )
                .alias("advantages_items"),
                pl.col("disadvantages")
                .map_elements(
                    parse_serialized_text_list, return_dtype=pl.List(pl.String)
                )
                .alias("disadvantages_items"),
            ]
        )
    )
