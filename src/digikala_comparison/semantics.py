"""Explicit, conservative normalizations for observed raw dataset values."""

from __future__ import annotations

import polars as pl

from .constants import VALID_RECOMMENDATION_STATUSES


def finite_float(column: str) -> pl.Expr:
    """Return a finite Float64 value or null without assigning semantic meaning."""
    numeric = pl.col(column).cast(pl.Float64, strict=False)
    return pl.when(numeric.is_finite()).then(numeric).otherwise(None)


def normalized_recommendation_status(column: str = "recommendation_status") -> pl.Expr:
    raw = pl.col(column).cast(pl.String)
    normalized = raw.str.strip_chars().str.to_lowercase()
    return pl.when(normalized.is_in(VALID_RECOMMENDATION_STATUSES)).then(
        normalized
    ).otherwise(None)


def recommendation_status_state(column: str = "recommendation_status") -> pl.Expr:
    raw = pl.col(column).cast(pl.String)
    normalized = raw.str.strip_chars().str.to_lowercase()
    missing = raw.is_null() | normalized.eq("") | normalized.eq("nan")
    return (
        pl.when(missing)
        .then(pl.lit("missing"))
        .when(normalized.is_in(VALID_RECOMMENDATION_STATUSES))
        .then(pl.lit("known"))
        .otherwise(pl.lit("unknown"))
    )
