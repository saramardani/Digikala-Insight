"""Duplicate analysis and deterministic canonical product identities."""

from __future__ import annotations

from typing import Any

import polars as pl

from .product_identity import normalize_product_text

IDENTITY_FIELDS = (
    "title_fa",
    "Brand",
    "Category1",
    "Category2",
    "sub_category",
)
MUTABLE_FIELDS = ("Price", "Seller", "Is_Fake", "Rate", "Rate_cnt", "min_price_last_month")
MEASURED_FIELDS = (*IDENTITY_FIELDS, *MUTABLE_FIELDS)


def _normalised_products(products: pl.LazyFrame) -> pl.LazyFrame:
    return products.with_columns(
        [
            pl.col("title_fa")
            .map_elements(normalize_product_text, return_dtype=pl.String)
            .alias("normalized_title"),
            pl.col("Brand")
            .map_elements(normalize_product_text, return_dtype=pl.String)
            .alias("normalized_brand"),
        ]
    )


def _mode_per_product(products: pl.LazyFrame, field: str) -> pl.LazyFrame:
    """Stable mode: highest count, then lexical representation as the tie-break."""
    return (
        products.group_by(["id", field])
        .len(name="value_count")
        .sort(["id", "value_count", field], descending=[False, True, False])
        .group_by("id", maintain_order=True)
        .agg(pl.col(field).first().alias(field))
    )


def build_canonical_products(products: pl.LazyFrame) -> pl.LazyFrame:
    """Return one auditable canonical record per product_id.

    The selected field values are per-field modes. This is deterministic and is
    not a claim that conflicting snapshots describe one unambiguous product.
    """
    source = _normalised_products(products)
    distinct_counts = [
        pl.col(field).drop_nulls().n_unique().alias(f"{field}_distinct_count")
        for field in MEASURED_FIELDS
    ]
    observed_fields = ("title_fa", "Brand", "Category1", "Category2", "sub_category", "Price", "Seller")
    observed = [
        pl.col(field).drop_nulls().unique().sort().alias(f"observed_{field}")
        for field in observed_fields
    ]
    canonical = source.group_by("id").agg(
        [pl.len().alias("source_row_count"), *distinct_counts, *observed]
    )
    for field in (
        "title_fa",
        "normalized_title",
        "normalized_brand",
        "Brand",
        "Category1",
        "Category2",
        "sub_category",
        *MUTABLE_FIELDS,
    ):
        canonical = canonical.join(_mode_per_product(source, field), on="id", how="left")

    identity_conflicts = [pl.col(f"{field}_distinct_count") > 1 for field in IDENTITY_FIELDS]
    category_brand_conflicts = [
        pl.col(f"{field}_distinct_count") > 1
        for field in ("Brand", "Category1", "Category2", "sub_category")
    ]
    mutable_conflicts = [pl.col(f"{field}_distinct_count") > 1 for field in MUTABLE_FIELDS]
    field_list = pl.concat_list(
        [
            pl.when(pl.col(f"{field}_distinct_count") > 1)
            .then(pl.lit(field))
            .otherwise(None)
            for field in MEASURED_FIELDS
        ]
    ).list.drop_nulls()
    all_agree = pl.all_horizontal(
        [pl.col(f"{field}_distinct_count") <= 1 for field in MEASURED_FIELDS]
    )
    any_identity_conflict = pl.any_horizontal(identity_conflicts)
    any_category_brand_conflict = pl.any_horizontal(category_brand_conflicts)
    any_mutable_conflict = pl.any_horizontal(mutable_conflicts)
    status = (
        pl.when(pl.col("source_row_count") == 1)
        .then(pl.lit("unique_source_row"))
        .when(all_agree)
        .then(pl.lit("exact_duplicate_metadata"))
        .when(any_category_brand_conflict)
        .then(pl.lit("identity_conflict_category_brand_title"))
        .when(any_identity_conflict)
        .then(pl.lit("identity_conflict_descriptive"))
        .when(any_mutable_conflict)
        .then(pl.lit("same_identity_mutable_fields_differ"))
        .otherwise(pl.lit("metadata_conflict"))
    )
    return (
        canonical.with_columns(
            [
                field_list.alias("conflicting_fields"),
                (pl.col("source_row_count") > 1).alias("has_duplicate_source_rows"),
                ((pl.col("source_row_count") > 1) & ~all_agree).alias(
                    "has_metadata_conflict"
                ),
                status.alias("canonicalization_status"),
            ]
        )
        .rename({"id": "product_id"})
    )


def duplicate_conflict_report(canonical_products: pl.LazyFrame) -> dict[str, Any]:
    """Summarize duplicate-ID metadata behavior without hiding disagreements."""
    duplicates = canonical_products.filter(pl.col("source_row_count") > 1)
    summary = duplicates.select(
        [
            pl.len().alias("duplicate_product_id_count"),
            (pl.col("canonicalization_status") == "exact_duplicate_metadata")
            .sum()
            .alias("exact_duplicate_metadata_count"),
            pl.col("has_metadata_conflict").sum().alias("metadata_conflict_count"),
            (pl.col("canonicalization_status") == "identity_conflict_descriptive")
            .sum()
            .alias("descriptive_identity_conflict_count"),
            (pl.col("canonicalization_status") == "identity_conflict_category_brand_title")
            .sum()
            .alias("category_brand_title_conflict_count"),
            (pl.col("canonicalization_status") == "same_identity_mutable_fields_differ")
            .sum()
            .alias("same_identity_mutable_fields_differ_count"),
        ]
    ).collect().to_dicts()[0]
    field_rows = (
        duplicates.select(["product_id", "conflicting_fields"])
        .explode("conflicting_fields", empty_as_null=True)
        .filter(pl.col("conflicting_fields").is_not_null())
        .group_by("conflicting_fields")
        .len(name="product_id_count")
        .sort("product_id_count", descending=True)
        .collect()
        .to_dicts()
    )
    total = int(summary["duplicate_product_id_count"])
    return {
        "duplicate_product_id_count": total,
        "exact_duplicate_metadata_count": int(summary["exact_duplicate_metadata_count"]),
        "exact_duplicate_metadata_percentage": None
        if total == 0
        else summary["exact_duplicate_metadata_count"] / total,
        "metadata_conflict_count": int(summary["metadata_conflict_count"]),
        "metadata_conflict_percentage": None
        if total == 0
        else summary["metadata_conflict_count"] / total,
        "classification_counts": {
            key.removesuffix("_count"): int(value)
            for key, value in summary.items()
            if key.endswith("_count")
            and key not in {"duplicate_product_id_count", "metadata_conflict_count"}
        },
        "conflicting_field_counts": field_rows,
        "canonicalization_policy": {
            "canonical_key": "product_id",
            "field_selection": "per-field mode; lexical ascending tie-break",
            "identity_conflicts": "preserved and marked; never treated as an unambiguous identity",
            "mutable_values": "not averaged, min/maxed, or otherwise merged",
        },
    }
