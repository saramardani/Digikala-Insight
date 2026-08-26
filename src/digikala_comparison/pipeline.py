"""Phase 1 orchestration: quality report plus Parquet conversion."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from .config import Settings
from .canonicalization import build_canonical_products, duplicate_conflict_report
from .ingestion import clean_comments, clean_products, load_comments, load_products
from .quality import analyze_canonical_quality
from .runtime import peak_process_memory_bytes
from .statistics import build_product_statistics, product_coverage_analysis


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _canonical_sources(settings: Settings) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if (
        settings.paths.processed_products.is_file()
        and settings.paths.processed_comments.is_file()
    ):
        return (
            pl.scan_parquet(settings.paths.processed_products),
            pl.scan_parquet(settings.paths.processed_comments),
        )
    return (
        clean_products(load_products(settings.paths.raw_products), settings.normalization),
        clean_comments(load_comments(settings.paths.raw_comments), settings.normalization),
    )


def generate_quality_report(settings: Settings) -> dict[str, Any]:
    started_at = perf_counter()
    products, comments = _canonical_sources(settings)
    report = {
        "dataset": asdict(settings.dataset),
        "random_seed": settings.random_seed,
        "quality": analyze_canonical_quality(
            products, comments, settings.review_eligibility
        ),
        "performance": {
            "runtime_seconds": perf_counter() - started_at,
            "peak_process_memory_bytes": peak_process_memory_bytes(),
        },
    }
    _write_json(settings.paths.quality_report, report)
    return report


def run_preprocessing(settings: Settings) -> dict[str, Path]:
    """Write cleaned Parquet datasets and a source-quality report."""
    started_at = perf_counter()
    products = load_products(settings.paths.raw_products)
    comments = load_comments(settings.paths.raw_comments)

    settings.paths.processed_products.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.processed_comments.parent.mkdir(parents=True, exist_ok=True)
    clean_products(products, settings.normalization).sink_parquet(
        settings.paths.processed_products
    )
    clean_comments(comments, settings.normalization).sink_parquet(
        settings.paths.processed_comments
    )
    report = generate_quality_report(settings)
    report["preprocessing_performance"] = {
        "runtime_seconds": perf_counter() - started_at,
        "peak_process_memory_bytes": peak_process_memory_bytes(),
    }
    _write_json(settings.paths.quality_report, report)
    return {
        "products_parquet": settings.paths.processed_products,
        "comments_parquet": settings.paths.processed_comments,
        "quality_report": settings.paths.quality_report,
    }


def build_statistics_artifact(settings: Settings) -> dict[str, Path]:
    """Persist full-population product statistics from canonical Parquet only."""
    if settings.paths.product_statistics is None or settings.paths.statistics_report is None:
        raise ValueError("product statistics paths must be configured")
    if not settings.paths.processed_products.is_file() or not settings.paths.processed_comments.is_file():
        raise FileNotFoundError(
            "Canonical Parquet files are required. Run digikala-preprocess first."
        )

    started_at = perf_counter()
    products = pl.scan_parquet(settings.paths.processed_products)
    reviews = pl.scan_parquet(settings.paths.processed_comments)
    statistics = build_product_statistics(products, reviews)
    settings.paths.product_statistics.parent.mkdir(parents=True, exist_ok=True)
    statistics.sink_parquet(settings.paths.product_statistics)

    persisted_statistics = pl.scan_parquet(settings.paths.product_statistics)
    example_rows = persisted_statistics.head(3).collect().to_dicts()
    report = {
        "dataset": asdict(settings.dataset),
        "statistics_row_count": persisted_statistics.select(pl.len()).collect()[0, 0],
        "product_statistics_path": str(settings.paths.product_statistics),
        "product_statistics_size_bytes": settings.paths.product_statistics.stat().st_size,
        "coverage": product_coverage_analysis(products, persisted_statistics),
        "example_rows": example_rows,
        "performance": {
            "runtime_seconds": perf_counter() - started_at,
            "peak_process_memory_bytes": peak_process_memory_bytes(),
        },
    }
    _write_json(settings.paths.statistics_report, report)
    return {
        "product_statistics": settings.paths.product_statistics,
        "statistics_report": settings.paths.statistics_report,
    }


def build_canonical_products_artifact(settings: Settings) -> dict[str, Path]:
    """Analyze duplicate snapshots and persist one conflict-aware record per ID."""
    required_paths = (
        settings.paths.canonical_products,
        settings.paths.duplicate_conflict_report,
        settings.paths.product_statistics,
    )
    if any(path is None for path in required_paths):
        raise ValueError("canonical product output paths must be configured")
    if not settings.paths.processed_products.is_file():
        raise FileNotFoundError("products.parquet is required. Run digikala-preprocess first.")
    if not settings.paths.product_statistics.is_file():
        raise FileNotFoundError(
            "product_statistics.parquet is required. Run digikala-build-statistics first."
        )

    started_at = perf_counter()
    products = pl.scan_parquet(settings.paths.processed_products)
    statistics = pl.scan_parquet(settings.paths.product_statistics).select(
        [
            "product_id",
            "total_review_count",
            "buyer_review_count",
            "recommendation_known_count",
        ]
    )
    canonical = build_canonical_products(products).join(
        statistics, on="product_id", how="left"
    )
    settings.paths.canonical_products.parent.mkdir(parents=True, exist_ok=True)
    canonical.sink_parquet(settings.paths.canonical_products)
    persisted = pl.scan_parquet(settings.paths.canonical_products)
    report = {
        "dataset": asdict(settings.dataset),
        "canonical_product_count": persisted.select(pl.len()).collect()[0, 0],
        "conflicted_canonical_product_count": persisted.select(
            pl.col("has_metadata_conflict").sum()
        ).collect()[0, 0],
        "canonical_products_path": str(settings.paths.canonical_products),
        "canonical_products_size_bytes": settings.paths.canonical_products.stat().st_size,
        "duplicate_analysis": duplicate_conflict_report(persisted),
        "performance": {
            "runtime_seconds": perf_counter() - started_at,
            "peak_process_memory_bytes": peak_process_memory_bytes(),
        },
    }
    _write_json(settings.paths.duplicate_conflict_report, report)
    return {
        "canonical_products": settings.paths.canonical_products,
        "duplicate_conflict_report": settings.paths.duplicate_conflict_report,
    }
