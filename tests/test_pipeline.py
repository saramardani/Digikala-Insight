from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from digikala_comparison.config import (
    DatasetSettings,
    NormalizationSettings,
    PathSettings,
    ReviewEligibilitySettings,
    Settings,
)
from digikala_comparison.pipeline import run_preprocessing


def test_preprocessing_writes_parquet_and_quality_report(source_files, tmp_path: Path) -> None:
    products_path, comments_path = source_files
    settings = Settings(
        dataset=DatasetSettings("test-revision", "https://example.invalid/dataset"),
        paths=PathSettings(
            raw_products=products_path,
            raw_comments=comments_path,
            processed_products=tmp_path / "processed" / "products.parquet",
            processed_comments=tmp_path / "processed" / "comments.parquet",
            quality_report=tmp_path / "reports" / "quality.json",
        ),
        random_seed=42,
        normalization=NormalizationSettings("NFC", True, True, True, True),
        review_eligibility=ReviewEligibilitySettings(
            True, 1, False, ("recommended", "not_recommended", "no_idea")
        ),
    )

    artifacts = run_preprocessing(settings)

    assert all(path.is_file() for path in artifacts.values())
    comments = pl.read_parquet(artifacts["comments_parquet"])
    assert comments[0, "review_id"] == "007"
    assert comments[0, "review_text_normalized"] == "متن تست"
    assert comments[0, "advantages_items"] is None
    report = json.loads(artifacts["quality_report"].read_text(encoding="utf-8"))
    assert report["quality"]["products"]["rate"]["minimum"] == 90.0
