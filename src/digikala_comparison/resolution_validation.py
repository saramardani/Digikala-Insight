"""Small deterministic real-data validation for product-reference resolution."""

from __future__ import annotations

from statistics import median
from time import perf_counter
from typing import Any

import polars as pl

from .config import Settings
from .resolver import ProductResolver
from .runtime import peak_process_memory_bytes


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = round((len(ordered) - 1) * percentile)
    return ordered[position]


def _minor_typo(text: str) -> str:
    for index, character in enumerate(text):
        if character.isalpha() and index + 1 < len(text):
            return text[:index] + text[index + 1 :]
    return text


def run_resolution_validation(settings: Settings) -> dict[str, Any]:
    if settings.paths.canonical_products is None or settings.paths.resolution_validation_report is None:
        raise ValueError("resolver validation paths must be configured")
    products = pl.read_parquet(settings.paths.canonical_products)
    title_counts = products.group_by("normalized_title").len(name="title_count")
    pool = (
        products.join(title_counts, on="normalized_title")
        .filter(
            pl.col("normalized_title").is_not_null()
            & (pl.col("title_count") == 1)
            & ~pl.col("canonicalization_status").str.starts_with("identity_conflict")
        )
        .sort("product_id")
        .head(40)
    )
    index_started = perf_counter()
    resolver = ProductResolver(products, settings.resolution)
    resolver.index_build_seconds = perf_counter() - index_started
    cases: list[dict[str, Any]] = []
    for row in pool.iter_rows(named=True):
        title = row["title_fa"]
        normalized = row["normalized_title"]
        cases.extend(
            [
                {"kind": "exact", "query": title, "expected_product_id": row["product_id"]},
                {
                    "kind": "spacing",
                    "query": f"  {normalized.replace(' ', '   ')}  ",
                    "expected_product_id": row["product_id"],
                },
                {
                    "kind": "minor_typo",
                    "query": _minor_typo(normalized),
                    "expected_product_id": row["product_id"],
                },
            ]
        )
    # Negative cases deliberately use an impossible model token. They must not
    # be confidently mapped to a product merely because surrounding title words match.
    model_rows = pool.filter(pl.col("normalized_title").str.contains(r"[A-Za-z]+[0-9]+"))
    for row in model_rows.head(10).iter_rows(named=True):
        cases.append(
            {
                "kind": "wrong_model_negative",
                "query": f"{row['normalized_title']} z999model",
                "expected_product_id": None,
            }
        )

    outcomes: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        started = perf_counter()
        result = resolver.resolve(case["query"])
        latencies.append((perf_counter() - started) * 1000)
        top_id = result.candidates[0].product_id if result.candidates else None
        expected = case["expected_product_id"]
        outcomes.append(
            {
                **case,
                "status": result.status,
                "selected_product_id": result.selected_product_id,
                "top_candidate_product_id": top_id,
                "correct_selected": expected is not None and result.selected_product_id == expected,
                "correct_top_candidate": expected is not None and top_id == expected,
                "false_positive": expected is None and result.status in {"exact", "resolved"},
                "latency_ms": latencies[-1],
            }
        )
    positive = [outcome for outcome in outcomes if outcome["expected_product_id"] is not None]
    exact_cases = [outcome for outcome in outcomes if outcome["kind"] == "exact"]
    report = {
        "validation_definition": {
            "positive_cases": "exact, whitespace-normalized, and one-character-deletion queries derived from unique real canonical titles",
            "negative_cases": "real model-bearing titles with an impossible appended model token",
            "priority": "minimize false confident matches; ambiguous is preferable to wrong resolution",
        },
        "case_count": len(outcomes),
        "positive_case_count": len(positive),
        "exact_resolution_accuracy": (
            None
            if not exact_cases
            else sum(item["correct_selected"] for item in exact_cases) / len(exact_cases)
        ),
        "top_1_accuracy": (
            None if not positive else sum(item["correct_top_candidate"] for item in positive) / len(positive)
        ),
        "ambiguous_rate": sum(item["status"] == "ambiguous" for item in outcomes) / len(outcomes),
        "false_positive_resolution_rate": sum(item["false_positive"] for item in outcomes)
        / len(outcomes),
        "not_found_rate": sum(item["status"] == "not_found" for item in outcomes) / len(outcomes),
        "performance": {
            "index_build_seconds": resolver.index_build_seconds,
            "p50_resolution_latency_ms": median(latencies) if latencies else None,
            "p95_resolution_latency_ms": _percentile(latencies, 0.95),
            "peak_process_memory_bytes": peak_process_memory_bytes(),
        },
        "representative_outcomes": outcomes[:12],
        "representative_ambiguous_or_failure_cases": [
            item for item in outcomes if item["status"] in {"ambiguous", "not_found"}
        ][:12],
    }
    settings.paths.resolution_validation_report.parent.mkdir(parents=True, exist_ok=True)
    import json

    settings.paths.resolution_validation_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
