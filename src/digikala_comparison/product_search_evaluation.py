"""Frozen final-evaluation cases and metrics for product search/discovery."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import polars as pl

from .config import Settings
from .final_evaluation_v2 import (
    FinalEvaluationV2Case,
    append_frozen_component_cases,
    initialize_final_evaluation_v2,
    load_v2_cases,
)
from .product_identity import normalize_product_text
from .resolution_validation import _minor_typo
from .resolver import ProductResolver
from .runtime import peak_process_memory_bytes


SEARCH_PREDICTIONS_NAME = "product_search_predictions.json"
SEARCH_METRICS_NAME = "product_search_metrics.json"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _unique_product_rows(products: pl.DataFrame) -> list[dict[str, Any]]:
    title_counts = products.group_by("normalized_title").len(name="title_count")
    rows = (
        products.join(title_counts, on="normalized_title")
        .filter(
            pl.col("normalized_title").is_not_null()
            & pl.col("title_fa").is_not_null()
            & (pl.col("title_count") == 1)
            & ~pl.col("canonicalization_status").str.starts_with("identity_conflict")
        )
        .select("product_id", "title_fa", "normalized_title", "normalized_brand", "Category1")
        .sort("product_id")
        .head(10_000)
        .to_dicts()
    )
    selected: list[dict[str, Any]] = []
    categories: set[str] = set()
    for row in rows:
        category = normalize_product_text(row.get("Category1")) or "__missing__"
        if category not in categories:
            selected.append(row)
            categories.add(category)
        if len(selected) == 8:
            return selected
    if len(selected) < 8:
        raise RuntimeError("not enough category-diverse, unique canonical products for product-search evaluation")
    return selected


def build_product_search_cases(settings: Settings) -> list[FinalEvaluationV2Case]:
    """Build 25 real-record cases by deterministic rules, not model outcomes."""

    if settings.paths.canonical_products is None:
        raise ValueError("canonical_products is required for product-search evaluation")
    products = pl.read_parquet(settings.paths.canonical_products)
    unique = _unique_product_rows(products)
    cases: list[FinalEvaluationV2Case] = []

    def add(
        suffix: str,
        scenario: str,
        user_input: str,
        *,
        expected_status: str,
        expected_product_id: str | None = None,
        resolver_input: object | None = None,
        notes: str = "",
    ) -> None:
        reference: dict[str, Any] = {"expected_status": expected_status}
        if expected_product_id is not None:
            reference["expected_product_id"] = expected_product_id
        if resolver_input is not None:
            reference["resolver_input"] = resolver_input
        cases.append(
            FinalEvaluationV2Case(
                case_id=f"search_{suffix}", component="product_search", split="final",
                scenario=scenario, user_input=user_input,
                product_ids=[expected_product_id] if expected_product_id else [],
                reference=reference, expected_status=expected_status, notes=notes,
            )
        )

    # Exact title, whitespace normalization, and one-character deletion cover
    # real Persian/Latin product names across eight top-level categories.
    for index, row in enumerate(unique[:6], start=1):
        add(f"exact_{index:02}", "exact_title", str(row["title_fa"]), expected_status="exact", expected_product_id=str(row["product_id"]))
    for index, row in enumerate(unique[2:5], start=1):
        add(f"spacing_{index:02}", "whitespace_normalization", f"  {row['normalized_title'].replace(' ', '   ')}  ", expected_status="exact", expected_product_id=str(row["product_id"]))
    for index, row in enumerate(unique[5:8], start=1):
        add(f"typo_{index:02}", "minor_typo", _minor_typo(str(row["normalized_title"])), expected_status="resolved", expected_product_id=str(row["product_id"]))

    model_rows = [row for row in unique if any(char.isdigit() for char in str(row["normalized_title"])) and row.get("normalized_brand")]
    if len(model_rows) < 4:
        raise RuntimeError("not enough brand/model records for product-search evaluation")
    for index, row in enumerate(model_rows[:4], start=1):
        structured = {"title": row["normalized_title"], "brand": row["normalized_brand"]}
        add(f"brand_model_{index:02}", "brand_and_model", f"{row['normalized_brand']} {row['normalized_title']}", expected_status="exact", expected_product_id=str(row["product_id"]), resolver_input=structured)
        add(f"wrong_variant_{index:02}", "variant_protection", f"{row['normalized_title']} z999model", expected_status="not_found", resolver_input={"title": f"{row['normalized_title']} z999model", "brand": row["normalized_brand"]}, notes="An impossible model token must prevent a nearby variant match.")

    duplicate_titles = (
        products.group_by("normalized_title").len(name="count")
        .filter(pl.col("normalized_title").is_not_null() & (pl.col("count") > 1))
        .sort("normalized_title")
        .head(3)
        .get_column("normalized_title")
        .to_list()
    )
    if len(duplicate_titles) < 3:
        raise RuntimeError("not enough duplicate canonical titles for ambiguity evaluation")
    for index, title in enumerate(duplicate_titles, start=1):
        add(f"ambiguous_{index:02}", "ambiguous_title", str(title), expected_status="ambiguous", notes="Multiple canonical product records share this normalized title.")

    add("invalid_01", "invalid_product_name", "zzzzzz محصول نامعتبر 987654321", expected_status="not_found")
    add("invalid_02", "empty_reference", "   ", expected_status="not_found")
    if len(cases) != 25:
        raise AssertionError(f"expected 25 deterministic product-search cases, got {len(cases)}")
    return cases


def freeze_product_search_cases(settings: Settings, *, output_root: Path | None = None) -> Path:
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, existing = load_v2_cases(root)
    found = [case for case in existing if case.component == "product_search"]
    if found:
        return root / "evaluation_cases.json"
    return append_frozen_component_cases(settings, root=root, component="product_search", cases=build_product_search_cases(settings))


def evaluate_product_search(settings: Settings, *, output_root: Path | None = None) -> dict[str, Any]:
    """Run frozen product-search cases and persist deterministic metrics."""

    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, all_cases = load_v2_cases(root)
    cases = [case for case in all_cases if case.component == "product_search"]
    if not cases:
        raise ValueError("no frozen product_search cases; run digikala-freeze-product-search-evaluation first")
    if settings.paths.canonical_products is None:
        raise ValueError("canonical_products is required")
    build_started = perf_counter()
    resolver = ProductResolver.from_parquet(str(settings.paths.canonical_products), settings.resolution)
    build_ms = (perf_counter() - build_started) * 1000
    outcomes: list[dict[str, Any]] = []
    for case in cases:
        query = case.reference.get("resolver_input", case.user_input)
        started = perf_counter()
        result = resolver.resolve(query)
        latency_ms = (perf_counter() - started) * 1000
        expected_product_id = case.reference.get("expected_product_id")
        expected_status = case.reference["expected_status"]
        status_correct = result.status == expected_status
        selected_correct = expected_product_id is None or result.selected_product_id == expected_product_id
        outcomes.append({
            "case_id": case.case_id, "scenario": case.scenario, "query": case.user_input,
            "expected_status": expected_status, "expected_product_id": expected_product_id,
            "status": result.status, "selected_product_id": result.selected_product_id,
            "top_candidate_product_id": result.candidates[0].product_id if result.candidates else None,
            "status_correct": status_correct, "selected_product_correct": selected_correct,
            "correct": status_correct and selected_correct, "latency_ms": latency_ms,
            "reason": result.reason,
        })
    positives = [item for item in outcomes if item["expected_product_id"] is not None]
    variant = [item for item in outcomes if item["scenario"] == "variant_protection"]
    abstentions = [item for item in outcomes if item["expected_status"] in {"ambiguous", "not_found"}]
    latencies = [float(item["latency_ms"]) for item in outcomes]
    scenario_metrics = {
        scenario: {"case_count": len(items), "correct_count": sum(item["correct"] for item in items), "accuracy": sum(item["correct"] for item in items) / len(items)}
        for scenario in sorted({item["scenario"] for item in outcomes})
        for items in [[item for item in outcomes if item["scenario"] == scenario]]
    }
    metrics = {
        "schema_version": "product-search-evaluation-v2",
        "case_count": len(outcomes),
        "all_case_accuracy": sum(item["correct"] for item in outcomes) / len(outcomes),
        "positive_resolution_accuracy": sum(item["correct"] for item in positives) / len(positives),
        "abstention_status_accuracy": sum(item["status_correct"] for item in abstentions) / len(abstentions),
        "variant_protection_accuracy": sum(item["correct"] for item in variant) / len(variant),
        "latency": {"resolver_build_ms": build_ms, "p50_ms": median(latencies), "p95_ms": _percentile(latencies, 0.95), "peak_process_memory_bytes": peak_process_memory_bytes()},
        "scenario_metrics": scenario_metrics,
        "limitations": ["This evaluates product-reference resolution, not semantic recommendation or product ranking.", "The frozen references are deterministic records from the canonical product artifact; no LLM judgment is used."],
    }
    predictions_path = root / SEARCH_PREDICTIONS_NAME
    metrics_path = root / SEARCH_METRICS_NAME
    predictions_path.write_text(json.dumps({"schema_version": "product-search-predictions-v2", "predictions": outcomes}, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": metrics, "paths": {"predictions": str(predictions_path), "metrics": str(metrics_path)}}
