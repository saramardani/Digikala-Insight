"""Deterministic, auditable manager-analysis outputs and final-evaluation cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from .config import Settings
from .final_evaluation_v2 import FinalEvaluationV2Case, append_frozen_component_cases, initialize_final_evaluation_v2, load_v2_cases
from .runtime import peak_process_memory_bytes


MANAGER_PREDICTIONS_NAME = "manager_analysis_predictions.json"
MANAGER_METRICS_NAME = "manager_analysis_metrics.json"
MANAGER_HUMAN_TEMPLATE_NAME = "manager_insight_annotation_template.csv"


class ManagerAnalytics:
    """Queries typed full-product aggregates; review Top-K is never an input."""

    def __init__(self, settings: Settings) -> None:
        if any(path is None for path in (settings.paths.canonical_products, settings.paths.product_statistics, settings.paths.retrieval_corpus)):
            raise ValueError("canonical_products, product_statistics, and retrieval_corpus are required")
        self.canonical = pl.scan_parquet(settings.paths.canonical_products)  # type: ignore[arg-type]
        self.statistics = pl.scan_parquet(settings.paths.product_statistics)  # type: ignore[arg-type]
        self.retrieval = pl.scan_parquet(settings.paths.retrieval_corpus)  # type: ignore[arg-type]

    def high_volume_low_recommendation(self, limit: int) -> dict[str, Any]:
        rows = (
            self.canonical.select("product_id", "title_fa", "Brand", "Category1")
            .join(self.statistics.select("product_id", "total_review_count", "recommendation_known_count", "recommended_count", "recommended_percentage"), on="product_id")
            .filter((pl.col("recommendation_known_count") >= 20) & (pl.col("recommended_percentage") <= 0.40))
            .sort(["total_review_count", "recommended_percentage", "product_id"], descending=[True, False, False])
            .head(limit).collect().to_dicts()
        )
        return {"analysis_type": "high_volume_low_recommendation", "scope": "full_product_statistics", "definition": "recommendation_known_count >= 20 and recommended_percentage <= 0.40", "rows": rows}

    def lowest_category_satisfaction(self, limit: int) -> dict[str, Any]:
        rows = (
            self.canonical.select("product_id", "Category1")
            .join(self.statistics.select("product_id", "recommended_count", "recommendation_known_count"), on="product_id")
            .filter(pl.col("Category1").is_not_null() & (pl.col("recommendation_known_count") > 0))
            .group_by("Category1").agg(pl.col("recommended_count").sum().alias("recommended_count"), pl.col("recommendation_known_count").sum().alias("recommendation_known_count"), pl.len().alias("product_count"))
            .filter(pl.col("recommendation_known_count") >= 100)
            .with_columns((pl.col("recommended_count") / pl.col("recommendation_known_count")).alias("recommended_percentage"))
            .sort(["recommended_percentage", "recommendation_known_count", "Category1"], descending=[False, True, False]).head(limit).collect().to_dicts()
        )
        return {"analysis_type": "lowest_category_satisfaction", "scope": "full_product_statistics", "definition": "category aggregate; at least 100 known recommendation statuses", "rows": rows}

    def rate_recommendation_gaps(self, limit: int) -> dict[str, Any]:
        rows = (
            self.canonical.select("product_id", "title_fa", "Brand", "Category1", "Rate", "Rate_cnt", "has_metadata_conflict")
            .join(self.statistics.select("product_id", "recommendation_known_count", "recommended_percentage"), on="product_id")
            .filter((pl.col("Rate").is_not_null()) & (pl.col("Rate_cnt") >= 10) & (pl.col("recommendation_known_count") >= 20) & ~pl.col("has_metadata_conflict") & pl.col("recommended_percentage").is_not_null())
            .with_columns((pl.col("Rate") / 100.0).alias("snapshot_rate_fraction"))
            .with_columns((pl.col("snapshot_rate_fraction") - pl.col("recommended_percentage")).abs().alias("absolute_gap"))
            .sort(["absolute_gap", "recommendation_known_count", "product_id"], descending=[True, True, False]).head(limit).collect().to_dicts()
        )
        return {"analysis_type": "rate_recommendation_gap", "scope": "full_product_statistics + validated 0_to_100 product snapshot Rate", "definition": "absolute difference between Rate/100 and full-population recommended_percentage; this is a discrepancy signal, not a causal conclusion", "rows": rows}

    def frequent_disadvantages(self, limit: int) -> dict[str, Any]:
        categories = self.canonical.select("product_id", "Category1")
        rows = (
            self.retrieval.select("product_id", "disadvantages_items")
            .explode("disadvantages_items")
            .rename({"disadvantages_items": "disadvantage"})
            .filter(pl.col("disadvantage").is_not_null() & (pl.col("disadvantage").str.len_chars() >= 2))
            .join(categories, on="product_id", how="inner")
            .filter(pl.col("Category1").is_not_null())
            .group_by(["Category1", "disadvantage"]).len(name="mention_count")
            .sort(["mention_count", "Category1", "disadvantage"], descending=[True, False, False]).head(limit).collect().to_dicts()
        )
        return {"analysis_type": "frequent_disadvantages", "scope": "eligible_retrieval_review_corpus_only", "definition": "exact user-entered disadvantage items, not semantic aspect extraction and not the entire raw review population", "rows": rows}


def build_manager_analysis_cases() -> list[FinalEvaluationV2Case]:
    specs = (
        ("high_volume_low_recommendation", "کدام محصولات نظر زیاد اما درصد پیشنهاد خرید پایین دارند؟"),
        ("lowest_category_satisfaction", "کدام دسته‌ها کمترین رضایت کاربران را دارند؟"),
        ("rate_recommendation_gap", "کدام محصولات بین Rate ثبت‌شده و رضایت کاربران شکاف زیادی دارند؟"),
        ("frequent_disadvantages", "پرتکرارترین نکات منفی ثبت‌شده توسط کاربران چیست؟"),
    )
    cases: list[FinalEvaluationV2Case] = []
    for analysis_type, question in specs:
        for limit in (3, 5, 10, 15, 20):
            cases.append(FinalEvaluationV2Case(
                case_id=f"manager_{analysis_type}_{limit}", component="manager_analysis", split="final",
                scenario=analysis_type, user_input=question, reference={"analysis_type": analysis_type, "limit": limit, "expected_source": "eligible_retrieval_review_corpus_only" if analysis_type == "frequent_disadvantages" else "full_product_statistics"},
                notes="All ranking inputs and denominators are deterministic and retained in the output.",
            ))
    return cases


def freeze_manager_analysis_cases(settings: Settings, *, output_root: Path | None = None) -> Path:
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, existing = load_v2_cases(root)
    if any(case.component == "manager_analysis" for case in existing):
        return root / "evaluation_cases.json"
    return append_frozen_component_cases(settings, root=root, component="manager_analysis", cases=build_manager_analysis_cases())


def _write_human_template(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = ("case_id", "analysis_type", "annotator_id", "status", "clarity_1_to_5", "actionability_1_to_5", "appropriate_uncertainty_1_to_5", "reference_correctness_1_to_5", "comments")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in predictions:
            writer.writerow({"case_id": item["case_id"], "analysis_type": item["scenario"], "annotator_id": "", "status": "pending", "clarity_1_to_5": "", "actionability_1_to_5": "", "appropriate_uncertainty_1_to_5": "", "reference_correctness_1_to_5": "", "comments": ""})


def evaluate_manager_analysis(settings: Settings, *, output_root: Path | None = None) -> dict[str, Any]:
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, all_cases = load_v2_cases(root)
    cases = [case for case in all_cases if case.component == "manager_analysis"]
    if not cases:
        raise ValueError("no frozen manager_analysis cases; run digikala-freeze-manager-analysis-evaluation first")
    analytics = ManagerAnalytics(settings)
    functions = {"high_volume_low_recommendation": analytics.high_volume_low_recommendation, "lowest_category_satisfaction": analytics.lowest_category_satisfaction, "rate_recommendation_gap": analytics.rate_recommendation_gaps, "frequent_disadvantages": analytics.frequent_disadvantages}
    maximums = {name: max(int(case.reference["limit"]) for case in cases if case.reference["analysis_type"] == name) for name in functions}
    results: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    for name, function in functions.items():
        started = perf_counter(); results[name] = function(maximums[name]); timings[name] = (perf_counter() - started) * 1000
    predictions = []
    for case in cases:
        result = results[str(case.reference["analysis_type"])]
        rows = result["rows"][:int(case.reference["limit"])]
        # A case may require a source layer while an analysis truthfully
        # declares an additional direct snapshot source (Rate/100 gap).
        source_correct = str(case.reference["expected_source"]) in str(result["scope"])
        predictions.append({"case_id": case.case_id, "scenario": case.scenario, "question": case.user_input, "limit": case.reference["limit"], "source_scope_correct": source_correct, "result": {**result, "rows": rows}})
    template_path = root / MANAGER_HUMAN_TEMPLATE_NAME; _write_human_template(template_path, predictions)
    metrics = {"schema_version": "manager-analysis-evaluation-v2", "case_count": len(cases), "execution_success_ratio": 1.0, "source_scope_correct_ratio": sum(item["source_scope_correct"] for item in predictions) / len(predictions), "analysis_latency_ms": timings, "peak_process_memory_bytes": peak_process_memory_bytes(), "human_annotation": {"status": "pending_human_annotation", "template_path": str(template_path)}, "limitations": ["Frequent disadvantages uses explicit disadvantage items from the eligible retrieval corpus, not a semantic topic model or all raw reviews.", "Human actionability and clarity scores remain pending."]}
    predictions_path, metrics_path = root / MANAGER_PREDICTIONS_NAME, root / MANAGER_METRICS_NAME
    predictions_path.write_text(json.dumps({"schema_version": "manager-analysis-predictions-v2", "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": metrics, "paths": {"predictions": str(predictions_path), "metrics": str(metrics_path), "human_template": str(template_path)}}
