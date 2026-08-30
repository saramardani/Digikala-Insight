"""Evidence-first final evaluation for review-grounded product QA.

The repository has a production evidence retriever, but intentionally no
standalone QA text generator.  This evaluator therefore never fabricates an
answer: it freezes real review-QA questions, retrieves auditable evidence, and
creates the human relevance sheet required before retrieval quality is claimed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import polars as pl

from .config import Settings
from .evidence import ProductionEvidenceRetriever
from .final_evaluation_v2 import (
    FinalEvaluationV2Case,
    append_frozen_component_cases,
    initialize_final_evaluation_v2,
    load_v2_cases,
)
from .runtime import peak_process_memory_bytes


QA_PREDICTIONS_NAME = "review_qa_evidence_predictions.json"
QA_METRICS_NAME = "review_qa_evidence_metrics.json"
QA_RELEVANCE_TEMPLATE_NAME = "review_qa_evidence_relevance_template.csv"

_QUESTION_SPECS = (
    ("quality", "کیفیت", "کاربران درباره کیفیت این محصول چه گفته‌اند؟"),
    ("negative", "مشکلات و ایرادها", "کاربران چه ایرادها یا مشکلاتی را گزارش کرده‌اند؟"),
    ("satisfaction", "رضایت از خرید", "آیا کاربران از خرید این محصول راضی بوده‌اند؟"),
    ("value", "ارزش خرید", "کاربران درباره ارزش خرید این محصول چه نظری دارند؟"),
    ("positive", "نکات مثبت", "کاربران بیشتر از چه نکات مثبتی گفته‌اند؟"),
)


def _select_qa_products(settings: Settings) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if settings.paths.canonical_products is None or settings.paths.product_statistics is None:
        raise ValueError("canonical_products and product_statistics are required for review-QA evaluation")
    canonical = pl.scan_parquet(settings.paths.canonical_products).select("product_id", "Category1")
    statistics = pl.scan_parquet(settings.paths.product_statistics).select("product_id", "total_review_count")
    candidates = (
        canonical.join(statistics, on="product_id", how="inner")
        .filter(pl.col("total_review_count") >= 10)
        .sort(["total_review_count", "product_id"], descending=[True, False])
        .collect()
        .to_dicts()
    )
    selected: list[dict[str, Any]] = []
    categories: set[str] = set()
    for row in candidates:
        category = str(row.get("Category1") or "__missing__")
        if category not in categories:
            selected.append(row)
            categories.add(category)
        if len(selected) == 18:
            break
    if len(selected) < 18:
        raise RuntimeError("not enough category-diverse products with review support for review-QA evaluation")
    sparse = (
        statistics.filter((pl.col("total_review_count") > 0) & (pl.col("total_review_count") < 3))
        .sort(["total_review_count", "product_id"])
        .head(1)
        .collect()
        .to_dicts()
    )
    absent = (
        statistics.filter(pl.col("total_review_count") == 0)
        .sort("product_id")
        .head(1)
        .collect()
        .to_dicts()
    )
    if not sparse or not absent:
        raise RuntimeError("review-QA evaluation requires one sparse and one zero-review product")
    return selected, sparse[0], absent[0]


def build_review_qa_cases(settings: Settings) -> list[FinalEvaluationV2Case]:
    """Freeze 20 real product/question pairs, including insufficient-evidence cases."""

    selected, sparse, absent = _select_qa_products(settings)
    cases: list[FinalEvaluationV2Case] = []
    for index, product in enumerate(selected, start=1):
        criterion, query, question = _QUESTION_SPECS[(index - 1) % len(_QUESTION_SPECS)]
        product_id = str(product["product_id"])
        cases.append(FinalEvaluationV2Case(
            case_id=f"review_qa_supported_{index:02}", component="review_qa", split="final",
            scenario=f"supported_{criterion}", user_input=question, product_ids=[product_id], criteria=[criterion],
            reference={
                "retrieval_query": query,
                "evidence_expectation": "human_relevance_annotation_required",
                "full_population_statistics_allowed": False,
                "selection_basis": "category-diverse product with at least 10 full-population reviews",
            },
            notes="Top-K review evidence is for a review-experience claim only; it cannot estimate global satisfaction percentages.",
        ))
    for suffix, product, expectation, question in (
        ("sparse", sparse, "limited_or_no_evidence", "آیا برای کیفیت این محصول شواهد کافی از نظر کاربران وجود دارد؟"),
        ("none", absent, "no_evidence", "کاربران درباره کیفیت این محصول چه گفته‌اند؟"),
    ):
        cases.append(FinalEvaluationV2Case(
            case_id=f"review_qa_{suffix}_evidence", component="review_qa", split="final",
            scenario=f"{suffix}_evidence", user_input=question, product_ids=[str(product["product_id"])], criteria=["quality"],
            reference={
                "retrieval_query": "کیفیت",
                "evidence_expectation": expectation,
                "full_population_statistics_allowed": False,
                "selection_basis": f"full-product total_review_count={product['total_review_count']}",
            },
            notes="This case tests abstention/qualification, not forced answer completion.",
        ))
    if len(cases) != 20:
        raise AssertionError(f"expected 20 review-QA cases, got {len(cases)}")
    return cases


def freeze_review_qa_cases(settings: Settings, *, output_root: Path | None = None) -> Path:
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, existing = load_v2_cases(root)
    if any(case.component == "review_qa" for case in existing):
        return root / "evaluation_cases.json"
    return append_frozen_component_cases(settings, root=root, component="review_qa", cases=build_review_qa_cases(settings))


def _write_relevance_template(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = ("case_id", "product_id", "criterion", "review_id", "rank", "relevance_0_to_2", "annotator_id", "status", "comments")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            for item in prediction["evidence"]["evidence_items"]:
                writer.writerow({
                    "case_id": prediction["case_id"], "product_id": prediction["product_id"],
                    "criterion": prediction["criterion"], "review_id": item["review_id"],
                    "rank": item["rank"], "relevance_0_to_2": "", "annotator_id": "", "status": "pending", "comments": "",
                })


def evaluate_review_qa_evidence(settings: Settings, *, output_root: Path | None = None, top_k: int = 5) -> dict[str, Any]:
    """Retrieve review evidence and enforce provenance; human relevance remains pending."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, all_cases = load_v2_cases(root)
    cases = [case for case in all_cases if case.component == "review_qa"]
    if not cases:
        raise ValueError("no frozen review_qa cases; run digikala-freeze-review-qa-evaluation first")
    retriever_started = perf_counter()
    retriever = ProductionEvidenceRetriever.from_settings(settings)
    retriever_build_ms = (perf_counter() - retriever_started) * 1000
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    provenance_failures: list[str] = []
    constrained_expectation_correct = 0
    constrained_expectation_count = 0
    for case in cases:
        product_id = case.product_ids[0]
        started = perf_counter()
        evidence = retriever.retrieve_evidence(product_id, case.criteria[0], str(case.reference["retrieval_query"]), top_k=top_k)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        item_ids = [item.review_id for item in evidence.evidence_items]
        valid_ownership = all(item.product_id == product_id for item in evidence.evidence_items)
        valid_unique = len(item_ids) == len(set(item_ids))
        if not valid_ownership:
            provenance_failures.append(f"{case.case_id}:cross_product_evidence")
        if not valid_unique:
            provenance_failures.append(f"{case.case_id}:duplicate_review_id")
        expected = str(case.reference["evidence_expectation"])
        expectation_ok: bool | None = None
        if expected == "no_evidence":
            expectation_ok = evidence.retrieval_status == "no_evidence"
        elif expected == "limited_or_no_evidence":
            expectation_ok = evidence.retrieval_status in {"limited_candidates", "no_evidence"}
        if expectation_ok is not None:
            constrained_expectation_count += 1
            constrained_expectation_correct += expectation_ok
        predictions.append({
            "case_id": case.case_id, "scenario": case.scenario, "question": case.user_input,
            "product_id": product_id, "criterion": case.criteria[0], "reference": case.reference,
            "retrieval_latency_ms": latency_ms, "provenance_valid": valid_ownership and valid_unique,
            "evidence_expectation_met": expectation_ok, "evidence": evidence.model_dump(mode="json"),
            "answer_generation_status": "not_run_no_standalone_qa_generator",
        })
    predictions_path = root / QA_PREDICTIONS_NAME
    metrics_path = root / QA_METRICS_NAME
    relevance_path = root / QA_RELEVANCE_TEMPLATE_NAME
    _write_relevance_template(relevance_path, predictions)
    supported = [item for item in predictions if item["reference"]["evidence_expectation"] == "human_relevance_annotation_required"]
    metrics = {
        "schema_version": "review-qa-evidence-evaluation-v2",
        "case_count": len(cases), "top_k": top_k,
        "provenance_valid_case_count": sum(item["provenance_valid"] for item in predictions),
        "provenance_valid_case_ratio": sum(item["provenance_valid"] for item in predictions) / len(predictions),
        "supported_case_evidence_availability": sum(item["evidence"]["retrieval_status"] != "no_evidence" for item in supported) / len(supported),
        "constrained_low_evidence_case_count": constrained_expectation_count,
        "constrained_low_evidence_expectation_accuracy": constrained_expectation_correct / constrained_expectation_count if constrained_expectation_count else None,
        "retrieval_status_distribution": {status: sum(item["evidence"]["retrieval_status"] == status for item in predictions) for status in ("sufficient_candidates", "limited_candidates", "no_evidence")},
        "latency": {"retriever_build_ms": retriever_build_ms, "p50_ms": median(latencies), "p95_ms": sorted(latencies)[round((len(latencies) - 1) * .95)], "peak_process_memory_bytes": peak_process_memory_bytes()},
        "human_relevance": {"status": "pending_human_annotation", "template_path": str(relevance_path), "unit": "one retrieved review evidence item", "scale": "0=not relevant, 1=partly relevant, 2=directly relevant"},
        "grounding": {"status": "structural_provenance_checked", "checks": ["review_id is preserved", "all evidence belongs to requested product_id", "review_id is unique within an EvidenceSet", "Top-K evidence is not used as global product statistics"]},
        "limitations": ["No standalone review-QA answer generator exists in this repository, so no answer fluency/relevance score is claimed.", "Evidence availability is not evidence relevance; retrieval relevance metrics require the pending human labels in the emitted template."],
    }
    predictions_path.write_text(json.dumps({"schema_version": "review-qa-evidence-predictions-v2", "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": metrics, "paths": {"predictions": str(predictions_path), "metrics": str(metrics_path), "human_relevance_template": str(relevance_path)}}
