"""Deterministic comparison evaluation imported from the existing frozen v1 suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from .comparison import CriterionRequest, PreferencePolicy, ProductComparisonService
from .config import Settings
from .final_evaluation import (
    FinalEvaluationCase,
    aggregate_grounding_results,
    deterministic_template_answer,
)
from .final_evaluation_v2 import FinalEvaluationV2Case, append_frozen_component_cases, initialize_final_evaluation_v2, load_v2_cases
from .generation import build_generation_context
from .grounding import DeterministicGroundingValidator
from .runtime import peak_process_memory_bytes


COMPARISON_PREDICTIONS_NAME = "comparison_predictions.json"
COMPARISON_METRICS_NAME = "comparison_metrics.json"
COMPARISON_HUMAN_TEMPLATE_NAME = "human_answer_quality_template.csv"


def build_comparison_cases(settings: Settings) -> list[FinalEvaluationV2Case]:
    """Import the already frozen v1 comparison inputs without altering them."""

    if settings.paths.final_evaluation_root is None:
        raise ValueError("final_evaluation_root is required to import frozen comparison cases")
    source = settings.paths.final_evaluation_root / "evaluation_cases.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    legacy = [FinalEvaluationCase.model_validate(item) for item in payload["cases"]]
    cases: list[FinalEvaluationV2Case] = []
    for item in legacy:
        if item.case_type != "comparison":
            continue
        cases.append(FinalEvaluationV2Case(
            case_id=f"comparison_{item.case_id}", component="comparison",
            split="development" if item.split == "development_debug" else "final",
            scenario=item.scenario, user_input=item.question,
            product_ids=item.product_ids, criteria=item.criteria,
            reference={
                "legacy_case_id": item.case_id,
                "expected_inconclusive": item.expects_inconclusive,
                "evidence_queries": item.evidence_queries,
                "preference_weights": item.preference_weights,
                "source_case_set": str(source),
            },
            notes=item.notes,
        ))
    if len(cases) < 2:
        raise RuntimeError("the legacy frozen case set has too few comparison cases")
    return cases


def freeze_comparison_cases(settings: Settings, *, output_root: Path | None = None) -> Path:
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, existing = load_v2_cases(root)
    if any(case.component == "comparison" for case in existing):
        return root / "evaluation_cases.json"
    return append_frozen_component_cases(settings, root=root, component="comparison", cases=build_comparison_cases(settings))


def evaluate_comparisons(settings: Settings, *, output_root: Path | None = None, evidence_top_k: int = 3) -> dict[str, Any]:
    if evidence_top_k <= 0:
        raise ValueError("evidence_top_k must be positive")
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, all_cases = load_v2_cases(root)
    cases = [case for case in all_cases if case.component == "comparison"]
    if not cases:
        raise ValueError("no frozen comparison cases; run digikala-freeze-comparison-evaluation first")
    service = ProductComparisonService.from_settings(settings)
    validator = DeterministicGroundingValidator(settings.grounding)
    predictions: list[dict[str, Any]] = []
    groundings = []
    latencies: list[float] = []
    inconclusive_correct = 0
    for case in cases:
        requests = [CriterionRequest(name=name, evidence_query=case.reference["evidence_queries"].get(name)) for name in case.criteria]
        policy = PreferencePolicy(weights=case.reference["preference_weights"]) if case.reference["preference_weights"] else None
        started = perf_counter()
        result = service.compare_product_ids(case.product_ids, requests, evidence_top_k=evidence_top_k, preference_policy=policy)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        context = build_generation_context(result, settings.generation, user_question=case.user_input)
        answer = deterministic_template_answer(result, context)
        grounding = validator.validate(answer, context)
        groundings.append(grounding)
        observed_inconclusive = any(decision.status == "inconclusive" for decision in result.criterion_decisions)
        expected_inconclusive = bool(case.reference["expected_inconclusive"])
        inconclusive_correct += observed_inconclusive == expected_inconclusive
        evidence_sets = [item for attachment in result.retrieved_evidence for item in attachment.evidence_sets]
        invariants = {
            "product_ownership": all(item.product_id == evidence.product_id for evidence in evidence_sets for item in evidence.evidence_items),
            "unique_review_ids_per_evidence_set": all(len({item.review_id for item in evidence.evidence_items}) == len(evidence.evidence_items) for evidence in evidence_sets),
            "full_statistics_separate_from_top_k_evidence": all(not hasattr(evidence, "recommended_percentage") for evidence in evidence_sets),
        }
        predictions.append({
            "case_id": case.case_id, "scenario": case.scenario, "product_ids": case.product_ids,
            "criteria": case.criteria, "expected_inconclusive": expected_inconclusive,
            "observed_inconclusive": observed_inconclusive, "inconclusive_expectation_met": observed_inconclusive == expected_inconclusive,
            "latency_ms": latency_ms, "comparison": result.model_dump(mode="json"),
            "deterministic_template_answer": answer.model_dump(mode="json"),
            "grounding": grounding.model_dump(mode="json"), "invariants": invariants,
        })
    metrics = {
        "schema_version": "comparison-evaluation-v2",
        "case_count": len(cases), "evidence_top_k": evidence_top_k,
        "inconclusive_expectation_accuracy": inconclusive_correct / len(cases),
        "evidence_invariant_case_ratio": sum(all(item["invariants"].values()) for item in predictions) / len(predictions),
        "grounding": aggregate_grounding_results(groundings),
        "latency": {"p50_ms": median(latencies), "p95_ms": sorted(latencies)[round((len(latencies) - 1) * .95)], "peak_process_memory_bytes": peak_process_memory_bytes()},
        "limitations": ["The imported frozen suite contains 9 comparison cases; expand to the planned 30–40 cases before final presentation.", "This run uses a deterministic template, not an LLM quality score; human answer-quality annotation remains required."],
    }
    predictions_path, metrics_path = root / COMPARISON_PREDICTIONS_NAME, root / COMPARISON_METRICS_NAME
    predictions_path.write_text(json.dumps({"schema_version": "comparison-predictions-v2", "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    template_path = root / COMPARISON_HUMAN_TEMPLATE_NAME
    if template_path.read_text(encoding="utf-8").count("\n") <= 1:
        fields = ("case_id", "response_id", "component", "annotator_id", "status", "relevance_1_to_5", "clarity_1_to_5", "source_separation_1_to_5", "recommendation_usefulness_1_to_5", "uncertainty_appropriateness_1_to_5", "citation_usefulness_1_to_5", "semantic_citation_support_1_to_5", "reference_correctness", "failure_category", "comments")
        with template_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for prediction in predictions:
                writer.writerow({"case_id": prediction["case_id"], "response_id": prediction["case_id"], "component": "comparison", "annotator_id": "", "status": "pending", "relevance_1_to_5": "", "clarity_1_to_5": "", "source_separation_1_to_5": "", "recommendation_usefulness_1_to_5": "", "uncertainty_appropriateness_1_to_5": "", "citation_usefulness_1_to_5": "", "semantic_citation_support_1_to_5": "", "reference_correctness": "", "failure_category": "", "comments": ""})
    return {"metrics": metrics, "paths": {"predictions": str(predictions_path), "metrics": str(metrics_path), "human_template": str(template_path)}}
