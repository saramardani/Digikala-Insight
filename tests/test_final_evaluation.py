from __future__ import annotations

import csv

from digikala_comparison.final_evaluation import (
    FinalEvaluationCase,
    aggregate_grounding_results,
    aggregate_human_annotations,
    write_human_annotation_template,
)
from digikala_comparison.grounding import (
    GroundingMetrics,
    GroundingValidationResult,
)


def _grounding_result(*, grounded: int, unsupported: int, valid_citations: int) -> GroundingValidationResult:
    total = grounded + unsupported
    return GroundingValidationResult(
        validator_version="test",
        overall_status="accepted" if unsupported == 0 else "rejected",
        claim_results=[],
        citation_results=[],
        unsupported_claim_count=unsupported,
        grounded_claim_count=grounded,
        action_taken="accepted" if unsupported == 0 else "rejected",
        metrics=GroundingMetrics(
            factual_claim_count=total,
            grounded_claim_count=grounded,
            unsupported_claim_count=unsupported,
            contradiction_count=unsupported,
            grounded_claim_ratio=grounded / total if total else None,
            unsupported_claim_ratio=unsupported / total if total else None,
            contradiction_rate=unsupported / total if total else None,
            citation_count=2,
            valid_citation_count=valid_citations,
            citation_correctness=valid_citations / 2,
            support_requiring_claim_count=total,
            evidence_covered_claim_count=grounded,
            evidence_coverage=grounded / total if total else None,
            inconclusive_case_count=1,
            correct_inconclusive_count=1,
            inconclusive_correctness=1.0,
        ),
    )


def test_final_case_requires_distinct_product_ids_and_resolver_reference() -> None:
    comparison = FinalEvaluationCase(
        case_id="comparison",
        split="final",
        case_type="comparison",
        scenario="price",
        question="مقایسه کن",
        product_ids=["a", "b"],
        criteria=["price"],
    )
    assert comparison.product_ids == ["a", "b"]

    resolver = FinalEvaluationCase(
        case_id="resolver",
        split="final",
        case_type="resolver",
        scenario="ambiguous",
        question="پیدا کن",
        resolver_reference="عنوان",
    )
    assert resolver.resolver_reference == "عنوان"


def test_grounding_aggregate_pools_counts_instead_of_averaging_ratios() -> None:
    summary = aggregate_grounding_results(
        [_grounding_result(grounded=1, unsupported=0, valid_citations=2), _grounding_result(grounded=1, unsupported=3, valid_citations=1)]
    )

    assert summary["factual_claim_count"] == 5
    assert summary["grounded_claim_ratio"] == 0.4
    assert summary["citation_correctness"] == 0.75
    assert summary["inconclusive_correctness"] == 1.0


def test_human_template_remains_pending_until_independent_annotations_exist(tmp_path) -> None:
    template = tmp_path / "human_answer_quality_template.csv"
    predictions = [
        {
            "case_id": "case-1",
            "scenario": "review_evidence",
            "case_type": "comparison",
            "final": {"status": "completed"},
        }
    ]
    write_human_annotation_template(template, predictions, sample_size=5)
    assert aggregate_human_annotations(tmp_path / "human_answer_quality_annotations.csv")["status"] == "pending_human_annotation"

    annotation = tmp_path / "human_answer_quality_annotations.csv"
    with template.open(encoding="utf-8", newline="") as source, annotation.open("w", encoding="utf-8", newline="") as destination:
        rows = list(csv.DictReader(source))
        rows[0].update(
            {
                "annotator_id": "annotator-1",
                "status": "completed",
                "relevance_1_to_5": "5",
                "clarity_1_to_5": "4",
                "source_separation_1_to_5": "5",
                "recommendation_usefulness_1_to_5": "4",
                "uncertainty_appropriateness_1_to_5": "5",
                "citation_usefulness_1_to_5": "4",
                "semantic_citation_support_1_to_5": "5",
            }
        )
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate_human_annotations(annotation)

    assert summary["status"] == "completed"
    assert summary["valid_completed_row_count"] == 1
    assert summary["mean_scores"]["relevance_1_to_5"] == 5.0
