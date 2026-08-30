"""Deterministic aggregation of independently completed human annotations."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any


ANSWER_SCORE_COLUMNS = (
    "relevance_1_to_5", "clarity_1_to_5", "source_separation_1_to_5",
    "recommendation_usefulness_1_to_5", "uncertainty_appropriateness_1_to_5",
    "citation_usefulness_1_to_5", "semantic_citation_support_1_to_5",
    "reference_correctness",
)
QA_SCORE_COLUMNS = ("relevance_0_to_2",)
MANAGER_SCORE_COLUMNS = (
    "clarity_1_to_5", "actionability_1_to_5", "appropriate_uncertainty_1_to_5", "reference_correctness_1_to_5",
)


def aggregate_human_annotations(path: Path, *, score_columns: tuple[str, ...], minimum: float, maximum: float) -> dict[str, Any]:
    """Aggregate valid completed rows; agreement requires independent annotators."""

    if not path.is_file():
        return {"status": "missing_template", "path": str(path), "row_count": 0, "completed_row_count": 0, "valid_completed_row_count": 0}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows if (row.get("status") or "").strip().lower() == "completed"]
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in completed:
        case_id, annotator = (row.get("case_id") or "").strip(), (row.get("annotator_id") or "").strip()
        try:
            values = {column: float(row[column]) for column in score_columns}
            if not case_id or not annotator or any(value < minimum or value > maximum for value in values.values()):
                raise ValueError
            if (case_id, annotator) in seen:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            invalid.append({"case_id": case_id or "unknown", "annotator_id": annotator or "missing"})
            continue
        seen.add((case_id, annotator)); valid.append({"case_id": case_id, "annotator_id": annotator, "values": values})
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in valid: by_case.setdefault(row["case_id"], []).append(row)
    means = {column: (sum(row["values"][column] for row in valid) / len(valid) if valid else None) for column in score_columns}
    agreement: dict[str, Any] = {}
    for column in score_columns:
        pairs = [(left["values"][column], right["values"][column]) for rows_for_case in by_case.values() for left, right in combinations(rows_for_case, 2)]
        agreement[column] = {"pair_count": len(pairs), "exact_agreement": (sum(left == right for left, right in pairs) / len(pairs) if pairs else None), "mean_absolute_difference": (sum(abs(left - right) for left, right in pairs) / len(pairs) if pairs else None)}
    return {"status": "completed" if valid else "pending_human_annotation", "path": str(path), "scale": {"minimum": minimum, "maximum": maximum}, "row_count": len(rows), "completed_row_count": len(completed), "valid_completed_row_count": len(valid), "invalid_completed_rows": invalid, "annotator_count": len({row["annotator_id"] for row in valid}), "case_count_with_valid_annotations": len(by_case), "mean_scores": means, "inter_annotator_agreement": agreement, "agreement_note": "Exact agreement and mean absolute difference are pairwise descriptive statistics; they are null unless at least two independent annotators scored the same case."}


def aggregate_human_evaluation_bundle(root: Path) -> dict[str, Any]:
    return {
        "answer_quality": aggregate_human_annotations(root / "human_answer_quality_template.csv", score_columns=ANSWER_SCORE_COLUMNS, minimum=1, maximum=5),
        "qa_evidence_relevance": aggregate_human_annotations(root / "review_qa_evidence_relevance_template.csv", score_columns=QA_SCORE_COLUMNS, minimum=0, maximum=2),
        "manager_insight_quality": aggregate_human_annotations(root / "manager_insight_annotation_template.csv", score_columns=MANAGER_SCORE_COLUMNS, minimum=1, maximum=5),
    }
