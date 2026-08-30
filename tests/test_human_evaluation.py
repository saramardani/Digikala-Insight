from __future__ import annotations

import csv

from digikala_comparison.human_evaluation import aggregate_human_annotations


def test_human_annotation_means_and_pairwise_agreement(tmp_path) -> None:
    path = tmp_path / "annotations.csv"
    fields = ["case_id", "annotator_id", "status", "score"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows([
            {"case_id": "a", "annotator_id": "one", "status": "completed", "score": "5"},
            {"case_id": "a", "annotator_id": "two", "status": "completed", "score": "4"},
            {"case_id": "b", "annotator_id": "one", "status": "completed", "score": "3"},
            {"case_id": "c", "annotator_id": "", "status": "completed", "score": "5"},
        ])
    result = aggregate_human_annotations(path, score_columns=("score",), minimum=1, maximum=5)
    assert result["valid_completed_row_count"] == 3
    assert result["mean_scores"]["score"] == 4.0
    assert result["inter_annotator_agreement"]["score"] == {"pair_count": 1, "exact_agreement": 0.0, "mean_absolute_difference": 1.0}
