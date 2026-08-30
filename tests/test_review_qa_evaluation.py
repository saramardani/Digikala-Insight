from __future__ import annotations

from digikala_comparison.config import Settings
from digikala_comparison.final_evaluation_v2 import initialize_final_evaluation_v2
from digikala_comparison.review_qa_evaluation import evaluate_review_qa_evidence


def test_review_qa_evaluation_requires_frozen_cases(tmp_path) -> None:
    settings = Settings.from_toml("config/default.toml")
    initialize_final_evaluation_v2(settings, output_root=tmp_path)
    try:
        evaluate_review_qa_evidence(settings, output_root=tmp_path)
    except ValueError as error:
        assert "no frozen review_qa cases" in str(error)
    else:
        raise AssertionError("evaluation must not invent review-QA cases")
