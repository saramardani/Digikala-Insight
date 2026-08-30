from __future__ import annotations

import json

import pytest

from digikala_comparison.config import Settings
from digikala_comparison.final_evaluation_v2 import (
    FinalEvaluationV2Case,
    initialize_final_evaluation_v2,
)


def test_v2_comparison_case_requires_two_product_ids_and_reference() -> None:
    case = FinalEvaluationV2Case(
        case_id="comparison-1",
        component="comparison",
        split="final",
        scenario="clear_price",
        user_input="مقایسه کن",
        product_ids=["10", "20"],
        criteria=["price"],
        reference={"winner": "10"},
    )
    assert case.product_ids == ["10", "20"]

    with pytest.raises(ValueError, match="at least two"):
        FinalEvaluationV2Case(
            case_id="comparison-2", component="comparison", split="final", scenario="bad",
            user_input="مقایسه کن", product_ids=["10"], reference={"winner": "10"},
        )


def test_v2_foundation_initialization_is_idempotent_and_never_persists_secret(tmp_path) -> None:
    settings = Settings.from_toml("config/default.toml")
    paths = initialize_final_evaluation_v2(settings, output_root=tmp_path / "final_v2")
    again = initialize_final_evaluation_v2(settings, output_root=tmp_path / "final_v2")

    assert paths == again
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    case_set = json.loads(paths["case_set"].read_text(encoding="utf-8"))
    assert manifest["status"] == "foundation_initialized_pending_cases"
    assert manifest["generation"]["secret_value_persisted"] is False
    assert manifest["generation"]["api_key_environment_variable"] == "METIS_API_KEY"
    assert case_set["cases"] == []
    assert paths["human_template"].read_text(encoding="utf-8").startswith("case_id,response_id,component")
