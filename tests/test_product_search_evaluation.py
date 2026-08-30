from __future__ import annotations

import json

import polars as pl

from digikala_comparison.config import Settings
from digikala_comparison.final_evaluation_v2 import FinalEvaluationV2Case, initialize_final_evaluation_v2
from digikala_comparison.product_search_evaluation import evaluate_product_search


def test_product_search_evaluation_requires_frozen_cases(tmp_path) -> None:
    settings = Settings.from_toml("config/default.toml")
    initialize_final_evaluation_v2(settings, output_root=tmp_path)
    try:
        evaluate_product_search(settings, output_root=tmp_path)
    except ValueError as error:
        assert "no frozen product_search cases" in str(error)
    else:
        raise AssertionError("evaluation must not invent product-search cases")


def test_search_case_contract_accepts_not_found_without_product_id() -> None:
    case = FinalEvaluationV2Case(
        case_id="search-invalid", component="product_search", split="final",
        scenario="invalid_product_name", user_input="zzzz", expected_status="not_found",
        reference={"expected_status": "not_found"},
    )
    assert case.product_ids == []


def test_search_case_contract_allows_only_explicit_empty_reference() -> None:
    empty = FinalEvaluationV2Case(
        case_id="search-empty", component="product_search", split="final",
        scenario="empty_reference", user_input="   ", expected_status="not_found",
        reference={"expected_status": "not_found"},
    )
    assert empty.user_input.isspace()
