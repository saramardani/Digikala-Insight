from __future__ import annotations

from digikala_comparison.config import Settings
from digikala_comparison.final_evaluation_v2 import initialize_final_evaluation_v2, load_v2_cases
from digikala_comparison.manager_analysis_evaluation import build_manager_analysis_cases, freeze_manager_analysis_cases


def test_manager_case_set_has_four_auditable_analysis_types() -> None:
    cases = build_manager_analysis_cases()
    assert len(cases) == 20
    assert {case.scenario for case in cases} == {"high_volume_low_recommendation", "lowest_category_satisfaction", "rate_recommendation_gap", "frequent_disadvantages"}
    assert all(case.reference["expected_source"] for case in cases)


def test_manager_cases_freeze_once(tmp_path) -> None:
    settings = Settings.from_toml("config/default.toml")
    root = initialize_final_evaluation_v2(settings, output_root=tmp_path)["root"]
    freeze_manager_analysis_cases(settings, output_root=root)
    _, cases = load_v2_cases(root)
    assert len([case for case in cases if case.component == "manager_analysis"]) == 20
