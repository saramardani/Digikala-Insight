from __future__ import annotations

from dataclasses import replace
import json

from digikala_comparison.comparison_evaluation_v2 import freeze_comparison_cases
from digikala_comparison.config import Settings
from digikala_comparison.final_evaluation_v2 import initialize_final_evaluation_v2, load_v2_cases


def test_comparison_cases_import_once_without_modifying_source(tmp_path) -> None:
    settings = Settings.from_toml("config/default.toml")
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    source = legacy_root / "evaluation_cases.json"
    source.write_text(json.dumps({"cases": [
        {"case_id": "a", "split": "final", "case_type": "comparison", "scenario": "price", "question": "compare", "product_ids": ["1", "2"], "criteria": ["price"]},
        {"case_id": "b", "split": "final", "case_type": "comparison", "scenario": "recommendation", "question": "compare", "product_ids": ["3", "4"], "criteria": ["recommendation"]},
    ]}), encoding="utf-8")
    settings = replace(settings, paths=replace(settings.paths, final_evaluation_root=legacy_root))
    root = initialize_final_evaluation_v2(settings, output_root=tmp_path)["root"]
    before = source.read_bytes()
    first = freeze_comparison_cases(settings, output_root=root)
    second = freeze_comparison_cases(settings, output_root=root)
    _, cases = load_v2_cases(root)
    assert first == second
    assert any(case.component == "comparison" for case in cases)
    assert source.read_bytes() == before
