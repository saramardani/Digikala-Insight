from __future__ import annotations

import json

from digikala_comparison.config import Settings
from digikala_comparison.final_report_v2 import build_final_report_v2


def test_final_report_marks_missing_prediction_as_pending(tmp_path) -> None:
    settings = Settings.from_toml("config/default.toml")
    root = tmp_path / "final_v2"; root.mkdir()
    (root / "evaluation_cases.json").write_text(json.dumps({"schema_version": "x", "cases": []}), encoding="utf-8")
    (root / "evaluation_manifest.json").write_text(json.dumps({"retrieval": {"selected_production_retriever": "bm25"}}), encoding="utf-8")
    result = build_final_report_v2(settings, output_root=root)
    assert result["summary"]["prediction"]["status"] == "pending_prediction_metrics"
    assert result["summary"]["cost"]["api_calls"] == 0
    assert (root / "recommendation_prediction_metrics_template.json").is_file()
