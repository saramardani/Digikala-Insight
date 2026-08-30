"""Honest aggregation/reporting for the staged final-system evaluation v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .final_evaluation_v2 import initialize_final_evaluation_v2, load_v2_cases
from .human_evaluation import aggregate_human_evaluation_bundle


SUMMARY_NAME = "final_evaluation_summary.json"
REPORT_NAME = "final_evaluation_report.md"
PREDICTION_TEMPLATE_NAME = "recommendation_prediction_metrics_template.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _prediction_template(path: Path) -> None:
    if path.exists():
        return
    path.write_text(json.dumps({
        "schema_version": "recommendation-prediction-metrics-v1",
        "status": "pending_external_prediction_experiment",
        "test_split_definition": "",
        "test_row_count": None,
        "macro_f1": None,
        "class_f1": {"recommended": None, "not_recommended": None, "no_idea": None},
        "confusion_matrix": None,
        "error_analysis_path": "",
        "notes": "Populate only with a held-out test-set result. Do not use train or validation Macro-F1.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _prediction_status(path: Path | None, template_path: Path) -> dict[str, Any]:
    source = path if path is not None else template_path
    payload = _read_json(source)
    if payload is None or payload.get("macro_f1") is None:
        return {"status": "pending_prediction_metrics", "path": str(source), "macro_f1": None}
    score = payload["macro_f1"]
    if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        raise ValueError("prediction macro_f1 must be a number between 0 and 1")
    return {"status": "reported", "path": str(source), "macro_f1": float(score), "test_split_definition": payload.get("test_split_definition"), "test_row_count": payload.get("test_row_count"), "class_f1": payload.get("class_f1"), "confusion_matrix": payload.get("confusion_matrix"), "error_analysis_path": payload.get("error_analysis_path")}


def _markdown(summary: dict[str, Any]) -> str:
    counts = summary["case_counts"]
    lines = [
        "# گزارش ارزیابی نهایی سیستم — v2", "",
        "## وضعیت", "",
        f"- وضعیت گزارش: `{summary['status']}`",
        f"- تعداد کل caseهای frozen: {summary['case_count']}",
        f"- روش retrieval تولیدی: `{summary['production_retriever']}`",
        "", "## پوشش caseها", "",
        "| بخش | تعداد |", "|---|---:|",
        *[f"| {name} | {count} |" for name, count in counts.items()],
        "", "## نتایج خودکار", "",
        f"- جست‌وجو: accuracy={summary['metrics']['product_search'].get('all_case_accuracy')}; variant protection={summary['metrics']['product_search'].get('variant_protection_accuracy')}",
        f"- QA evidence: provenance={summary['metrics']['review_qa'].get('provenance_valid_case_ratio')}; availability={summary['metrics']['review_qa'].get('supported_case_evidence_availability')}",
        f"- مقایسه: grounded claim ratio={summary['metrics']['comparison'].get('grounding', {}).get('grounded_claim_ratio')}; inconclusive correctness={summary['metrics']['comparison'].get('inconclusive_expectation_accuracy')}",
        f"- تحلیل مدیریتی: execution={summary['metrics']['manager_analysis'].get('execution_success_ratio')}; source scope={summary['metrics']['manager_analysis'].get('source_scope_correct_ratio')}",
        "", "## پیش‌بینی recommendation_status", "",
        f"- وضعیت: `{summary['prediction']['status']}`",
        f"- Macro-F1: {summary['prediction'].get('macro_f1')}",
        "", "## موارد تکمیل‌نشده", "",
        *[f"- {item}" for item in summary["pending_items"]],
        "", "## محدودیت‌های گزارش", "",
        "- کیفیت retrieval از کیفیت متن تولیدی جدا گزارش شده است.",
        "- آمار کامل محصول از Top-K evidence محاسبه نشده است.",
        "- این گزارش هیچ API یا LLMی اجرا نکرده و هزینه‌ی جدیدی ایجاد نکرده است.",
    ]
    return "\n".join(lines) + "\n"


def build_final_report_v2(settings: Settings, *, output_root: Path | None = None, prediction_metrics_path: Path | None = None) -> dict[str, Any]:
    root = initialize_final_evaluation_v2(settings, output_root=output_root)["root"]
    _, cases = load_v2_cases(root)
    by_component = {name: sum(case.component == name for case in cases) for name in ("product_search", "review_qa", "comparison", "manager_analysis")}
    required = {"product_search": root / "product_search_metrics.json", "review_qa": root / "review_qa_evidence_metrics.json", "comparison": root / "comparison_metrics.json", "manager_analysis": root / "manager_analysis_metrics.json"}
    metrics = {name: (_read_json(path) or {"status": "missing_artifact", "path": str(path)}) for name, path in required.items()}
    prediction_template = root / PREDICTION_TEMPLATE_NAME; _prediction_template(prediction_template)
    prediction = _prediction_status(prediction_metrics_path, prediction_template)
    annotations = aggregate_human_evaluation_bundle(root)
    pending: list[str] = []
    if prediction["status"] != "reported": pending.append("Macro-F1، confusion matrix و تحلیل خطای prediction باید از test set وارد شوند.")
    if any(item["status"] != "completed" for item in annotations.values()): pending.append("فرم‌های annotation انسانی هنوز تکمیل نشده‌اند؛ امتیاز کیفیت/ارتباط انسانی گزارش نمی‌شود.")
    if by_component["comparison"] < 30: pending.append("تعداد caseهای comparison فعلاً کمتر از هدف ۳۰ تا ۴۰ مورد است.")
    summary = {"schema_version": "final-system-evaluation-summary-v2", "status": "completed_with_pending_human_and_prediction" if pending else "completed", "case_count": len(cases), "case_counts": by_component, "production_retriever": json.loads((root / "evaluation_manifest.json").read_text(encoding="utf-8"))["retrieval"]["selected_production_retriever"], "metrics": metrics, "annotations": annotations, "prediction": prediction, "cost": {"llm_execution_in_report": False, "api_calls": 0, "estimated_new_cost_usd": 0.0}, "pending_items": pending}
    summary_path, report_path = root / SUMMARY_NAME, root / REPORT_NAME
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return {"summary": summary, "paths": {"summary": str(summary_path), "report": str(report_path), "prediction_template": str(prediction_template)}}
