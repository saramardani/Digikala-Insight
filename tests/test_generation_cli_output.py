from __future__ import annotations

from types import SimpleNamespace

from digikala_comparison.cli import _final_generation_summary, _write_generation_text_output


class _Outcome:
    rendered_persian = "پاسخ کامل فارسی"
    context_fingerprint = "a" * 64

    @staticmethod
    def model_dump_json(*, indent: int) -> str:
        assert indent == 2
        return '{\n  "metadata": "audit"\n}'


def test_generation_cli_writes_complete_utf8_output_and_prints_only_final_decision(tmp_path) -> None:
    path = _write_generation_text_output(tmp_path, _Outcome())

    assert path.parent == tmp_path
    assert path.suffix == ".txt"
    assert "پاسخ کامل فارسی" in path.read_text(encoding="utf-8")
    assert "structured metadata" in path.read_text(encoding="utf-8")

    result = SimpleNamespace(
        products=[SimpleNamespace(product_id="82098", title_fa="شامپو فولیکا")],
        overall=SimpleNamespace(winner_product_ids=["82098"], status="weighted_winner", reason_code="WEIGHTED_PREFERENCE_POLICY"),
    )
    assert _final_generation_summary(result) == (
        "نتیجه نهایی: شامپو فولیکا (82098) بر اساس اولویت‌ها و محاسبات قطعی سیستم پیشنهاد می‌شود."
    )


def test_generation_cli_summary_preserves_inconclusive_status() -> None:
    result = SimpleNamespace(
        products=[],
        overall=SimpleNamespace(winner_product_ids=[], status="inconclusive", reason_code="METADATA_CONFLICT"),
    )
    assert _final_generation_summary(result) == "نتیجه نهایی: برندهٔ قطعی اعلام نشد (inconclusive: METADATA_CONFLICT)."
