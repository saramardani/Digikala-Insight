from digikala_comparison.config import NormalizationSettings
from digikala_comparison.normalization import (
    normalize_persian_text,
    parse_serialized_text_list,
)


SETTINGS = NormalizationSettings(
    unicode_form="NFC",
    replace_arabic_yeh=True,
    replace_arabic_kaf=True,
    collapse_whitespace=True,
    treat_string_nan_as_null=True,
)


def test_normalizes_arabic_letters_and_whitespace() -> None:
    assert normalize_persian_text("  يكي\nكالا  ", SETTINGS) == "یکی کالا"


def test_handles_null_and_string_nan() -> None:
    assert normalize_persian_text(None, SETTINGS) is None
    assert normalize_persian_text(" nan ", SETTINGS) is None
    assert normalize_persian_text("   ", SETTINGS) is None


def test_safely_parses_serialized_review_lists() -> None:
    assert parse_serialized_text_list("['سبک', 'خوش‌دست']") == ["سبک", "خوش‌دست"]
    assert parse_serialized_text_list("not valid python[") == ["not valid python["]
    assert parse_serialized_text_list("nan") is None
