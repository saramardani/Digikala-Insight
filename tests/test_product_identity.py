from __future__ import annotations

from digikala_comparison.product_identity import model_tokens, normalize_product_text


def test_product_normalization_preserves_variant_signals() -> None:
    assert normalize_product_text("  سامسونگ A55 / 5G - 256 GB ") == "سامسونگ a55 5g 256gb"
    assert normalize_product_text("Note 13 Pro+") == "note 13 pro+"
    assert normalize_product_text("Note 13 Pro") == "note 13 pro"
    assert normalize_product_text("nan") is None


def test_model_tokens_distinguish_variants() -> None:
    assert model_tokens("samsung a55 5g 128gb") != model_tokens("samsung a35 4g 256gb")
    assert "pro" in model_tokens("note 13 pro")
    assert "pro+" in model_tokens("note 13 pro+")
