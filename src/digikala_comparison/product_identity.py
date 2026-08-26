"""Conservative product-title normalization and model-token protection."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
_DIGITS_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_TOKEN_PATTERN = re.compile(r"[a-z0-9+]+|[\u0600-\u06ff]+")


def normalize_product_text(value: Any) -> str | None:
    """Reduce superficial title variation without deleting variant information."""
    if value is None or not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFC", value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.translate(_ARABIC_TO_PERSIAN).translate(_DIGITS_TO_ASCII)
    text = text.replace("\u200c", " ").replace("–", "-").replace("—", "-")
    # Hyphen and slash are token separators; '+' remains part of e.g. Pro+.
    text = re.sub(r"\s*[-/]\s*", " ", text)
    text = re.sub(r"\s*\+\s*", "+", text)
    text = re.sub(r"(\d+)\s+(gb|tb)\b", r"\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text or None


def product_tokens(value: str | None) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(value or ""))


def model_tokens(value: str | None) -> frozenset[str]:
    """Return tokens whose mismatch makes a candidate unsafe to auto-resolve."""
    tokens = product_tokens(value)
    result: set[str] = set()
    for index, token in enumerate(tokens):
        if re.fullmatch(r"[a-z]+\d+[a-z0-9]*", token):  # a55, s24ultra
            result.add(token)
        elif re.fullmatch(r"\d+(?:gb|tb)", token):  # 128gb
            result.add(token)
        elif token in {"4g", "5g", "pro", "pro+", "max", "plus", "ultra", "mini"}:
            result.add(token)
        elif token.isdigit():  # Note 13 / Series 9 must not be conflated.
            result.add(token)
        elif token == "series" and index + 1 < len(tokens) and tokens[index + 1].isdigit():
            result.add(f"series{tokens[index + 1]}")
    return frozenset(result)
