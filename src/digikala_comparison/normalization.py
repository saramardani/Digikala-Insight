"""Deterministic Persian text normalization used by the data pipeline."""

from __future__ import annotations

import math
import re
import unicodedata
from ast import literal_eval
from typing import Any

from .config import NormalizationSettings

_WHITESPACE = re.compile(r"\s+")


def normalize_persian_text(
    value: Any, settings: NormalizationSettings
) -> str | None:
    """Normalize text without turning missing values into the literal "nan"."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value)
    if settings.treat_string_nan_as_null and text.strip().casefold() == "nan":
        return None

    text = unicodedata.normalize(settings.unicode_form, text)
    if settings.replace_arabic_yeh:
        text = text.replace("ي", "ی")
    if settings.replace_arabic_kaf:
        text = text.replace("ك", "ک")
    if settings.collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    text = text.strip()
    return text or None


def parse_optional_bool(value: Any) -> bool | None:
    """Parse only explicit Boolean strings; leave unknown values as missing."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def parse_serialized_text_list(value: Any) -> list[str] | None:
    """Safely parse list-like review fields while preserving unparseable text."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return None
    try:
        parsed = literal_eval(text)
    except (SyntaxError, ValueError):
        # Keep malformed source text as one item; the raw source column remains
        # available for audit and no unsafe evaluation is performed.
        return [text]

    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if item is not None]
    return [text]
