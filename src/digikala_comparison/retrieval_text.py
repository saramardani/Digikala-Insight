"""Transparent Persian lexical preprocessing for the BM25 baseline."""

from __future__ import annotations

import re
from typing import Any

from .product_identity import normalize_product_text

_TOKEN_PATTERN = re.compile(r"[a-z0-9+]+|[\u0600-\u06ff]+")
# Persisted corpus tokens are produced by this explicit, versioned tokenizer.
PERSIAN_LEXICAL_TOKENIZER_VERSION = "persian-lexical-v1"


def tokenize_persian_lexical(value: Any) -> list[str]:
    """Normalize and tokenize without stemming or undocumented stopword removal."""
    normalized = normalize_product_text(value)
    return _TOKEN_PATTERN.findall(normalized or "")


def compose_retrieval_text(fields: dict[str, Any]) -> str | None:
    """Compose each useful review field once, retaining its source boundaries."""
    values: list[str] = []
    for key in ("title_normalized", "review_text_normalized"):
        value = fields.get(key)
        if isinstance(value, str) and value.strip() and value not in values:
            values.append(value)
    for key in ("advantages_items", "disadvantages_items"):
        items = fields.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str) and item.strip() and item not in values:
                    values.append(item)
    return "\n".join(values) or None
