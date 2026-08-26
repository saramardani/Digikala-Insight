"""Shared contract for product-scoped review-evidence retrievers."""

from __future__ import annotations

from typing import Protocol

from .bm25 import RetrievedReview


class ProductReviewRetriever(Protocol):
    """All retrieval methods return evidence with review/product provenance."""

    def retrieve(
        self, product_id: str | int, query: str, top_k: int | None = None
    ) -> list[RetrievedReview]: ...
