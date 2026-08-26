"""Deterministic, lexical product-reference resolution."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz, process

from .config import ResolutionSettings
from .product_identity import model_tokens, normalize_product_text, product_tokens


class ProductReference(BaseModel):
    """Structured boundary for a future query parser; it does not parse prose."""

    model_config = ConfigDict(extra="forbid")
    product_id: str | int | None = None
    title: str | None = None
    brand: str | None = None
    category: str | None = None


class ResolutionCandidate(BaseModel):
    product_id: str
    title_fa: str | None
    Brand: str | None
    Category1: str | None
    Category2: str | None
    sub_category: str | None
    score: float
    match_reasons: list[str]
    canonicalization_status: str
    total_review_count: int | None = None
    buyer_review_count: int | None = None
    recommendation_known_count: int | None = None


class ResolutionResult(BaseModel):
    query: str
    status: Literal["exact", "resolved", "ambiguous", "not_found"]
    selected_product_id: str | None = None
    confidence: float = 0.0
    candidates: list[ResolutionCandidate] = Field(default_factory=list)
    reason: str


class ProductResolver:
    """An in-memory index built once from canonical products, never review text."""

    FIELDS = (
        "product_id",
        "title_fa",
        "normalized_title",
        "normalized_brand",
        "Brand",
        "Category1",
        "Category2",
        "sub_category",
        "canonicalization_status",
        "total_review_count",
        "buyer_review_count",
        "recommendation_known_count",
    )

    def __init__(self, products: pl.DataFrame, settings: ResolutionSettings) -> None:
        missing = set(self.FIELDS) - set(products.columns)
        if missing:
            raise ValueError(f"Canonical product table lacks resolver columns: {sorted(missing)}")
        self.settings = settings
        selected = products.select(self.FIELDS)
        self.columns = {name: selected.get_column(name).to_list() for name in self.FIELDS}
        self.id_index: dict[str, int] = {}
        self.title_index: dict[str, list[int]] = defaultdict(list)
        # Lists have materially lower overhead than a set per token at this
        # scale. Query-time candidate sets are created only after narrowing.
        self.brand_index: dict[str, list[int]] = defaultdict(list)
        self.token_index: dict[str, list[int]] = defaultdict(list)
        self.model_index: dict[str, list[int]] = defaultdict(list)
        for row_index, product_id in enumerate(self.columns["product_id"]):
            self.id_index[str(product_id)] = row_index
            title = self.columns["normalized_title"][row_index]
            if title:
                self.title_index[title].append(row_index)
                for token in product_tokens(title):
                    if len(token) >= 2:
                        self.token_index[token].append(row_index)
                for token in model_tokens(title):
                    self.model_index[token].append(row_index)
            brand = self.columns["normalized_brand"][row_index]
            if brand:
                self.brand_index[brand].append(row_index)

    @classmethod
    def from_parquet(
        cls, path: str, settings: ResolutionSettings
    ) -> "ProductResolver":
        started_at = perf_counter()
        resolver = cls(pl.read_parquet(path), settings)
        resolver.index_build_seconds = perf_counter() - started_at
        return resolver

    def resolve_many(
        self, references: Sequence[str | int | Mapping[str, Any] | ProductReference]
    ) -> list[ResolutionResult]:
        return [self.resolve(reference) for reference in references]

    def resolve(
        self, reference: str | int | Mapping[str, Any] | ProductReference) -> ResolutionResult:
        structured = self._coerce_reference(reference)
        query = self._query_label(reference, structured)
        if structured.product_id is not None:
            row = self.id_index.get(str(structured.product_id).strip())
            if row is None:
                return ResolutionResult(
                    query=query, status="not_found", reason="product_id does not exist"
                )
            candidate = self._candidate(row, 100.0, ["exact product_id"])
            return ResolutionResult(
                query=query,
                status="exact",
                selected_product_id=candidate.product_id,
                confidence=1.0,
                candidates=[candidate],
                reason="exact product_id match",
            )

        normalized_title = normalize_product_text(structured.title)
        if not normalized_title:
            return ResolutionResult(
                query=query,
                status="not_found",
                reason="a non-empty product_id or title is required",
            )
        normalized_brand = normalize_product_text(structured.brand)
        exact_rows = self.title_index.get(normalized_title, [])
        exact_rows = self._apply_metadata_filters(exact_rows, normalized_brand, structured.category)
        if len(exact_rows) == 1:
            candidate = self._candidate(exact_rows[0], 100.0, ["exact normalized title"])
            if candidate.canonicalization_status.startswith("identity_conflict"):
                return ResolutionResult(
                    query=query,
                    status="ambiguous",
                    confidence=1.0,
                    candidates=[candidate],
                    reason="the matching product_id has conflicting identity metadata",
                )
            return ResolutionResult(
                query=query,
                status="exact",
                selected_product_id=candidate.product_id,
                confidence=1.0,
                candidates=[candidate],
                reason="unique exact normalized title match",
            )
        if len(exact_rows) > 1:
            candidates = [self._candidate(row, 100.0, ["exact normalized title"]) for row in exact_rows[:10]]
            return ResolutionResult(
                query=query,
                status="ambiguous",
                confidence=1.0,
                candidates=candidates,
                reason="multiple canonical products share this normalized title",
            )

        query_models = model_tokens(normalized_title)
        pool = self._candidate_pool(normalized_title, normalized_brand, structured.category)
        if not pool:
            return ResolutionResult(query=query, status="not_found", reason="no lexical candidates")
        # A query's model/variant tokens are hard constraints. In particular,
        # Pro != Pro+, A55 != A35, and 128GB != 256GB.
        if query_models:
            pool = [
                row
                for row in pool
                if query_models.issubset(model_tokens(self.columns["normalized_title"][row]))
            ]
        if not pool:
            return ResolutionResult(
                query=query,
                status="not_found",
                reason="all lexical candidates conflict with model-significant tokens",
            )
        scored = self._score_candidates(normalized_title, normalized_brand, pool)
        if not scored or scored[0][1] < self.settings.fuzzy_score_threshold:
            candidates = [self._candidate(row, score, reasons) for row, score, reasons in scored[:5]]
            return ResolutionResult(
                query=query,
                status="not_found",
                confidence=(scored[0][1] / 100 if scored else 0.0),
                candidates=candidates,
                reason="no candidate reached the documented fuzzy threshold",
            )
        best_row, best_score, best_reasons = scored[0]
        plausible = [item for item in scored if best_score - item[1] <= self.settings.ambiguity_score_margin]
        candidates = [self._candidate(row, score, reasons) for row, score, reasons in plausible[:10]]
        if len(plausible) > 1 or candidates[0].canonicalization_status.startswith("identity_conflict"):
            return ResolutionResult(
                query=query,
                status="ambiguous",
                confidence=best_score / 100,
                candidates=candidates,
                reason="multiple plausible candidates or conflicted canonical identity",
            )
        return ResolutionResult(
            query=query,
            status="resolved",
            selected_product_id=str(self.columns["product_id"][best_row]),
            confidence=best_score / 100,
            candidates=candidates,
            reason="brand-aware fuzzy lexical match",
        )

    def _candidate_pool(
        self, title: str, brand: str | None, category: str | None
    ) -> list[int]:
        token_sets: list[set[int]] = []
        for token in model_tokens(title):
            if token in self.model_index:
                token_sets.append(self.model_index[token])
        if not token_sets:
            for token in product_tokens(title):
                indexed = self.token_index.get(token)
                if indexed:
                    token_sets.append(indexed)
        if not token_sets:
            return []
        token_sets.sort(key=len)
        pool = set(token_sets[0])
        for indexed in token_sets[1:3]:
            pool.intersection_update(indexed)
        if brand and brand in self.brand_index:
            pool.intersection_update(self.brand_index[brand])
        filtered = self._apply_metadata_filters(list(pool), brand, category)
        return filtered

    def _apply_metadata_filters(
        self, rows: Sequence[int], brand: str | None, category: str | None
    ) -> list[int]:
        result = list(rows)
        if brand:
            result = [row for row in result if self.columns["normalized_brand"][row] == brand]
        normalized_category = normalize_product_text(category)
        if normalized_category:
            result = [
                row
                for row in result
                if normalized_category
                in {
                    normalize_product_text(self.columns["Category1"][row]),
                    normalize_product_text(self.columns["Category2"][row]),
                    normalize_product_text(self.columns["sub_category"][row]),
                }
            ]
        return result

    def _score_candidates(
        self, query: str, brand: str | None, rows: Sequence[int]
    ) -> list[tuple[int, float, list[str]]]:
        choices = {row: self.columns["normalized_title"][row] for row in rows}
        matches = process.extract(
            query,
            choices,
            scorer=fuzz.ratio,
            limit=self.settings.max_fuzzy_candidates,
            score_cutoff=0,
        )
        scored: list[tuple[int, float, list[str]]] = []
        for title_choice, ratio, row in matches:
            token_score = fuzz.token_set_ratio(query, title_choice)
            score = 0.7 * ratio + 0.3 * token_score
            reasons = [f"title ratio={ratio:.1f}", f"token-set ratio={token_score:.1f}"]
            if brand and self.columns["normalized_brand"][row] == brand:
                score = min(100.0, score + 5.0)
                reasons.append("exact brand")
            if model_tokens(query):
                reasons.append("model tokens match")
            scored.append((row, round(score, 3), reasons))
        return sorted(scored, key=lambda item: (-item[1], str(self.columns["product_id"][item[0]])))

    def _candidate(self, row: int, score: float, reasons: list[str]) -> ResolutionCandidate:
        return ResolutionCandidate(
            product_id=str(self.columns["product_id"][row]),
            title_fa=self.columns["title_fa"][row],
            Brand=self.columns["Brand"][row],
            Category1=self.columns["Category1"][row],
            Category2=self.columns["Category2"][row],
            sub_category=self.columns["sub_category"][row],
            score=score,
            match_reasons=reasons,
            canonicalization_status=self.columns["canonicalization_status"][row],
            total_review_count=self.columns["total_review_count"][row],
            buyer_review_count=self.columns["buyer_review_count"][row],
            recommendation_known_count=self.columns["recommendation_known_count"][row],
        )

    @staticmethod
    def _coerce_reference(
        reference: str | int | Mapping[str, Any] | ProductReference
    ) -> ProductReference:
        if isinstance(reference, ProductReference):
            return reference
        if isinstance(reference, int):
            return ProductReference(product_id=reference)
        if isinstance(reference, Mapping):
            return ProductReference.model_validate(reference)
        if isinstance(reference, str) and reference.strip().isdigit():
            return ProductReference(product_id=reference.strip())
        return ProductReference(title=str(reference))

    @staticmethod
    def _query_label(reference: Any, structured: ProductReference) -> str:
        if isinstance(reference, str):
            return reference
        if structured.product_id is not None:
            return str(structured.product_id)
        return structured.title or ""
