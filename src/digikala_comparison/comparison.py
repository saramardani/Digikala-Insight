"""Deterministic, support-aware product comparison without LLM decisions.

This module deliberately keeps three layers separate: product snapshot facts,
full-population review statistics, and Top-K retrieved review evidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Sequence

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import ComparisonSettings, Settings
from .evidence import EvidenceSet, ProductionEvidenceRetriever


DecisionStatus = Literal["winner", "tie", "inconclusive", "informational"]
OverallStatus = Literal["neutral", "weighted_winner", "inconclusive"]
ProvenanceStatus = Literal["stable", "conflicted", "missing"]
SourceLayer = Literal["direct_product_snapshot", "full_product_statistics", "retrieved_evidence"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanonicalProductIdentity(StrictModel):
    product_id: str
    title_fa: str | None
    brand: str | None
    category1: str | None
    category2: str | None
    sub_category: str | None
    canonicalization_status: str
    source_row_count: int
    has_metadata_conflict: bool
    conflicting_fields: list[str] = Field(default_factory=list)


class DirectFieldValue(StrictModel):
    """One canonical product-snapshot field with its duplicate provenance."""

    field: str
    value: str | int | bool | None
    semantic_type: str
    provenance_status: ProvenanceStatus
    source_distinct_count: int
    source_layer: Literal["direct_product_snapshot"] = "direct_product_snapshot"
    note: str | None = None


class DirectProductFields(StrictModel):
    price: DirectFieldValue
    rate: DirectFieldValue
    rate_count: DirectFieldValue
    min_price_last_month: DirectFieldValue
    is_fake: DirectFieldValue
    brand: DirectFieldValue
    category1: DirectFieldValue
    category2: DirectFieldValue
    sub_category: DirectFieldValue


class FullDataProductStatistics(StrictModel):
    """Statistics calculated from the complete canonical review population."""

    product_id: str
    total_review_count: int
    buyer_review_count: int
    non_buyer_review_count: int
    unknown_buyer_review_count: int
    review_rate_valid_count: int
    average_review_rate: float | None
    median_review_rate: float | None
    recommended_count: int
    not_recommended_count: int
    no_idea_count: int
    recommendation_known_count: int
    opinionated_review_count: int
    recommended_percentage: float | None
    not_recommended_percentage: float | None
    no_idea_percentage: float | None
    opinionated_recommend_percentage: float | None
    source_layer: Literal["full_product_statistics"] = "full_product_statistics"

    @model_validator(mode="after")
    def _preserve_recommendation_denominator(self) -> "FullDataProductStatistics":
        known = self.recommended_count + self.not_recommended_count + self.no_idea_count
        if known != self.recommendation_known_count:
            raise ValueError("recommendation counts must sum to recommendation_known_count")
        if self.opinionated_review_count != self.recommended_count + self.not_recommended_count:
            raise ValueError("opinionated_review_count must preserve the two opinionated counts")
        return self


class ProductComparisonInput(StrictModel):
    identity: CanonicalProductIdentity
    direct_fields: DirectProductFields
    full_statistics: FullDataProductStatistics
    evidence_sets: list[EvidenceSet] = Field(default_factory=list)

    @model_validator(mode="after")
    def _preserve_product_ownership(self) -> "ProductComparisonInput":
        product_id = self.identity.product_id
        if self.full_statistics.product_id != product_id:
            raise ValueError("full statistics product_id must match canonical identity")
        for evidence in self.evidence_sets:
            if evidence.product_id != product_id:
                raise ValueError("evidence product_id must match canonical identity")
            if any(item.product_id != product_id for item in evidence.evidence_items):
                raise ValueError("evidence item product_id must match canonical identity")
        return self


class CriterionRequest(StrictModel):
    name: str
    evidence_query: str | None = None
    attach_evidence: bool = True

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("criterion name cannot be blank")
        return cleaned


class PreferencePolicy(StrictModel):
    """Explicit caller-provided weights; the engine never invents these."""

    weights: dict[str, float]

    @model_validator(mode="after")
    def _positive_weights(self) -> "PreferencePolicy":
        if not self.weights or any(not key.strip() or value <= 0 for key, value in self.weights.items()):
            raise ValueError("preference weights must be non-empty positive values")
        return self


class CriterionValue(StrictModel):
    product_id: str
    value: str | int | float | bool | None
    source_layer: SourceLayer
    availability: Literal["available", "missing", "conflicted", "unvalidated"]
    semantic_type: str


class CriterionSupport(StrictModel):
    product_id: str
    source_layer: SourceLayer
    numerator: int | None = None
    denominator: int | None = None
    percentage: float | None = None
    support_count: int | None = None
    scope: str


class DecisionExplanation(StrictModel):
    direction: Literal["higher_is_better", "lower_is_better", "none"]
    practical_threshold: float | None = None
    observed_difference: float | None = None
    difference_unit: str | None = None
    notes: list[str] = Field(default_factory=list)


class CriterionDecision(StrictModel):
    criterion: str
    status: DecisionStatus
    winner_product_ids: list[str] = Field(default_factory=list)
    ranking_product_ids: list[str] | None = None
    reason_code: str
    values: list[CriterionValue]
    support: list[CriterionSupport]
    explanation: DecisionExplanation


class RetrievedEvidenceCounts(StrictModel):
    product_id: str
    retrieved_count: int
    eligible_product_review_count: int
    positive_count: int
    negative_count: int
    neutral_or_unknown_count: int
    scope: Literal["within_retrieved_evidence"] = "within_retrieved_evidence"


class CriterionEvidenceAttachment(StrictModel):
    criterion: str
    evidence_sets: list[EvidenceSet] = Field(default_factory=list)
    evidence_counts: list[RetrievedEvidenceCounts] = Field(default_factory=list)


class OverallScore(StrictModel):
    product_id: str
    score: float


class OverallDecision(StrictModel):
    status: OverallStatus
    winner_product_ids: list[str] = Field(default_factory=list)
    reason_code: str
    preference_policy: PreferencePolicy | None = None
    scores: list[OverallScore] = Field(default_factory=list)


class ComparisonWarning(StrictModel):
    code: str
    message: str
    product_id: str | None = None
    criterion: str | None = None


class ComparisonResult(StrictModel):
    products: list[CanonicalProductIdentity]
    criteria: list[CriterionRequest]
    direct_facts: list[DirectProductFields]
    aggregate_statistics: list[FullDataProductStatistics]
    retrieved_evidence: list[CriterionEvidenceAttachment]
    criterion_decisions: list[CriterionDecision]
    overall: OverallDecision
    warnings: list[ComparisonWarning] = Field(default_factory=list)


_ALIASES = {
    "price": "price",
    "min_price_last_month": "min_price_last_month",
    "rate": "rate",
    "rate_cnt": "rate_count",
    "rate_count": "rate_count",
    "is_fake": "is_fake",
    "brand": "brand",
    "category": "category",
    "category1": "category1",
    "category2": "category2",
    "sub_category": "sub_category",
    "recommendation": "recommended_percentage",
    "recommended_percentage": "recommended_percentage",
    "not_recommendation": "not_recommended_percentage",
    "not_recommended_percentage": "not_recommended_percentage",
    "opinionated_recommendation": "opinionated_recommend_percentage",
    "opinionated_recommend_percentage": "opinionated_recommend_percentage",
    "buyer_review_count": "buyer_review_count",
    "average_review_rate": "average_review_rate",
    "median_review_rate": "median_review_rate",
}
_DIRECT_CRITERIA = {"price", "min_price_last_month", "rate", "rate_count", "is_fake", "brand", "category", "category1", "category2", "sub_category"}
_NON_EVIDENCE_CRITERIA = _DIRECT_CRITERIA | {"buyer_review_count", "average_review_rate", "median_review_rate"}
_PERCENTAGE_CRITERIA = {
    "recommended_percentage": ("recommended_count", "recommendation_known_count", "recommended_percentage", True),
    "not_recommended_percentage": ("not_recommended_count", "recommendation_known_count", "not_recommended_percentage", False),
    "opinionated_recommend_percentage": ("recommended_count", "opinionated_review_count", "opinionated_recommend_percentage", True),
}


def canonical_criterion(name: str) -> str:
    return _ALIASES.get(name.strip().lower(), name.strip().lower())


def criterion_uses_retrieved_evidence(name: str) -> bool:
    """Only experience-oriented criteria receive review attachments."""
    return canonical_criterion(name) not in _NON_EVIDENCE_CRITERIA


class ProductComparisonEngine:
    """Pure-Python decision layer; it never calls a model or interprets Top-K as population data."""

    def __init__(self, settings: ComparisonSettings):
        self.settings = settings

    def compare(
        self,
        products: Sequence[ProductComparisonInput],
        criteria: Sequence[CriterionRequest],
        preference_policy: PreferencePolicy | None = None,
    ) -> ComparisonResult:
        products = list(products)
        criteria = list(criteria)
        if len(products) < 2:
            raise ValueError("at least two products are required for comparison")
        ids = [product.identity.product_id for product in products]
        if len(ids) != len(set(ids)):
            raise ValueError("comparison product IDs must be unique")
        if not criteria:
            raise ValueError("at least one criterion is required")

        decisions = [self._compare_criterion(products, request) for request in criteria]
        evidence = [self._attach_evidence(products, request) for request in criteria]
        evidence = [attachment for attachment in evidence if attachment.evidence_sets]
        warnings = self._warnings(products, evidence)
        overall = self._overall(ids, decisions, preference_policy)
        return ComparisonResult(
            products=[product.identity for product in products],
            criteria=criteria,
            direct_facts=[product.direct_fields for product in products],
            aggregate_statistics=[product.full_statistics for product in products],
            retrieved_evidence=evidence,
            criterion_decisions=decisions,
            overall=overall,
            warnings=warnings,
        )

    def _compare_criterion(
        self, products: list[ProductComparisonInput], request: CriterionRequest
    ) -> CriterionDecision:
        criterion = canonical_criterion(request.name)
        if criterion == "price":
            return self._direct_numeric(products, criterion, "price", False, self.settings.practical_price_relative_difference, "relative_raw_price_difference")
        if criterion == "min_price_last_month":
            return self._direct_numeric(products, criterion, "min_price_last_month", False, self.settings.practical_price_relative_difference, "relative_raw_price_difference")
        if criterion == "rate":
            return self._product_rate(products)
        if criterion == "is_fake":
            return self._is_fake(products)
        if criterion in {"rate_count", "brand", "category", "category1", "category2", "sub_category", "buyer_review_count"}:
            return self._informational(products, criterion)
        if criterion in _PERCENTAGE_CRITERIA:
            return self._percentage(products, criterion)
        if criterion in {"average_review_rate", "median_review_rate"}:
            return self._unvalidated_review_rate(products, criterion)
        return self._evidence_only(products, criterion)

    def _direct_numeric(
        self,
        products: list[ProductComparisonInput],
        criterion: str,
        field: str,
        higher_is_better: bool,
        threshold: float,
        unit: str,
    ) -> CriterionDecision:
        values: list[CriterionValue] = []
        supports: list[CriterionSupport] = []
        numbers: list[tuple[str, float]] = []
        blocked_reason: str | None = None
        for product in products:
            direct = getattr(product.direct_fields, field)
            availability = self._direct_availability(direct, positive_required=True)
            values.append(CriterionValue(product_id=product.identity.product_id, value=direct.value, source_layer="direct_product_snapshot", availability=availability, semantic_type=direct.semantic_type))
            supports.append(CriterionSupport(product_id=product.identity.product_id, source_layer="direct_product_snapshot", scope="one canonical product snapshot"))
            if availability == "conflicted":
                blocked_reason = "METADATA_CONFLICT"
            elif availability != "available" and blocked_reason is None:
                blocked_reason = "MISSING_REQUIRED_DATA"
            elif availability == "available":
                numbers.append((product.identity.product_id, float(direct.value)))
        if blocked_reason:
            return self._inconclusive(criterion, blocked_reason, values, supports, "lower_is_better" if not higher_is_better else "higher_is_better", threshold, unit)
        return self._decide_numbers(criterion, numbers, values, supports, higher_is_better, threshold, unit)

    def _product_rate(self, products: list[ProductComparisonInput]) -> CriterionDecision:
        values: list[CriterionValue] = []
        supports: list[CriterionSupport] = []
        numbers: list[tuple[str, float]] = []
        reason: str | None = None
        for product in products:
            rate = product.direct_fields.rate
            rate_count = product.direct_fields.rate_count
            availability = self._direct_availability(rate, positive_required=False)
            if self._direct_availability(rate_count, positive_required=False) == "conflicted" or availability == "conflicted":
                reason = "METADATA_CONFLICT"
                availability = "conflicted"
            count = int(rate_count.value) if isinstance(rate_count.value, int) and not isinstance(rate_count.value, bool) else 0
            if availability == "available" and count < self.settings.minimum_product_rate_count and reason is None:
                reason = "INSUFFICIENT_SUPPORT"
            if availability != "available" and reason is None:
                reason = "MISSING_REQUIRED_DATA"
            values.append(CriterionValue(product_id=product.identity.product_id, value=rate.value, source_layer="direct_product_snapshot", availability=availability, semantic_type="validated_product_rate_0_to_100"))
            supports.append(CriterionSupport(product_id=product.identity.product_id, source_layer="direct_product_snapshot", support_count=count, scope="Rate_cnt on canonical product snapshot"))
            if availability == "available":
                numbers.append((product.identity.product_id, float(rate.value)))
        if reason:
            return self._inconclusive("rate", reason, values, supports, "higher_is_better", self.settings.practical_product_rate_point_difference, "product_rate_points")
        return self._decide_numbers("rate", numbers, values, supports, True, self.settings.practical_product_rate_point_difference, "product_rate_points")

    def _is_fake(self, products: list[ProductComparisonInput]) -> CriterionDecision:
        values: list[CriterionValue] = []
        supports: list[CriterionSupport] = []
        numbers: list[tuple[str, float]] = []
        reason: str | None = None
        for product in products:
            direct = product.direct_fields.is_fake
            availability = self._direct_availability(direct, positive_required=False)
            if not isinstance(direct.value, bool) and availability == "available":
                availability = "missing"
            values.append(CriterionValue(product_id=product.identity.product_id, value=direct.value, source_layer="direct_product_snapshot", availability=availability, semantic_type=direct.semantic_type))
            supports.append(CriterionSupport(product_id=product.identity.product_id, source_layer="direct_product_snapshot", scope="canonical Is_Fake snapshot flag"))
            if availability == "conflicted":
                reason = "METADATA_CONFLICT"
            elif availability != "available" and reason is None:
                reason = "MISSING_REQUIRED_DATA"
            else:
                numbers.append((product.identity.product_id, 1.0 if direct.value else 0.0))
        if reason:
            return self._inconclusive("is_fake", reason, values, supports, "lower_is_better", 1.0, "boolean_flag")
        return self._decide_numbers("is_fake", numbers, values, supports, False, 1.0, "boolean_flag")

    def _percentage(self, products: list[ProductComparisonInput], criterion: str) -> CriterionDecision:
        numerator_field, denominator_field, _percentage_field, higher_is_better = _PERCENTAGE_CRITERIA[criterion]
        values: list[CriterionValue] = []
        supports: list[CriterionSupport] = []
        numbers: list[tuple[str, float]] = []
        reason: str | None = None
        for product in products:
            stats = product.full_statistics
            numerator = int(getattr(stats, numerator_field))
            denominator = int(getattr(stats, denominator_field))
            # Percentages used for decisions are recalculated from the retained
            # full-population counts, never read from retrieval evidence or
            # delegated to a language model.
            percentage = None if denominator == 0 else numerator / denominator
            availability: Literal["available", "missing", "conflicted", "unvalidated"]
            if denominator == 0:
                availability, reason = "missing", "ZERO_DENOMINATOR"
            elif percentage is None:
                availability, reason = "missing", reason or "MISSING_REQUIRED_DATA"
            elif denominator < self.settings.minimum_percentage_denominator:
                availability, reason = "available", reason or "INSUFFICIENT_SUPPORT"
            else:
                availability = "available"
                numbers.append((product.identity.product_id, float(percentage)))
            values.append(CriterionValue(product_id=product.identity.product_id, value=percentage, source_layer="full_product_statistics", availability=availability, semantic_type="full_population_percentage"))
            supports.append(CriterionSupport(product_id=product.identity.product_id, source_layer="full_product_statistics", numerator=numerator, denominator=denominator, percentage=percentage, support_count=denominator, scope="complete canonical review population"))
        if reason:
            return self._inconclusive(criterion, reason, values, supports, "higher_is_better" if higher_is_better else "lower_is_better", self.settings.practical_percentage_point_difference, "percentage_points")
        return self._decide_numbers(criterion, numbers, values, supports, higher_is_better, self.settings.practical_percentage_point_difference, "percentage_points")

    def _informational(self, products: list[ProductComparisonInput], criterion: str) -> CriterionDecision:
        values: list[CriterionValue] = []
        supports: list[CriterionSupport] = []
        direct_field = "category1" if criterion == "category" else criterion
        for product in products:
            if criterion == "buyer_review_count":
                stats = product.full_statistics
                values.append(CriterionValue(product_id=product.identity.product_id, value=stats.buyer_review_count, source_layer="full_product_statistics", availability="available", semantic_type="support_count_not_quality"))
                supports.append(CriterionSupport(product_id=product.identity.product_id, source_layer="full_product_statistics", support_count=stats.total_review_count, scope="complete canonical review population"))
                continue
            direct = getattr(product.direct_fields, direct_field)
            availability = self._direct_availability(direct, positive_required=False)
            values.append(CriterionValue(product_id=product.identity.product_id, value=direct.value, source_layer="direct_product_snapshot", availability=availability, semantic_type=direct.semantic_type))
            supports.append(CriterionSupport(product_id=product.identity.product_id, source_layer="direct_product_snapshot", scope="one canonical product snapshot"))
        code = "INFORMATIONAL_SUPPORT_SIGNAL" if criterion in {"rate_count", "buyer_review_count"} else "INFORMATIONAL_DIRECT_FACT"
        return CriterionDecision(criterion=criterion, status="informational", reason_code=code, values=values, support=supports, explanation=DecisionExplanation(direction="none", notes=["This field is surfaced as a direct fact or support signal; it does not imply product quality."]))

    def _unvalidated_review_rate(self, products: list[ProductComparisonInput], criterion: str) -> CriterionDecision:
        values = [
            CriterionValue(product_id=product.identity.product_id, value=getattr(product.full_statistics, criterion), source_layer="full_product_statistics", availability="unvalidated", semantic_type="review_rate_scale_not_validated")
            for product in products
        ]
        supports = [
            CriterionSupport(product_id=product.identity.product_id, source_layer="full_product_statistics", support_count=product.full_statistics.review_rate_valid_count, scope="complete canonical review population")
            for product in products
        ]
        return self._inconclusive(criterion, "UNVALIDATED_FIELD_SEMANTICS", values, supports, "none", None, None, "The observed review-rate distribution includes outliers, so no common numeric scale is assumed.")

    def _evidence_only(self, products: list[ProductComparisonInput], criterion: str) -> CriterionDecision:
        evidence = self._evidence_sets(products, criterion)
        values = [
            CriterionValue(product_id=product.identity.product_id, value=sum(item.retrieved_count for item in evidence if item.product_id == product.identity.product_id), source_layer="retrieved_evidence", availability="available" if any(item.product_id == product.identity.product_id and item.retrieved_count > 0 for item in evidence) else "missing", semantic_type="retrieved_evidence_count_only")
            for product in products
        ]
        supports = [
            CriterionSupport(product_id=product.identity.product_id, source_layer="retrieved_evidence", support_count=sum(item.retrieved_count for item in evidence if item.product_id == product.identity.product_id), scope="within retrieved evidence")
            for product in products
        ]
        counts = [support.support_count or 0 for support in supports]
        if not evidence or any(count == 0 for count in counts):
            return self._inconclusive(criterion, "NO_RETRIEVED_EVIDENCE", values, supports, "none", None, None)
        if any(count < self.settings.minimum_retrieved_evidence_items for count in counts):
            return self._inconclusive(criterion, "EVIDENCE_TOO_SPARSE", values, supports, "none", float(self.settings.minimum_retrieved_evidence_items), "retrieved_items")
        return CriterionDecision(criterion=criterion, status="informational", reason_code="RETRIEVED_EVIDENCE_ONLY", values=values, support=supports, explanation=DecisionExplanation(direction="none", notes=["Retrieved evidence is attached for inspection only; no population percentage or winner is inferred from Top-K items."]))

    def _decide_numbers(
        self,
        criterion: str,
        numbers: list[tuple[str, float]],
        values: list[CriterionValue],
        supports: list[CriterionSupport],
        higher_is_better: bool,
        threshold: float,
        unit: str,
    ) -> CriterionDecision:
        ordered = sorted(numbers, key=lambda pair: (pair[1], pair[0]), reverse=higher_is_better)
        best_value = ordered[0][1]
        winners = [product_id for product_id, value in ordered if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)]
        direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better" if higher_is_better else "lower_is_better"
        if len(winners) > 1:
            return CriterionDecision(criterion=criterion, status="tie", winner_product_ids=sorted(winners), reason_code="EXACT_TIE", values=values, support=supports, explanation=DecisionExplanation(direction=direction, practical_threshold=threshold, observed_difference=0.0, difference_unit=unit))
        runner_up = ordered[1][1]
        difference = self._difference(best_value, runner_up, unit)
        if difference < threshold:
            return self._inconclusive(criterion, "PRACTICAL_DIFFERENCE_NOT_MET", values, supports, direction, threshold, unit, observed_difference=difference)
        ranking = self._full_ranking_if_justified(ordered, threshold, unit)
        return CriterionDecision(criterion=criterion, status="winner", winner_product_ids=winners, ranking_product_ids=ranking, reason_code="WINNER_BY_PRACTICAL_DIFFERENCE", values=values, support=supports, explanation=DecisionExplanation(direction=direction, practical_threshold=threshold, observed_difference=difference, difference_unit=unit))

    @staticmethod
    def _difference(first: float, second: float, unit: str) -> float:
        if unit == "relative_raw_price_difference":
            return abs(first - second) / min(abs(first), abs(second))
        return abs(first - second)

    def _full_ranking_if_justified(self, ordered: list[tuple[str, float]], threshold: float, unit: str) -> list[str] | None:
        if all(self._difference(first[1], second[1], unit) >= threshold for first, second in zip(ordered, ordered[1:])):
            return [product_id for product_id, _ in ordered]
        return None

    def _inconclusive(
        self,
        criterion: str,
        reason_code: str,
        values: list[CriterionValue],
        supports: list[CriterionSupport],
        direction: Literal["higher_is_better", "lower_is_better", "none"],
        threshold: float | None,
        unit: str | None,
        note: str | None = None,
        observed_difference: float | None = None,
    ) -> CriterionDecision:
        return CriterionDecision(criterion=criterion, status="inconclusive", reason_code=reason_code, values=values, support=supports, explanation=DecisionExplanation(direction=direction, practical_threshold=threshold, observed_difference=observed_difference, difference_unit=unit, notes=[] if note is None else [note]))

    def _direct_availability(self, value: DirectFieldValue, *, positive_required: bool) -> Literal["available", "missing", "conflicted"]:
        if value.provenance_status == "conflicted" and self.settings.require_stable_field_metadata:
            return "conflicted"
        if value.value is None:
            return "missing"
        if positive_required and (not isinstance(value.value, (int, float)) or isinstance(value.value, bool) or value.value <= 0):
            return "missing"
        return "available"

    def _evidence_sets(self, products: list[ProductComparisonInput], criterion: str) -> list[EvidenceSet]:
        return [
            evidence
            for product in products
            for evidence in product.evidence_sets
            if evidence.product_id == product.identity.product_id and canonical_criterion(evidence.criterion) == criterion
        ]

    def _attach_evidence(self, products: list[ProductComparisonInput], request: CriterionRequest) -> CriterionEvidenceAttachment:
        criterion = canonical_criterion(request.name)
        if not request.attach_evidence or not criterion_uses_retrieved_evidence(request.name):
            return CriterionEvidenceAttachment(criterion=criterion)
        evidence = self._evidence_sets(products, criterion)
        counts = []
        for item in evidence:
            statuses = [review.recommendation_status for review in item.evidence_items]
            positive = sum(status == "recommended" for status in statuses)
            negative = sum(status == "not_recommended" for status in statuses)
            neutral = len(statuses) - positive - negative
            counts.append(RetrievedEvidenceCounts(product_id=item.product_id, retrieved_count=item.retrieved_count, eligible_product_review_count=item.eligible_product_review_count, positive_count=positive, negative_count=negative, neutral_or_unknown_count=neutral))
        return CriterionEvidenceAttachment(criterion=criterion, evidence_sets=evidence, evidence_counts=counts)

    def _warnings(self, products: list[ProductComparisonInput], evidence: list[CriterionEvidenceAttachment]) -> list[ComparisonWarning]:
        warnings: list[ComparisonWarning] = []
        for product in products:
            if product.identity.has_metadata_conflict:
                warnings.append(ComparisonWarning(code="METADATA_CONFLICT_PRESENT", product_id=product.identity.product_id, message=f"Canonical snapshot retains conflicts in: {', '.join(product.identity.conflicting_fields)}."))
        for attachment in evidence:
            for counts in attachment.evidence_counts:
                if counts.positive_count and counts.negative_count:
                    warnings.append(ComparisonWarning(code="RETRIEVED_EVIDENCE_MIXED", product_id=counts.product_id, criterion=attachment.criterion, message="Positive and negative recommendation statuses coexist within retrieved evidence."))
        return warnings

    def _overall(self, ids: list[str], decisions: list[CriterionDecision], policy: PreferencePolicy | None) -> OverallDecision:
        if policy is None:
            return OverallDecision(status="neutral", reason_code="NO_PREFERENCE_POLICY")
        scores = {product_id: 0.0 for product_id in ids}
        applied = 0
        normalized_weights: dict[str, float] = {}
        for raw_criterion, weight in policy.weights.items():
            criterion = canonical_criterion(raw_criterion)
            if criterion in normalized_weights:
                raise ValueError(f"preference policy duplicates criterion after normalization: {criterion}")
            normalized_weights[criterion] = weight
        for decision in decisions:
            weight = normalized_weights.get(decision.criterion)
            if weight is None or decision.status not in {"winner", "tie"} or not decision.winner_product_ids:
                continue
            applied += 1
            share = weight / len(decision.winner_product_ids)
            for winner in decision.winner_product_ids:
                scores[winner] += share
        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        score_models = [OverallScore(product_id=product_id, score=score) for product_id, score in ordered]
        if applied == 0:
            return OverallDecision(status="inconclusive", reason_code="NO_COMPARABLE_WEIGHTED_CRITERIA", preference_policy=policy, scores=score_models)
        best = ordered[0][1]
        winners = [product_id for product_id, score in ordered if math.isclose(score, best, rel_tol=0.0, abs_tol=1e-12)]
        if len(winners) > 1:
            return OverallDecision(status="inconclusive", winner_product_ids=winners, reason_code="WEIGHTED_TIE", preference_policy=policy, scores=score_models)
        return OverallDecision(status="weighted_winner", winner_product_ids=winners, reason_code="WEIGHTED_PREFERENCE_POLICY", preference_policy=policy, scores=score_models)


class ComparisonDataStore:
    """Loads only requested products from the Phase 2 and Phase 3 Parquet artifacts."""

    _DIRECT_FIELDS = (
        ("price", "Price", "numeric_raw_snapshot_price"),
        ("rate", "Rate", "validated_product_rate_0_to_100"),
        ("rate_count", "Rate_cnt", "rating_support_count"),
        ("min_price_last_month", "min_price_last_month", "numeric_raw_historical_price"),
        ("is_fake", "Is_Fake", "boolean_snapshot_flag"),
        ("brand", "Brand", "categorical_product_metadata"),
        ("category1", "Category1", "categorical_product_metadata"),
        ("category2", "Category2", "categorical_product_metadata"),
        ("sub_category", "sub_category", "categorical_product_metadata"),
    )

    def __init__(self, canonical_products_path: Path, product_statistics_path: Path):
        if not canonical_products_path.is_file():
            raise FileNotFoundError("canonical_products.parquet is required for comparison")
        if not product_statistics_path.is_file():
            raise FileNotFoundError("product_statistics.parquet is required for comparison")
        self.canonical_products_path = canonical_products_path
        self.product_statistics_path = product_statistics_path

    @classmethod
    def from_settings(cls, settings: Settings) -> "ComparisonDataStore":
        if settings.paths.canonical_products is None or settings.paths.product_statistics is None:
            raise ValueError("canonical_products and product_statistics paths must be configured")
        return cls(settings.paths.canonical_products, settings.paths.product_statistics)

    def load_products(self, product_ids: Sequence[str | int]) -> list[ProductComparisonInput]:
        ids = [str(product_id) for product_id in product_ids]
        if len(ids) < 2:
            raise ValueError("at least two product IDs are required")
        if len(ids) != len(set(ids)):
            raise ValueError("comparison product IDs must be unique")
        canonical = pl.scan_parquet(self.canonical_products_path).filter(pl.col("product_id").cast(pl.String).is_in(ids)).collect()
        found = set(canonical.get_column("product_id").cast(pl.String).to_list())
        missing = sorted(set(ids) - found)
        if missing:
            raise ValueError(f"product IDs not found in canonical products: {missing}")
        stats = pl.scan_parquet(self.product_statistics_path).filter(pl.col("product_id").cast(pl.String).is_in(ids)).collect()
        stats_by_id = {str(row["product_id"]): row for row in stats.iter_rows(named=True)}
        if missing_stats := sorted(set(ids) - set(stats_by_id)):
            raise ValueError(f"product IDs missing full statistics: {missing_stats}")
        canonical_by_id = {str(row["product_id"]): row for row in canonical.iter_rows(named=True)}
        return [self._input(canonical_by_id[product_id], stats_by_id[product_id]) for product_id in ids]

    def _input(self, canonical: dict[str, object], stats: dict[str, object]) -> ProductComparisonInput:
        product_id = str(canonical["product_id"])
        conflicts = [str(field) for field in (canonical.get("conflicting_fields") or [])]
        identity = CanonicalProductIdentity(product_id=product_id, title_fa=_string(canonical.get("title_fa")), brand=_string(canonical.get("Brand")), category1=_string(canonical.get("Category1")), category2=_string(canonical.get("Category2")), sub_category=_string(canonical.get("sub_category")), canonicalization_status=str(canonical["canonicalization_status"]), source_row_count=int(canonical["source_row_count"]), has_metadata_conflict=bool(canonical["has_metadata_conflict"]), conflicting_fields=conflicts)
        direct: dict[str, DirectFieldValue] = {}
        for name, source_field, semantic_type in self._DIRECT_FIELDS:
            value = _scalar(canonical.get(source_field))
            distinct = int(canonical.get(f"{source_field}_distinct_count") or 0)
            status: ProvenanceStatus = "missing" if value is None else "conflicted" if source_field in conflicts else "stable"
            direct[name] = DirectFieldValue(field=source_field, value=value, semantic_type=semantic_type, provenance_status=status, source_distinct_count=distinct)
        full_statistics = FullDataProductStatistics(
            product_id=product_id,
            total_review_count=int(stats["total_review_count"]),
            buyer_review_count=int(stats["buyer_review_count"]),
            non_buyer_review_count=int(stats["non_buyer_review_count"]),
            unknown_buyer_review_count=int(stats["unknown_buyer_review_count"]),
            review_rate_valid_count=int(stats["review_rate_valid_count"]),
            average_review_rate=_float_or_none(stats.get("average_review_rate")),
            median_review_rate=_float_or_none(stats.get("median_review_rate")),
            recommended_count=int(stats["recommended_count"]),
            not_recommended_count=int(stats["not_recommended_count"]),
            no_idea_count=int(stats["no_idea_count"]),
            recommendation_known_count=int(stats["recommendation_known_count"]),
            opinionated_review_count=int(stats["opinionated_review_count"]),
            recommended_percentage=_float_or_none(stats.get("recommended_percentage")),
            not_recommended_percentage=_float_or_none(stats.get("not_recommended_percentage")),
            no_idea_percentage=_float_or_none(stats.get("no_idea_percentage")),
            opinionated_recommend_percentage=_float_or_none(stats.get("opinionated_recommend_percentage")),
        )
        return ProductComparisonInput(identity=identity, direct_fields=DirectProductFields(**direct), full_statistics=full_statistics)


class ProductComparisonService:
    """Wires the frozen evidence retriever into the deterministic engine."""

    def __init__(self, store: ComparisonDataStore, engine: ProductComparisonEngine, evidence_retriever: ProductionEvidenceRetriever | None = None, settings: Settings | None = None):
        self.store = store
        self.engine = engine
        self.evidence_retriever = evidence_retriever
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProductComparisonService":
        return cls(ComparisonDataStore.from_settings(settings), ProductComparisonEngine(settings.comparison), settings=settings)

    def compare_product_ids(self, product_ids: Sequence[str | int], criteria: Sequence[CriterionRequest], *, evidence_top_k: int = 10, preference_policy: PreferencePolicy | None = None) -> ComparisonResult:
        products = self.store.load_products(product_ids)
        by_id: dict[str, list[EvidenceSet]] = {product.identity.product_id: [] for product in products}
        for request in criteria:
            if not request.attach_evidence or not criterion_uses_retrieved_evidence(request.name):
                continue
            if self.evidence_retriever is None:
                if self.settings is None:
                    raise RuntimeError("an evidence retriever is required for review-based criteria")
                self.evidence_retriever = ProductionEvidenceRetriever.from_settings(self.settings)
            query = request.evidence_query or request.name
            for product in products:
                by_id[product.identity.product_id].append(self.evidence_retriever.retrieve_evidence(product.identity.product_id, request.name, query, evidence_top_k))
        products_with_evidence = [product.model_copy(update={"evidence_sets": by_id[product.identity.product_id]}) for product in products]
        return self.engine.compare(products_with_evidence, criteria, preference_policy)


def _scalar(value: object) -> str | int | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else int(value) if value.is_integer() else str(value)
    return str(value)


def _string(value: object) -> str | None:
    return None if value is None else str(value)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
