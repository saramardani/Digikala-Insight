"""Auditable deterministic grounding validation for generated comparisons.

This module validates structured claims against the Phase 9 generation context
and, when configured, independently rechecks canonical products, full-product
statistics, and review identifiers from the persisted Parquet artifacts.
Lexical overlap is deliberately only a conservative review-claim signal; it is
not presented as semantic entailment.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal, Protocol
import unicodedata

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from .config import GroundingSettings, Settings
from .generation import (
    AggregateFinding,
    ClaimLayer,
    GeneratedComparisonAnswer,
    GenerationContext,
    Recommendation,
    ReviewFinding,
    Scalar,
)


ClaimStatus = Literal["grounded", "unsupported", "contradicted", "inconclusive"]
CitationStatus = Literal["valid", "invalid"]
ValidationOverallStatus = Literal["accepted", "rejected"]
ActionTaken = Literal[
    "accepted",
    "rejected",
    "removed_unsupported",
    "rewrite_regenerated",
]
UnsupportedClaimAction = Literal["reject", "remove_unsupported", "rewrite_regenerate"]


class StrictGroundingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimResult(StrictGroundingModel):
    claim_id: str
    source_layer: ClaimLayer
    status: ClaimStatus
    reason_code: str
    authoritative_source_reference: str | None = None
    authoritative_value: Scalar | None = None
    detail: str | None = None


class CitationResult(StrictGroundingModel):
    claim_id: str
    review_id: str
    claimed_product_id: str
    status: CitationStatus
    reason_code: str
    authoritative_source_reference: str | None = None
    detail: str | None = None


class GroundingMetrics(StrictGroundingModel):
    factual_claim_count: int
    grounded_claim_count: int
    unsupported_claim_count: int
    contradiction_count: int
    grounded_claim_ratio: float | None
    unsupported_claim_ratio: float | None
    contradiction_rate: float | None
    citation_count: int
    valid_citation_count: int
    citation_correctness: float | None
    support_requiring_claim_count: int
    evidence_covered_claim_count: int
    evidence_coverage: float | None
    inconclusive_case_count: int
    correct_inconclusive_count: int
    inconclusive_correctness: float | None


class GroundingValidationResult(StrictGroundingModel):
    validator_version: str
    overall_status: ValidationOverallStatus
    claim_results: list[ClaimResult]
    citation_results: list[CitationResult]
    unsupported_claim_count: int
    grounded_claim_count: int
    warnings: list[str] = Field(default_factory=list)
    action_taken: ActionTaken
    metrics: GroundingMetrics

    @property
    def valid(self) -> bool:
        return self.overall_status == "accepted"

    @property
    def unsupported_claims(self) -> list[ClaimResult]:
        return [
            result
            for result in self.claim_results
            if result.status in {"unsupported", "contradicted"}
        ]


class GroundingPolicyOutcome(StrictGroundingModel):
    original_answer: GeneratedComparisonAnswer
    initial_validation: GroundingValidationResult
    final_answer: GeneratedComparisonAnswer | None = None
    final_validation: GroundingValidationResult | None = None
    action_taken: ActionTaken


class GroundingAuditRecord(StrictGroundingModel):
    validator_version: str
    context_fingerprint: str
    original_answer: GeneratedComparisonAnswer
    initial_validation: GroundingValidationResult
    final_answer: GeneratedComparisonAnswer | None = None
    final_validation: GroundingValidationResult | None = None
    action_taken: ActionTaken


METRIC_DEFINITIONS: dict[str, str] = {
    "grounded_claim_ratio": "grounded factual claims / all factual claims; one structured claim is one unit.",
    "unsupported_claim_ratio": "unsupported or contradicted factual claims / all factual claims.",
    "citation_correctness": "valid citations / all supplied citations; valid means existing, product-owned, in context, non-duplicate, and excerpt-matching.",
    "evidence_coverage": "factual claims with at least one valid authoritative reference / all factual claims.",
    "contradiction_rate": "contradicted factual claims / all factual claims.",
    "inconclusive_correctness": "inconclusive Phase 9 criterion/overall authorizations respected / all such authorizations; null when no abstention case exists.",
}


_CANONICAL_FIELD_COLUMNS = {
    "price": "Price",
    "rate": "Rate",
    "rate_count": "Rate_cnt",
    "min_price_last_month": "min_price_last_month",
    "is_fake": "Is_Fake",
    "brand": "Brand",
    "category1": "Category1",
    "category2": "Category2",
    "sub_category": "sub_category",
}
_PERCENTAGE_DENOMINATORS = {
    "recommended_percentage": ("recommended_count", "recommendation_known_count"),
    "not_recommended_percentage": ("not_recommended_count", "recommendation_known_count"),
    "no_idea_percentage": ("no_idea_count", "recommendation_known_count"),
    "opinionated_recommend_percentage": ("recommended_count", "opinionated_review_count"),
}
_STOP_TOKENS = frozenset(
    {
        "از", "به", "در", "با", "برای", "این", "آن", "را", "و", "یا", "که", "یک", "است",
        "بود", "شد", "می", "محصول", "کاربر", "دیدگاه", "گفته", "دارد", "خوب", "بد",
    }
)


class AuthoritativeLookup(Protocol):
    def direct_value(self, product_id: str, field: str) -> tuple[Scalar | None, str | None]: ...

    def statistic_value(self, product_id: str, metric: str) -> tuple[Scalar | None, str | None]: ...

    def review(self, review_id: str) -> tuple[dict[str, Scalar] | None, str | None]: ...


class ParquetAuthoritativeLookup:
    """Lazy Parquet rechecks; only cited products/reviews are collected."""

    def __init__(self, canonical_products: Path, statistics: Path, reviews: Path):
        self.canonical_products = canonical_products
        self.statistics = statistics
        self.reviews = reviews
        self._direct_cache: dict[tuple[str, str], tuple[Scalar | None, str | None]] = {}
        self._statistics_cache: dict[tuple[str, str], tuple[Scalar | None, str | None]] = {}
        self._review_cache: dict[str, tuple[dict[str, Scalar] | None, str | None]] = {}

    def direct_value(self, product_id: str, field: str) -> tuple[Scalar | None, str | None]:
        key = (product_id, field)
        if key in self._direct_cache:
            return self._direct_cache[key]
        column = _CANONICAL_FIELD_COLUMNS.get(field)
        if column is None:
            value = (None, None)
        else:
            frame = (
                pl.scan_parquet(self.canonical_products)
                .filter(pl.col("product_id").cast(pl.String) == product_id)
                .select(column)
                .collect()
            )
            raw = frame.item(0, 0) if frame.height else None
            value = (raw, f"{self.canonical_products.name}:{product_id}:{column}")
        self._direct_cache[key] = value
        return value

    def statistic_value(self, product_id: str, metric: str) -> tuple[Scalar | None, str | None]:
        key = (product_id, metric)
        if key in self._statistics_cache:
            return self._statistics_cache[key]
        schema = pl.read_parquet_schema(self.statistics)
        if metric not in schema:
            value = (None, None)
        else:
            frame = (
                pl.scan_parquet(self.statistics)
                .filter(pl.col("product_id").cast(pl.String) == product_id)
                .select(metric)
                .collect()
            )
            raw = frame.item(0, 0) if frame.height else None
            value = (raw, f"{self.statistics.name}:{product_id}:{metric}")
        self._statistics_cache[key] = value
        return value

    def review(self, review_id: str) -> tuple[dict[str, Scalar] | None, str | None]:
        if review_id in self._review_cache:
            return self._review_cache[review_id]
        frame = (
            pl.scan_parquet(self.reviews)
            .filter(pl.col("review_id").cast(pl.String) == review_id)
            .select("review_id", "product_id", "review_text_raw")
            .collect()
        )
        record = frame.row(0, named=True) if frame.height else None
        value = (record, f"{self.reviews.name}:{review_id}" if record else None)
        self._review_cache[review_id] = value
        return value


class InMemoryAuthoritativeLookup:
    """Small deterministic lookup for tests and offline validator callers."""

    def __init__(
        self,
        *,
        direct: dict[tuple[str, str], Scalar] | None = None,
        statistics: dict[tuple[str, str], Scalar] | None = None,
        reviews: dict[str, dict[str, Scalar]] | None = None,
    ):
        self.direct = direct or {}
        self.statistics = statistics or {}
        self.reviews = reviews or {}

    def direct_value(self, product_id: str, field: str) -> tuple[Scalar | None, str | None]:
        key = (product_id, field)
        return self.direct.get(key), f"in_memory:canonical:{product_id}:{field}" if key in self.direct else None

    def statistic_value(self, product_id: str, metric: str) -> tuple[Scalar | None, str | None]:
        key = (product_id, metric)
        return self.statistics.get(key), f"in_memory:statistics:{product_id}:{metric}" if key in self.statistics else None

    def review(self, review_id: str) -> tuple[dict[str, Scalar] | None, str | None]:
        return self.reviews.get(review_id), f"in_memory:review:{review_id}" if review_id in self.reviews else None


class DeterministicGroundingValidator:
    """Validate every structured claim without trusting fluent generated text."""

    def __init__(self, settings: GroundingSettings, authority: AuthoritativeLookup | None = None):
        self.settings = settings
        self.authority = authority

    @classmethod
    def from_settings(cls, settings: Settings) -> "DeterministicGroundingValidator":
        paths = settings.paths
        required = (paths.canonical_products, paths.product_statistics, paths.retrieval_corpus)
        authority = (
            ParquetAuthoritativeLookup(*required)  # type: ignore[arg-type]
            if all(path is not None and path.is_file() for path in required)
            else None
        )
        return cls(settings.grounding, authority)

    def validate(
        self,
        answer: GeneratedComparisonAnswer,
        context: GenerationContext,
        *,
        action_taken: ActionTaken | None = None,
    ) -> GroundingValidationResult:
        claims: list[ClaimResult] = []
        citations: list[CitationResult] = []
        warnings: list[str] = []
        claim_ids: set[str] = set()
        context_direct = {(item.product_id, item.field): item for item in context.direct_facts}
        context_stats = {
            (str(row["product_id"]), metric): value
            for row in context.aggregate_statistics
            for metric, value in row.items()
            if metric not in {"product_id", "source_layer"}
        }
        evidence = {
            item.review_id: (evidence_set, item)
            for evidence_set in context.retrieved_evidence
            for item in evidence_set.items
        }
        seen_citation_ids: set[str] = set()

        for claim in answer.direct_facts:
            claims.append(self._validate_direct(claim, context_direct, claim_ids))
        for finding in answer.aggregate_findings:
            claims.append(self._validate_aggregate(finding, context_stats, claim_ids))
        for finding in answer.review_findings:
            claim_result, citation_results, finding_warnings = self._validate_review(
                finding, evidence, seen_citation_ids, claim_ids
            )
            claims.append(claim_result)
            citations.extend(citation_results)
            warnings.extend(finding_warnings)
        claims.append(self._validate_recommendation(answer.recommendation, context, claim_ids))

        metrics = self._metrics(claims, citations, answer.recommendation, context)
        valid = not any(result.status in {"unsupported", "contradicted"} for result in claims)
        return GroundingValidationResult(
            validator_version=self.settings.validator_version,
            overall_status="accepted" if valid else "rejected",
            claim_results=claims,
            citation_results=citations,
            unsupported_claim_count=metrics.unsupported_claim_count,
            grounded_claim_count=metrics.grounded_claim_count,
            warnings=warnings,
            action_taken=action_taken or ("accepted" if valid else "rejected"),
            metrics=metrics,
        )

    def _validate_direct(
        self,
        claim: Any,
        context_direct: dict[tuple[str, str], Any],
        claim_ids: set[str],
    ) -> ClaimResult:
        duplicate = _duplicate_claim(claim.claim_id, claim_ids)
        source = context_direct.get((claim.product_id, claim.field))
        reference = f"phase9_context:direct:{claim.product_id}:{claim.field}"
        if duplicate:
            return _claim(claim.claim_id, "direct_product_fact", "unsupported", "duplicate_claim_id", reference)
        if source is None:
            return _claim(claim.claim_id, "direct_product_fact", "unsupported", "direct_fact_not_in_context", reference)
        if source.provenance_status != claim.provenance_status:
            return _claim(claim.claim_id, "direct_product_fact", "contradicted", "direct_provenance_mismatch", reference, source.value)
        if not _same_scalar(source.value, claim.value, self.settings.numeric_absolute_tolerance):
            return _claim(claim.claim_id, "direct_product_fact", "contradicted", "direct_value_mismatch", reference, source.value)
        if self.authority is not None:
            authoritative, artifact_reference = self.authority.direct_value(claim.product_id, claim.field)
            if artifact_reference is None:
                return _claim(claim.claim_id, "direct_product_fact", "unsupported", "canonical_product_not_found", reference)
            if not _same_scalar(authoritative, claim.value, self.settings.numeric_absolute_tolerance):
                return _claim(claim.claim_id, "direct_product_fact", "contradicted", "canonical_value_mismatch", artifact_reference, authoritative)
            reference = artifact_reference
        return _claim(claim.claim_id, "direct_product_fact", "grounded", "direct_value_matches", reference, claim.value)

    def _validate_aggregate(
        self,
        finding: AggregateFinding,
        context_stats: dict[tuple[str, str], Scalar],
        claim_ids: set[str],
    ) -> ClaimResult:
        duplicate = _duplicate_claim(finding.claim_id, claim_ids)
        expected = context_stats.get((finding.product_id, finding.metric), _MISSING)
        reference = f"phase9_context:statistics:{finding.product_id}:{finding.metric}"
        if duplicate:
            return _claim(finding.claim_id, "aggregate_statistic", "unsupported", "duplicate_claim_id", reference)
        if expected is _MISSING:
            return _claim(finding.claim_id, "aggregate_statistic", "unsupported", "aggregate_not_in_context", reference)
        if not _same_scalar(expected, finding.value, self.settings.numeric_absolute_tolerance):
            return _claim(finding.claim_id, "aggregate_statistic", "contradicted", "numeric_value_mismatch", reference, expected)
        denominator_result = self._validate_denominator(finding, context_stats)
        if denominator_result is not None:
            return _claim(
                finding.claim_id,
                "aggregate_statistic",
                denominator_result,
                "unsupported_denominator" if denominator_result == "unsupported" else "denominator_mismatch",
                reference,
                expected,
            )
        if self.authority is not None:
            authoritative, artifact_reference = self.authority.statistic_value(finding.product_id, finding.metric)
            if artifact_reference is None:
                return _claim(finding.claim_id, "aggregate_statistic", "unsupported", "statistics_product_not_found", reference)
            if not _same_scalar(authoritative, finding.value, self.settings.numeric_absolute_tolerance):
                return _claim(finding.claim_id, "aggregate_statistic", "contradicted", "statistics_value_mismatch", artifact_reference, authoritative)
            reference = artifact_reference
        return _claim(finding.claim_id, "aggregate_statistic", "grounded", "aggregate_value_matches", reference, finding.value)

    def _validate_denominator(
        self,
        finding: AggregateFinding,
        context_stats: dict[tuple[str, str], Scalar],
    ) -> ClaimStatus | None:
        if finding.numerator is None and finding.denominator is None:
            return None
        definition = _PERCENTAGE_DENOMINATORS.get(finding.metric)
        if definition is None or finding.numerator is None or finding.denominator is None:
            return "unsupported"
        expected_numerator = context_stats.get((finding.product_id, definition[0]))
        expected_denominator = context_stats.get((finding.product_id, definition[1]))
        if finding.numerator != expected_numerator or finding.denominator != expected_denominator:
            return "contradicted"
        if finding.denominator == 0:
            return "contradicted"
        if not _same_scalar(finding.numerator / finding.denominator, finding.value, self.settings.numeric_absolute_tolerance):
            return "contradicted"
        return None

    def _validate_review(
        self,
        finding: ReviewFinding,
        evidence: dict[str, tuple[Any, Any]],
        seen_citation_ids: set[str],
        claim_ids: set[str],
    ) -> tuple[ClaimResult, list[CitationResult], list[str]]:
        duplicate_claim = _duplicate_claim(finding.claim_id, claim_ids)
        results: list[CitationResult] = []
        warnings: list[str] = []
        excerpts: list[str] = []
        for citation in finding.citations:
            duplicate_citation = citation.review_id in seen_citation_ids
            seen_citation_ids.add(citation.review_id)
            item = evidence.get(citation.review_id)
            reference = f"generation_context:evidence:{citation.review_id}"
            authoritative, artifact_reference = self.authority.review(citation.review_id) if self.authority else (None, None)
            if duplicate_citation:
                results.append(_citation(finding, citation.review_id, "invalid", "duplicate_citation", reference))
            elif item is None:
                if authoritative is None and self.authority is not None:
                    results.append(_citation(finding, citation.review_id, "invalid", "citation_not_found", None))
                elif authoritative is not None and str(authoritative["product_id"]) != finding.product_id:
                    results.append(_citation(finding, citation.review_id, "invalid", "wrong_product", artifact_reference))
                else:
                    results.append(_citation(finding, citation.review_id, "invalid", "citation_not_in_context", artifact_reference))
            else:
                evidence_set, source = item
                if source.product_id != finding.product_id:
                    results.append(_citation(finding, citation.review_id, "invalid", "wrong_product", reference))
                elif evidence_set.criterion != finding.criterion:
                    results.append(_citation(finding, citation.review_id, "invalid", "citation_not_in_context", reference, "citation belongs to another evidence criterion"))
                elif not _excerpt_is_present(citation.excerpt, source.evidence_text):
                    results.append(_citation(finding, citation.review_id, "invalid", "citation_excerpt_mismatch", reference))
                elif self.authority is not None and authoritative is None:
                    results.append(_citation(finding, citation.review_id, "invalid", "citation_not_found", None))
                elif self.authority is not None and str(authoritative["product_id"]) != finding.product_id:
                    results.append(_citation(finding, citation.review_id, "invalid", "wrong_product", artifact_reference))
                else:
                    retrieval_signal = (
                        f"retrieval_rank={source.rank}; final_score={source.final_score}; "
                        f"reranker_score={source.reranker_score}"
                    )
                    results.append(
                        _citation(
                            finding,
                            citation.review_id,
                            "valid",
                            "citation_valid",
                            artifact_reference or reference,
                            retrieval_signal,
                        )
                    )
                    excerpts.append(citation.excerpt)
        reference = f"generation_context:evidence:{finding.product_id}:{finding.criterion}"
        if duplicate_claim:
            return _claim(finding.claim_id, "retrieved_review_evidence", "unsupported", "duplicate_claim_id", reference), results, warnings
        if any(item.status == "invalid" for item in results):
            reason = next(item.reason_code for item in results if item.status == "invalid")
            return _claim(finding.claim_id, "retrieved_review_evidence", "unsupported", reason, reference), results, warnings
        overlap = _token_overlap(finding.text, " ".join(excerpts))
        if len(overlap) < self.settings.minimum_review_lexical_overlap_tokens:
            return _claim(finding.claim_id, "retrieved_review_evidence", "unsupported", "unsupported_review_claim", reference), results, warnings
        warnings.append(
            f"{finding.claim_id}: lexical overlap ({', '.join(sorted(overlap))}) is a conservative signal, not semantic entailment proof."
        )
        return _claim(finding.claim_id, "retrieved_review_evidence", "grounded", "lexical_overlap_only", reference), results, warnings

    def _validate_recommendation(
        self,
        recommendation: Recommendation,
        context: GenerationContext,
        claim_ids: set[str],
    ) -> ClaimResult:
        claim_id = "recommendation"
        duplicate = _duplicate_claim(claim_id, claim_ids)
        reference = "phase9_context:authorization"
        if duplicate:
            return _claim(claim_id, "inference_or_recommendation", "unsupported", "duplicate_claim_id", reference)
        if recommendation.type != "inference":
            return _claim(claim_id, "inference_or_recommendation", "unsupported", "recommendation_not_labeled_inference", reference)
        if any(name not in context.authorization.criterion_statuses for name in recommendation.based_on_criteria):
            return _claim(claim_id, "inference_or_recommendation", "unsupported", "unknown_criterion", reference)
        for criterion, winners in recommendation.criterion_winner_product_ids.items():
            status = context.authorization.criterion_statuses.get(criterion)
            expected = context.authorization.criterion_winner_product_ids.get(criterion)
            if status is None:
                return _claim(claim_id, "inference_or_recommendation", "unsupported", "unknown_criterion", reference)
            if status == "inconclusive" and winners:
                return _claim(claim_id, "inference_or_recommendation", "contradicted", "inconclusive_criterion_winner", reference)
            if winners != expected:
                return _claim(claim_id, "inference_or_recommendation", "contradicted", "criterion_decision_contradiction", reference, expected)
        if recommendation.status == "conditional":
            for criterion in recommendation.based_on_criteria:
                if context.authorization.criterion_statuses[criterion] == "inconclusive":
                    return _claim(claim_id, "inference_or_recommendation", "contradicted", "inconclusive_criterion_winner", reference)
        overall = context.authorization
        if overall.overall_status == "inconclusive":
            if recommendation.status != "inconclusive" or recommendation.overall_winner_product_ids:
                return _claim(claim_id, "inference_or_recommendation", "contradicted", "inconclusive_overall_winner", reference)
        elif overall.overall_status == "neutral":
            if recommendation.overall_winner_product_ids:
                return _claim(claim_id, "inference_or_recommendation", "contradicted", "overall_winner_not_authorized", reference)
        elif recommendation.overall_winner_product_ids != overall.overall_winner_product_ids:
            return _claim(claim_id, "inference_or_recommendation", "contradicted", "overall_decision_contradiction", reference, overall.overall_winner_product_ids)
        return _claim(claim_id, "inference_or_recommendation", "grounded", "inference_matches_authorization", reference)

    def _metrics(
        self,
        claims: list[ClaimResult],
        citations: list[CitationResult],
        recommendation: Recommendation,
        context: GenerationContext,
    ) -> GroundingMetrics:
        total = len(claims)
        grounded = sum(item.status == "grounded" for item in claims)
        unsupported = sum(item.status in {"unsupported", "contradicted"} for item in claims)
        contradictions = sum(item.status == "contradicted" for item in claims)
        valid_citations = sum(item.status == "valid" for item in citations)
        covered = sum(item.authoritative_source_reference is not None and item.status == "grounded" for item in claims)
        inconclusive_cases = sum(status == "inconclusive" for status in context.authorization.criterion_statuses.values())
        correct_inconclusive = sum(
            not recommendation.criterion_winner_product_ids.get(criterion)
            for criterion, status in context.authorization.criterion_statuses.items()
            if status == "inconclusive"
        )
        if context.authorization.overall_status == "inconclusive":
            inconclusive_cases += 1
            correct_inconclusive += int(
                recommendation.status == "inconclusive" and not recommendation.overall_winner_product_ids
            )
        return GroundingMetrics(
            factual_claim_count=total,
            grounded_claim_count=grounded,
            unsupported_claim_count=unsupported,
            contradiction_count=contradictions,
            grounded_claim_ratio=_ratio(grounded, total),
            unsupported_claim_ratio=_ratio(unsupported, total),
            contradiction_rate=_ratio(contradictions, total),
            citation_count=len(citations),
            valid_citation_count=valid_citations,
            citation_correctness=_ratio(valid_citations, len(citations)),
            support_requiring_claim_count=total,
            evidence_covered_claim_count=covered,
            evidence_coverage=_ratio(covered, total),
            inconclusive_case_count=inconclusive_cases,
            correct_inconclusive_count=correct_inconclusive,
            inconclusive_correctness=_ratio(correct_inconclusive, inconclusive_cases),
        )

    def enforce(
        self,
        answer: GeneratedComparisonAnswer,
        context: GenerationContext,
        *,
        action: UnsupportedClaimAction,
        regenerate: Callable[[GroundingValidationResult], GeneratedComparisonAnswer] | None = None,
    ) -> GroundingPolicyOutcome:
        initial = self.validate(answer, context)
        if initial.valid:
            accepted = initial.model_copy(update={"action_taken": "accepted"})
            return GroundingPolicyOutcome(original_answer=answer, initial_validation=initial, final_answer=answer, final_validation=accepted, action_taken="accepted")
        if action == "reject":
            rejected = initial.model_copy(update={"action_taken": "rejected"})
            return GroundingPolicyOutcome(original_answer=answer, initial_validation=rejected, action_taken="rejected")
        if action == "remove_unsupported":
            cleaned = self.remove_unsupported(answer, initial, context)
            final = self.validate(cleaned, context, action_taken="removed_unsupported")
            return GroundingPolicyOutcome(original_answer=answer, initial_validation=initial, final_answer=cleaned if final.valid else None, final_validation=final, action_taken="removed_unsupported")
        if regenerate is None:
            rejected = initial.model_copy(update={"action_taken": "rejected", "warnings": [*initial.warnings, "rewrite_regenerate requires an explicit regeneration callback"]})
            return GroundingPolicyOutcome(original_answer=answer, initial_validation=rejected, action_taken="rejected")
        regenerated = regenerate(initial)
        final = self.validate(regenerated, context, action_taken="rewrite_regenerated")
        return GroundingPolicyOutcome(original_answer=answer, initial_validation=initial, final_answer=regenerated if final.valid else None, final_validation=final, action_taken="rewrite_regenerated")

    def remove_unsupported(
        self,
        answer: GeneratedComparisonAnswer,
        result: GroundingValidationResult,
        context: GenerationContext,
    ) -> GeneratedComparisonAnswer:
        unsupported = {item.claim_id for item in result.unsupported_claims}
        recommendation = answer.recommendation
        if "recommendation" in unsupported:
            authorization = context.authorization
            if authorization.overall_status == "inconclusive":
                recommendation = Recommendation(
                    text="پیشنهاد به علت نتیجهٔ نامشخصِ قطعی ارائه نمی‌شود.",
                    status="inconclusive",
                )
            elif authorization.overall_status == "weighted_winner":
                recommendation = Recommendation(
                    text="نتیجهٔ کلی فقط در چارچوب اولویت‌های اعلام‌شده قابل استفاده است.",
                    status="conditional",
                    conditional_on=["بر اساس اولویت‌های صریح کاربر"],
                    overall_winner_product_ids=authorization.overall_winner_product_ids,
                )
            else:
                recommendation = Recommendation(
                    text="پیشنهاد به علت پشتیبانی ناکافیِ متن تولیدشده ارائه نمی‌شود.",
                    status="not_authorized",
                )
        return answer.model_copy(
            update={
                "direct_facts": [item for item in answer.direct_facts if item.claim_id not in unsupported],
                "aggregate_findings": [item for item in answer.aggregate_findings if item.claim_id not in unsupported],
                "review_findings": [item for item in answer.review_findings if item.claim_id not in unsupported],
                "recommendation": recommendation,
            }
        )


class GroundingAuditStore:
    """Persist original/final answers and validation decisions without secrets."""

    def __init__(self, root: Path):
        self.root = root

    def write(self, record: GroundingAuditRecord) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        key = sha256(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        path = self.root / f"{key}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path


_MISSING = object()


def _claim(
    claim_id: str,
    layer: ClaimLayer,
    status: ClaimStatus,
    reason: str,
    reference: str | None,
    value: Scalar | None = None,
    detail: str | None = None,
) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        source_layer=layer,
        status=status,
        reason_code=reason,
        authoritative_source_reference=reference,
        authoritative_value=value,
        detail=detail,
    )


def _citation(
    finding: ReviewFinding,
    review_id: str,
    status: CitationStatus,
    reason: str,
    reference: str | None,
    detail: str | None = None,
) -> CitationResult:
    return CitationResult(
        claim_id=finding.claim_id,
        review_id=review_id,
        claimed_product_id=finding.product_id,
        status=status,
        reason_code=reason,
        authoritative_source_reference=reference,
        detail=detail,
    )


def _duplicate_claim(claim_id: str, known: set[str]) -> bool:
    duplicate = claim_id in known
    known.add(claim_id)
    return duplicate


def _same_scalar(left: object, right: object, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def _excerpt_is_present(excerpt: str, evidence_text: str | None) -> bool:
    if evidence_text is None:
        return False
    return " ".join(excerpt.split()) in " ".join(evidence_text.split())


def _token_overlap(claim: str, evidence: str) -> set[str]:
    return _tokens(claim) & _tokens(evidence)


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", value).replace("ي", "ی").replace("ك", "ک").lower()
    return {token for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE) if len(token) > 1 and token not in _STOP_TOKENS}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
