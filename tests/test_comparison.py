from __future__ import annotations

from digikala_comparison.comparison import (
    CanonicalProductIdentity,
    ComparisonSettings,
    CriterionRequest,
    DirectFieldValue,
    DirectProductFields,
    FullDataProductStatistics,
    PreferencePolicy,
    ProductComparisonEngine,
    ProductComparisonInput,
)
from digikala_comparison.evidence import EvidenceAudit, EvidenceItem, EvidenceSet, ScoreDistributionSummary


def _direct(field: str, value: str | int | bool | None, *, status: str = "stable") -> DirectFieldValue:
    semantics = {
        "Price": "numeric_raw_snapshot_price",
        "Rate": "validated_product_rate_0_to_100",
        "Rate_cnt": "rating_support_count",
        "min_price_last_month": "numeric_raw_historical_price",
        "Is_Fake": "boolean_snapshot_flag",
    }
    return DirectFieldValue(
        field=field,
        value=value,
        semantic_type=semantics.get(field, "categorical_product_metadata"),
        provenance_status=status,  # type: ignore[arg-type]
        source_distinct_count=2 if status == "conflicted" else 1,
    )


def _product(
    product_id: str,
    *,
    price: int | None = 100,
    rate: int = 80,
    rate_count: int = 100,
    recommended: int = 80,
    not_recommended: int = 20,
    no_idea: int = 0,
    metadata_conflict_field: str | None = None,
    evidence: list[EvidenceSet] | None = None,
) -> ProductComparisonInput:
    known = recommended + not_recommended + no_idea
    opinionated = recommended + not_recommended
    conflicted = [metadata_conflict_field] if metadata_conflict_field else []
    direct = DirectProductFields(
        price=_direct("Price", price, status="conflicted" if metadata_conflict_field == "Price" else "missing" if price is None else "stable"),
        rate=_direct("Rate", rate, status="conflicted" if metadata_conflict_field == "Rate" else "stable"),
        rate_count=_direct("Rate_cnt", rate_count, status="conflicted" if metadata_conflict_field == "Rate_cnt" else "stable"),
        min_price_last_month=_direct("min_price_last_month", price, status="stable"),
        is_fake=_direct("Is_Fake", False, status="stable"),
        brand=_direct("Brand", "brand", status="stable"),
        category1=_direct("Category1", "category", status="stable"),
        category2=_direct("Category2", "subcategory", status="stable"),
        sub_category=_direct("sub_category", "sub", status="stable"),
    )
    return ProductComparisonInput(
        identity=CanonicalProductIdentity(
            product_id=product_id,
            title_fa=f"product {product_id}",
            brand="brand",
            category1="category",
            category2="subcategory",
            sub_category="sub",
            canonicalization_status="unique_source_row" if not conflicted else "same_identity_mutable_fields_differ",
            source_row_count=1 if not conflicted else 2,
            has_metadata_conflict=bool(conflicted),
            conflicting_fields=conflicted,
        ),
        direct_fields=direct,
        full_statistics=FullDataProductStatistics(
            product_id=product_id,
            total_review_count=max(known, 1),
            buyer_review_count=max(known - 5, 0),
            non_buyer_review_count=5 if known >= 5 else known,
            unknown_buyer_review_count=0,
            review_rate_valid_count=max(known, 1),
            average_review_rate=4.0,
            median_review_rate=4.0,
            recommended_count=recommended,
            not_recommended_count=not_recommended,
            no_idea_count=no_idea,
            recommendation_known_count=known,
            opinionated_review_count=opinionated,
            recommended_percentage=None if known == 0 else recommended / known,
            not_recommended_percentage=None if known == 0 else not_recommended / known,
            no_idea_percentage=None if known == 0 else no_idea / known,
            opinionated_recommend_percentage=None if opinionated == 0 else recommended / opinionated,
        ),
        evidence_sets=evidence or [],
    )


def _evidence(product_id: str, criterion: str, statuses: list[str | None]) -> EvidenceSet:
    items = [
        EvidenceItem(
            review_id=f"{product_id}-{index}",
            product_id=product_id,
            rank=index + 1,
            final_score=1.0 / (index + 1),
            raw_evidence_text=f"raw {index}",
            is_buyer=True,
            recommendation_status=status,
            likes=0,
            dislikes=0,
            audit=EvidenceAudit(final_rank=index + 1),
        )
        for index, status in enumerate(statuses)
    ]
    return EvidenceSet(
        product_id=product_id,
        criterion=criterion,
        query=criterion,
        retrieval_method="bm25",
        retrieval_method_version="frozen-v1",
        experiment_manifest_sha256="a" * 64,
        requested_top_k=len(items),
        retrieved_count=len(items),
        eligible_product_review_count=100,
        retrieval_status="sufficient_candidates" if items else "no_evidence",
        score_distribution=ScoreDistributionSummary(
            count=len(items),
            minimum=min((item.final_score for item in items), default=None),
            maximum=max((item.final_score for item in items), default=None),
            mean=sum(item.final_score for item in items) / len(items) if items else None,
        ),
        evidence_items=items,
    )


def _engine(**changes: object) -> ProductComparisonEngine:
    return ProductComparisonEngine(ComparisonSettings(**changes))


def test_recommendation_percentage_preserves_numerator_denominator_and_winner() -> None:
    result = _engine().compare(
        [_product("a", recommended=90, not_recommended=10), _product("b", recommended=70, not_recommended=30)],
        [CriterionRequest(name="recommendation")],
    )
    decision = result.criterion_decisions[0]
    assert decision.status == "winner"
    assert decision.winner_product_ids == ["a"]
    assert [(item.numerator, item.denominator, item.percentage) for item in decision.support] == [(90, 100, 0.9), (70, 100, 0.7)]


def test_zero_denominator_and_minimum_support_are_inconclusive() -> None:
    zero = _engine().compare(
        [_product("a", recommended=0, not_recommended=0), _product("b", recommended=80, not_recommended=20)],
        [CriterionRequest(name="recommendation")],
    ).criterion_decisions[0]
    small = _engine().compare(
        [_product("a", recommended=9, not_recommended=1), _product("b", recommended=80, not_recommended=20)],
        [CriterionRequest(name="recommendation")],
    ).criterion_decisions[0]
    assert (zero.status, zero.reason_code) == ("inconclusive", "ZERO_DENOMINATOR")
    assert (small.status, small.reason_code) == ("inconclusive", "INSUFFICIENT_SUPPORT")


def test_practical_threshold_and_exact_tie_are_distinct() -> None:
    near = _engine().compare(
        [_product("a", recommended=80, not_recommended=20), _product("b", recommended=77, not_recommended=23)],
        [CriterionRequest(name="recommendation")],
    ).criterion_decisions[0]
    tie = _engine().compare(
        [_product("a", recommended=80, not_recommended=20), _product("b", recommended=80, not_recommended=20)],
        [CriterionRequest(name="recommendation")],
    ).criterion_decisions[0]
    assert (near.status, near.reason_code) == ("inconclusive", "PRACTICAL_DIFFERENCE_NOT_MET")
    assert (tie.status, tie.reason_code, tie.winner_product_ids) == ("tie", "EXACT_TIE", ["a", "b"])


def test_missing_or_conflicted_direct_metadata_is_not_forced_to_a_winner() -> None:
    missing = _engine().compare([_product("a", price=None), _product("b", price=120)], [CriterionRequest(name="price")]).criterion_decisions[0]
    conflicted = _engine().compare([_product("a", price=100, metadata_conflict_field="Price"), _product("b", price=120)], [CriterionRequest(name="price")]).criterion_decisions[0]
    assert (missing.status, missing.reason_code) == ("inconclusive", "MISSING_REQUIRED_DATA")
    assert (conflicted.status, conflicted.reason_code) == ("inconclusive", "METADATA_CONFLICT")


def test_multi_product_ranking_is_emitted_only_when_all_gaps_are_practical() -> None:
    decision = _engine().compare(
        [_product("a", price=100), _product("b", price=120), _product("c", price=150)],
        [CriterionRequest(name="price")],
    ).criterion_decisions[0]
    assert decision.status == "winner"
    assert decision.winner_product_ids == ["a"]
    assert decision.ranking_product_ids == ["a", "b", "c"]


def test_evidence_is_attached_with_review_ids_but_cannot_change_population_decision() -> None:
    a = _product("a", recommended=60, not_recommended=40, evidence=[_evidence("a", "recommendation", ["recommended"] * 3)])
    b = _product("b", recommended=80, not_recommended=20, evidence=[_evidence("b", "recommendation", ["not_recommended"] * 3)])
    result = _engine().compare([a, b], [CriterionRequest(name="recommendation")])
    decision = result.criterion_decisions[0]
    attachment = result.retrieved_evidence[0]
    assert decision.winner_product_ids == ["b"]
    assert [item.review_id for item in attachment.evidence_sets[0].evidence_items] == ["a-0", "a-1", "a-2"]
    assert [(item.positive_count, item.negative_count) for item in attachment.evidence_counts] == [(3, 0), (0, 3)]
    assert attachment.evidence_counts[0].scope == "within_retrieved_evidence"


def test_evidence_only_criterion_is_informational_or_inconclusive_not_a_winner() -> None:
    sufficient = _engine().compare(
        [_product("a", evidence=[_evidence("a", "battery", ["recommended"] * 3)]), _product("b", evidence=[_evidence("b", "battery", ["not_recommended"] * 3)])],
        [CriterionRequest(name="battery")],
    ).criterion_decisions[0]
    absent = _engine().compare([_product("a"), _product("b")], [CriterionRequest(name="battery")]).criterion_decisions[0]
    assert (sufficient.status, sufficient.reason_code, sufficient.winner_product_ids) == ("informational", "RETRIEVED_EVIDENCE_ONLY", [])
    assert (absent.status, absent.reason_code) == ("inconclusive", "NO_RETRIEVED_EVIDENCE")


def test_no_overall_winner_without_explicit_policy_and_weighted_result_with_one() -> None:
    products = [_product("a", price=100), _product("b", price=130)]
    neutral = _engine().compare(products, [CriterionRequest(name="price")])
    weighted = _engine().compare(products, [CriterionRequest(name="price")], PreferencePolicy(weights={"price": 1.0}))
    assert (neutral.overall.status, neutral.overall.winner_product_ids) == ("neutral", [])
    assert (weighted.overall.status, weighted.overall.winner_product_ids) == ("weighted_winner", ["a"])


def test_preference_policy_normalizes_public_criterion_names() -> None:
    result = _engine().compare(
        [_product("a", recommended=90, not_recommended=10), _product("b", recommended=70, not_recommended=30)],
        [CriterionRequest(name="recommendation")],
        PreferencePolicy(weights={"recommendation": 1.0}),
    )
    assert (result.overall.status, result.overall.winner_product_ids) == ("weighted_winner", ["a"])


def test_unvalidated_review_rate_is_explicitly_inconclusive() -> None:
    decision = _engine().compare([_product("a"), _product("b")], [CriterionRequest(name="average_review_rate")]).criterion_decisions[0]
    assert (decision.status, decision.reason_code) == ("inconclusive", "UNVALIDATED_FIELD_SEMANTICS")
