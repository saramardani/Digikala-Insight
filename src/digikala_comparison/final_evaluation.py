"""Phase 12 reproducible end-to-end evaluation for product comparison.

The evaluator deliberately reuses frozen artifacts and the production evidence
retriever.  It never manufactures dense, hybrid, reranker, LLM, or human
evaluation results when those prerequisites are unavailable.  The default mode
is an offline deterministic rendering plus the real Phase 11 validator; an
explicit ``with_llm`` opt-in is required before an API call can be made.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .comparison import (
    ComparisonDataStore,
    CriterionRequest,
    PreferencePolicy,
    ProductComparisonEngine,
    canonical_criterion,
    criterion_uses_retrieved_evidence,
)
from .config import Settings
from .evidence import EvidenceSet, ProductionEvidenceRetriever
from .generation import (
    AggregateFinding,
    Caveat,
    DirectFactClaim,
    GeneratedComparisonAnswer,
    GenerationContext,
    Recommendation,
    ReviewCitation,
    ReviewFinding,
    StructuredComparisonGenerator,
    build_generation_context,
    render_persian_answer,
)
from .grounding import DeterministicGroundingValidator, GroundingValidationResult
from .resolver import ProductResolver
from .runtime import peak_process_memory_bytes


FINAL_SCHEMA_VERSION = "final-system-evaluation-v1"
CASE_FILE_NAME = "evaluation_cases.json"
HUMAN_TEMPLATE_NAME = "human_answer_quality_template.csv"
HUMAN_ANNOTATIONS_NAME = "human_answer_quality_annotations.csv"

_DIRECT_CRITERIA = {
    "price",
    "min_price_last_month",
    "rate",
    "rate_count",
    "is_fake",
    "brand",
    "category",
    "category1",
    "category2",
    "sub_category",
}
_PERCENTAGE_METRICS = {
    "recommended_percentage": ("recommended_count", "recommendation_known_count"),
    "not_recommended_percentage": ("not_recommended_count", "recommendation_known_count"),
    "opinionated_recommend_percentage": ("recommended_count", "opinionated_review_count"),
}
_HUMAN_SCORE_COLUMNS = (
    "relevance_1_to_5",
    "clarity_1_to_5",
    "source_separation_1_to_5",
    "recommendation_usefulness_1_to_5",
    "uncertainty_appropriateness_1_to_5",
    "citation_usefulness_1_to_5",
    "semantic_citation_support_1_to_5",
)


class StrictEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinalEvaluationCase(StrictEvaluationModel):
    """A frozen evaluation input, never a generated answer or judgment."""

    case_id: str
    split: Literal["development_debug", "final"]
    case_type: Literal["comparison", "resolver"]
    scenario: str
    question: str
    product_ids: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)
    evidence_queries: dict[str, str] = Field(default_factory=dict)
    preference_weights: dict[str, float] = Field(default_factory=dict)
    resolver_reference: str | None = None
    expected_status: str | None = None
    expects_inconclusive: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def _validate_case_shape(self) -> "FinalEvaluationCase":
        if not self.case_id.strip() or not self.scenario.strip() or not self.question.strip():
            raise ValueError("evaluation case identifiers, scenario, and question must be non-empty")
        if self.case_type == "comparison":
            if len(self.product_ids) < 2 or len(set(self.product_ids)) != len(self.product_ids):
                raise ValueError("comparison cases require at least two distinct product IDs")
            if not self.criteria:
                raise ValueError("comparison cases require at least one criterion")
        elif not self.resolver_reference:
            raise ValueError("resolver cases require resolver_reference")
        return self


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _selected_row(frame: pl.LazyFrame, *, descending: bool = False, column: str = "product_id") -> dict[str, Any]:
    rows = frame.sort(column, descending=descending).head(1).collect().to_dicts()
    if not rows:
        raise RuntimeError("the pinned processed data does not contain a required final-evaluation candidate")
    return rows[0]


def build_frozen_evaluation_cases(settings: Settings) -> list[FinalEvaluationCase]:
    """Derive a balanced case set from pinned real records, once, before freeze.

    The selection policy is deterministic (explicit filters and stable ordering),
    so the first written case file can subsequently be reused unchanged.
    """

    if settings.paths.canonical_products is None or settings.paths.product_statistics is None:
        raise ValueError("canonical_products and product_statistics paths are required")
    canonical = pl.scan_parquet(settings.paths.canonical_products)
    statistics = pl.scan_parquet(settings.paths.product_statistics).select(
        "product_id",
        "total_review_count",
        "recommendation_known_count",
        "recommended_count",
        "not_recommended_count",
        "recommended_percentage",
    )
    base = canonical.join(statistics, on="product_id", how="inner")
    stable_price = base.filter(
        pl.col("Price").is_not_null()
        & (pl.col("Price") > 0)
        & ~pl.col("has_metadata_conflict")
    )
    low_price = _selected_row(stable_price, column="Price")
    high_price = _selected_row(stable_price, descending=True, column="Price")
    middle_price = _selected_row(
        stable_price.filter(pl.col("product_id") != low_price["product_id"])
        .filter(pl.col("product_id") != high_price["product_id"]),
        column="product_id",
    )

    rate_rows = (
        base.filter(
            (pl.col("recommendation_known_count") >= settings.comparison.minimum_percentage_denominator)
            & pl.col("recommended_percentage").is_not_null()
            & ~pl.col("has_metadata_conflict")
        )
        .select(
            "product_id",
            "recommended_percentage",
            "recommendation_known_count",
            "total_review_count",
        )
        .sort("product_id")
        .head(5_000)
        .collect()
        .to_dicts()
    )
    if len(rate_rows) < 2:
        raise RuntimeError("not enough products with full recommendation statistics for final evaluation")
    sorted_rates = sorted(rate_rows, key=lambda row: (float(row["recommended_percentage"]), str(row["product_id"])))
    satisfaction_pair = (sorted_rates[0], sorted_rates[-1])
    near_pairs = [
        (left, right)
        for left, right in zip(sorted_rates, sorted_rates[1:])
        if 0.0 < abs(float(left["recommended_percentage"]) - float(right["recommended_percentage"]))
        < settings.comparison.practical_percentage_point_difference
    ]
    near_pair = near_pairs[0] if near_pairs else (sorted_rates[0], sorted_rates[1])

    zero_review = _selected_row(base.filter(pl.col("total_review_count") == 0))
    conflicting_price = _selected_row(
        base.filter(
            pl.col("has_metadata_conflict")
            & pl.col("conflicting_fields").list.contains("Price")
        )
    )
    conflicting_reviews = _selected_row(
        base.filter(
            (pl.col("recommendation_known_count") >= settings.comparison.minimum_percentage_denominator)
            & (pl.col("recommended_count") >= 5)
            & (pl.col("not_recommended_count") >= 5)
        ),
        column="product_id",
    )
    sparse = _selected_row(
        base.filter((pl.col("total_review_count") > 0) & (pl.col("total_review_count") < 3)),
        column="total_review_count",
    )

    duplicate_titles = (
        canonical.group_by("normalized_title")
        .len(name="title_count")
        .filter(pl.col("normalized_title").is_not_null() & (pl.col("title_count") > 1))
        .sort("normalized_title")
        .head(1)
        .collect()
    )
    if duplicate_titles.height == 0:
        raise RuntimeError("the pinned canonical table has no duplicate normalized title for ambiguity evaluation")
    duplicate_title = str(duplicate_titles[0, "normalized_title"])
    variant = _selected_row(
        canonical.filter(pl.col("normalized_title").str.contains(r"[A-Za-z]+[0-9]+")),
        column="product_id",
    )
    variant_query = f"{variant['normalized_title']} z999model"

    # These IDs occur in the frozen Phase 8 test benchmark and make the
    # review-evidence scenario traceable to an existing query/qrel artifact.
    frozen_evidence_ids = ["82098", "514309"]
    found_frozen = set(
        canonical.filter(pl.col("product_id").cast(pl.String).is_in(frozen_evidence_ids))
        .select(pl.col("product_id").cast(pl.String))
        .collect()
        .to_series()
        .to_list()
    )
    evidence_pair = frozen_evidence_ids if set(frozen_evidence_ids) == found_frozen else [
        str(satisfaction_pair[0]["product_id"]),
        str(satisfaction_pair[1]["product_id"]),
    ]

    def comparison(
        case_id: str,
        split: Literal["development_debug", "final"],
        scenario: str,
        question: str,
        product_ids: list[str],
        criteria: list[str],
        *,
        evidence_queries: dict[str, str] | None = None,
        preference_weights: dict[str, float] | None = None,
        expects_inconclusive: bool = False,
        notes: str = "",
    ) -> FinalEvaluationCase:
        return FinalEvaluationCase(
            case_id=case_id,
            split=split,
            case_type="comparison",
            scenario=scenario,
            question=question,
            product_ids=product_ids,
            criteria=criteria,
            evidence_queries=evidence_queries or {},
            preference_weights=preference_weights or {},
            expects_inconclusive=expects_inconclusive,
            notes=notes,
        )

    return [
        comparison(
            "debug_clear_price",
            "development_debug",
            "clear_direct_price_winner",
            "از نظر قیمت ثبت‌شده کدام محصول مناسب‌تر است؟",
            [str(low_price["product_id"]), str(high_price["product_id"])],
            ["price"],
            preference_weights={"price": 1.0},
            notes="Two stable canonical snapshots selected from the pinned data by minimum and maximum raw Price.",
        ),
        comparison(
            "debug_review_evidence",
            "development_debug",
            "review_evidence_with_ids",
            "تجربهٔ کاربران دربارهٔ کیفیت این دو محصول چیست؟",
            evidence_pair,
            ["quality"],
            evidence_queries={"quality": "کیفیت"},
            notes="Uses two product IDs already present in the frozen retrieval benchmark when available.",
        ),
        comparison(
            "final_multi_product_price",
            "final",
            "multi_product_direct_data",
            "سه محصول را از نظر قیمت ثبت‌شده مقایسه کن.",
            [str(low_price["product_id"]), str(middle_price["product_id"]), str(high_price["product_id"])],
            ["price"],
            notes="Multi-product support; no currency is inferred from raw snapshot prices.",
        ),
        comparison(
            "final_satisfaction",
            "final",
            "full_population_satisfaction",
            "کدام محصول از نظر نسبت توصیه‌شدن در کل دیدگاه‌های واجد وضعیت بهتر است؟",
            [str(satisfaction_pair[0]["product_id"]), str(satisfaction_pair[1]["product_id"])],
            ["recommendation"],
            evidence_queries={"recommendation": "رضایت از خرید"},
            preference_weights={"recommendation": 1.0},
            notes="Full-population percentages remain distinct from attached Top-K review evidence.",
        ),
        comparison(
            "final_near_tie",
            "final",
            "near_tie_statistics",
            "از نظر رضایت کاربران کدام‌یک بهتر است؟",
            [str(near_pair[0]["product_id"]), str(near_pair[1]["product_id"])],
            ["recommendation"],
            evidence_queries={"recommendation": "رضایت"},
            expects_inconclusive=True,
            notes="Pair selected with a sub-threshold non-zero difference when present; this tests the practical-difference abstention rule.",
        ),
        comparison(
            "final_conflicting_reviews",
            "final",
            "conflicting_review_population",
            "بازخوردهای مثبت و منفی این محصولات را با احتیاط مقایسه کن.",
            [str(conflicting_reviews["product_id"]), str(satisfaction_pair[1]["product_id"])],
            ["recommendation"],
            evidence_queries={"recommendation": "نکات مثبت و منفی"},
            notes="One selected product has at least five recommended and five not-recommended full-population records.",
        ),
        comparison(
            "final_sparse_reviews",
            "final",
            "sparse_review_product",
            "آیا برای کیفیت این محصولات شواهد کافی وجود دارد؟",
            [str(sparse["product_id"]), str(low_price["product_id"])],
            ["quality"],
            evidence_queries={"quality": "کیفیت"},
            expects_inconclusive=True,
            notes="One product has fewer than three total reviews in full statistics.",
        ),
        comparison(
            "final_no_evidence",
            "final",
            "no_evidence_inconclusive",
            "با تکیه بر نظر کاربران، کیفیت این دو محصول را مقایسه کن.",
            [str(zero_review["product_id"]), str(low_price["product_id"])],
            ["quality"],
            evidence_queries={"quality": "کیفیت"},
            expects_inconclusive=True,
            notes="One product is selected from the full-statistics zero-review population.",
        ),
        comparison(
            "final_metadata_conflict",
            "final",
            "conflicting_product_metadata",
            "از نظر قیمت ثبت‌شده کدام محصول ارزان‌تر است؟",
            [str(conflicting_price["product_id"]), str(low_price["product_id"])],
            ["price"],
            expects_inconclusive=True,
            notes="The first product has an explicit canonical Price conflict; configured policy blocks a price winner.",
        ),
        FinalEvaluationCase(
            case_id="final_ambiguous_reference",
            split="final",
            case_type="resolver",
            scenario="ambiguous_product_name",
            question="این نام محصول به کدام رکورد اشاره می‌کند؟",
            resolver_reference=duplicate_title,
            expected_status="ambiguous",
            notes="A normalized title with more than one canonical product ID.",
        ),
        FinalEvaluationCase(
            case_id="final_variant_protection",
            split="final",
            case_type="resolver",
            scenario="variant_model_protection",
            question="این نام دارای مدل نامعتبر را پیدا کن.",
            resolver_reference=variant_query,
            expected_status="not_found",
            notes="A real model-bearing title plus an impossible model token; a nearby variant must not be silently selected.",
        ),
    ]


def load_or_create_evaluation_cases(
    settings: Settings, root: Path, *, rebuild: bool = False
) -> tuple[list[FinalEvaluationCase], Path, str]:
    path = root / CASE_FILE_NAME
    if path.is_file() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = [FinalEvaluationCase.model_validate(item) for item in payload["cases"]]
        return cases, path, _sha256_text(_stable_json(payload))
    cases = build_frozen_evaluation_cases(settings)
    payload = {
        "schema_version": settings.final_evaluation.case_set_version,
        "dataset_revision": settings.dataset.revision,
        "selection_policy": "deterministic filters over pinned canonical products and full-product statistics; written once and reused unless --rebuild-evaluation-set is explicit",
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    _write_json(path, payload)
    return cases, path, _sha256_text(_stable_json(payload))


def build_final_system_manifest(settings: Settings, *, case_set_sha256: str) -> dict[str, Any]:
    """Capture every runtime-relevant version without reading any secret."""

    required_paths = {
        "canonical_products": settings.paths.canonical_products,
        "product_statistics": settings.paths.product_statistics,
        "retrieval_corpus": settings.paths.retrieval_corpus,
        "retrieval_experiment_manifest": settings.paths.retrieval_experiment_manifest,
        "retrieval_benchmark": settings.paths.reranker_evaluation_report,
    }
    file_inputs = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for name, path in required_paths.items()
        if path is not None and path.is_file()
    }
    selection = _load_json(settings.paths.reranker_selection_report)
    retrieval_manifest = _load_json(settings.paths.retrieval_experiment_manifest)
    package_versions = {}
    for package in ("polars", "pydantic", "rapidfuzz", "openai", "FlagEmbedding", "faiss-cpu"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not_installed"
    canonical_source = Path(__file__).with_name("canonicalization.py")
    implementation_sources = {
        name: Path(__file__).with_name(name)
        for name in (
            "comparison.py",
            "resolver.py",
            "generation.py",
            "grounding.py",
            "final_evaluation.py",
        )
    }
    return {
        "schema_version": FINAL_SCHEMA_VERSION,
        "evaluation_version": settings.final_evaluation.evaluation_version,
        "dataset": {
            "repository": settings.dataset.repository,
            "revision": settings.dataset.revision,
            "source_url": settings.dataset.source_url,
        },
        "case_set_sha256": case_set_sha256,
        "canonicalization": {
            "implementation": "deterministic_mode_with_lexical_tiebreak",
            "implementation_source_sha256": _sha256_file(canonical_source),
            "canonical_artifact": file_inputs.get("canonical_products"),
        },
        "review_eligibility": asdict(settings.review_eligibility),
        "normalization": asdict(settings.normalization),
        "resolver": asdict(settings.resolution),
        "retrieval": {
            "selected_production_retriever": selection.get("selected_method"),
            "selection": selection,
            "experiment_manifest_sha256": _sha256_text(_stable_json(retrieval_manifest)),
            "bm25": asdict(settings.bm25),
            "dense": asdict(settings.dense),
            "hybrid": asdict(settings.hybrid),
            "reranker": asdict(settings.reranker),
        },
        "comparison_thresholds": asdict(settings.comparison),
        "generation": {
            "provider": settings.generation.provider,
            "model": settings.generation.model,
            "temperature": settings.generation.temperature,
            "max_output_tokens": settings.generation.max_output_tokens,
            "timeout_seconds": settings.generation.timeout_seconds,
            "prompt_version": settings.generation.prompt_version,
            "schema_version": settings.generation.schema_version,
            "cost_rates_usd_per_million_tokens": {
                "input": settings.generation.input_token_cost_per_million_usd,
                "output": settings.generation.output_token_cost_per_million_usd,
            },
            "api_key_environment_variable": settings.generation.api_key_environment_variable,
            "secret_value_persisted": False,
        },
        "grounding": asdict(settings.grounding),
        "implementation_source_sha256": {
            name: _sha256_file(path) for name, path in implementation_sources.items()
        },
        "input_artifacts": file_inputs,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "reproduction_command": "digikala-evaluate-final --config config/default.toml --no-llm",
    }


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "missing", "path": str(path) if path else None}
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_inputs_and_result(
    case: FinalEvaluationCase,
    *,
    store: ComparisonDataStore,
    engine: ProductComparisonEngine,
    retriever: ProductionEvidenceRetriever,
    top_k: int,
) -> tuple[Any, list[Any], dict[str, float]]:
    timings: dict[str, float] = {"statistics_lookup": 0.0, "retrieval": 0.0, "deterministic_comparison": 0.0}
    started = perf_counter()
    products = store.load_products(case.product_ids)
    timings["statistics_lookup"] = (perf_counter() - started) * 1000
    requests = [
        CriterionRequest(name=name, evidence_query=case.evidence_queries.get(name))
        for name in case.criteria
    ]
    evidence_by_id: dict[str, list[EvidenceSet]] = {product.identity.product_id: [] for product in products}
    for request in requests:
        if not request.attach_evidence or not criterion_uses_retrieved_evidence(request.name):
            continue
        query = request.evidence_query or request.name
        for product in products:
            retrieval_started = perf_counter()
            evidence = retriever.retrieve_evidence(product.identity.product_id, request.name, query, top_k)
            timings["retrieval"] += (perf_counter() - retrieval_started) * 1000
            if evidence.product_id != product.identity.product_id or any(
                item.product_id != product.identity.product_id for item in evidence.evidence_items
            ):
                raise RuntimeError("cross-product review escaped the production evidence boundary")
            if len({item.review_id for item in evidence.evidence_items}) != len(evidence.evidence_items):
                raise RuntimeError("duplicate review_id returned by production evidence retriever")
            evidence_by_id[product.identity.product_id].append(evidence)
    inputs = [
        product.model_copy(update={"evidence_sets": evidence_by_id[product.identity.product_id]})
        for product in products
    ]
    policy = PreferencePolicy(weights=case.preference_weights) if case.preference_weights else None
    started = perf_counter()
    result = engine.compare(inputs, requests, policy)
    timings["deterministic_comparison"] = (perf_counter() - started) * 1000
    return result, inputs, timings


def deterministic_template_answer(
    result: Any, context: GenerationContext
) -> GeneratedComparisonAnswer:
    """Build a deliberately minimal baseline answer from already-authoritative data.

    Review content is quoted as retrieved evidence rather than summarized.  It
    makes the offline baseline auditable and avoids pretending that a template
    understands a review semantically.
    """

    source_by_key = {(source.product_id, source.field): source for source in context.direct_facts}
    direct_claims: list[DirectFactClaim] = []
    seen_direct: set[tuple[str, str]] = set()
    for decision in result.criterion_decisions:
        criterion = canonical_criterion(decision.criterion)
        if criterion not in _DIRECT_CRITERIA:
            continue
        fields = ("category1", "category2", "sub_category") if criterion == "category" else (criterion,)
        for product in result.products:
            for field in fields:
                source = source_by_key.get((product.product_id, field))
                if source is None or (product.product_id, field) in seen_direct:
                    continue
                seen_direct.add((product.product_id, field))
                direct_claims.append(
                    DirectFactClaim(
                        claim_id=f"direct:{product.product_id}:{field}",
                        product_id=product.product_id,
                        field=field,
                        value=source.value,
                        provenance_status=source.provenance_status,
                    )
                )

    statistics = {str(item["product_id"]): item for item in context.aggregate_statistics}
    aggregate_claims: list[AggregateFinding] = []
    for decision in result.criterion_decisions:
        metric = canonical_criterion(decision.criterion)
        if metric not in _PERCENTAGE_METRICS and metric not in {"buyer_review_count"}:
            continue
        for product_id, source in statistics.items():
            value = source.get(metric)
            if value is None:
                continue
            numerator = denominator = None
            if metric in _PERCENTAGE_METRICS:
                numerator_name, denominator_name = _PERCENTAGE_METRICS[metric]
                numerator = int(source[numerator_name])
                denominator = int(source[denominator_name])
            aggregate_claims.append(
                AggregateFinding(
                    claim_id=f"aggregate:{product_id}:{metric}",
                    product_id=product_id,
                    metric=metric,
                    value=value,
                    numerator=numerator,
                    denominator=denominator,
                )
            )

    review_findings: list[ReviewFinding] = []
    seen_review_ids: set[str] = set()
    for evidence_set in context.retrieved_evidence:
        for item in evidence_set.items:
            if item.review_id in seen_review_ids or not item.evidence_text:
                continue
            seen_review_ids.add(item.review_id)
            excerpt = item.evidence_text.strip()
            review_findings.append(
                ReviewFinding(
                    claim_id=f"review:{item.review_id}",
                    product_id=item.product_id,
                    criterion=evidence_set.criterion,
                    text=f"شاهد خام بازیابی‌شده برای معیار «{evidence_set.criterion}»: {excerpt}",
                    citations=[ReviewCitation(review_id=item.review_id, excerpt=excerpt)],
                )
            )
            break  # One auditable citation per product/criterion is sufficient for this compact baseline.

    authorization = context.authorization
    if authorization.overall_status == "inconclusive":
        recommendation = Recommendation(
            text="به دلیل نامشخص بودن نتیجهٔ قطعی در داده‌های تعیین‌کننده، توصیهٔ کلی ارائه نمی‌شود.",
            status="inconclusive",
        )
    elif authorization.overall_status == "weighted_winner":
        based_on = [
            criterion
            for criterion, status in authorization.criterion_statuses.items()
            if status in {"winner", "tie"}
        ]
        recommendation = Recommendation(
            text="این جمع‌بندی یک استنباط مشروط و فقط تابع اولویت‌های صریح اعلام‌شده است.",
            status="conditional",
            conditional_on=["اولویت‌های صریح کاربر"],
            based_on_criteria=based_on,
            criterion_winner_product_ids={
                criterion: authorization.criterion_winner_product_ids[criterion]
                for criterion in based_on
            },
            overall_winner_product_ids=authorization.overall_winner_product_ids,
        )
    else:
        recommendation = Recommendation(
            text="بدون اولویت صریح کاربر، برندهٔ کلی اعلام نمی‌شود.",
            status="not_authorized",
        )
    caveats = [
        Caveat(text=f"معیار «{decision.criterion}» قطعی نیست: {decision.reason_code}.")
        for decision in result.criterion_decisions
        if decision.status == "inconclusive"
    ]
    return GeneratedComparisonAnswer(
        direct_facts=direct_claims,
        aggregate_findings=aggregate_claims,
        review_findings=review_findings,
        recommendation=recommendation,
        caveats=caveats,
    )


def aggregate_grounding_results(results: list[GroundingValidationResult]) -> dict[str, Any]:
    """Pool numerator/denominator counts; never average per-answer ratios."""

    sums = Counter()
    for result in results:
        metrics = result.metrics
        for name in (
            "factual_claim_count",
            "grounded_claim_count",
            "unsupported_claim_count",
            "contradiction_count",
            "citation_count",
            "valid_citation_count",
            "support_requiring_claim_count",
            "evidence_covered_claim_count",
            "inconclusive_case_count",
            "correct_inconclusive_count",
        ):
            sums[name] += int(getattr(metrics, name))
    factual = sums["factual_claim_count"]
    return {
        "unit_of_analysis": "one structured claim; citations are additionally counted per cited review ID",
        "empty_answer_policy": "An answer with zero support-requiring claims contributes zero denominators; its ratios are null rather than treated as perfect.",
        "answer_count": len(results),
        **dict(sums),
        "grounded_claim_ratio": _ratio(sums["grounded_claim_count"], factual),
        "unsupported_claim_ratio": _ratio(sums["unsupported_claim_count"], factual),
        "citation_correctness": _ratio(sums["valid_citation_count"], sums["citation_count"]),
        "evidence_coverage": _ratio(sums["evidence_covered_claim_count"], sums["support_requiring_claim_count"]),
        "contradiction_rate": _ratio(sums["contradiction_count"], factual),
        "inconclusive_correctness": _ratio(sums["correct_inconclusive_count"], sums["inconclusive_case_count"]),
        "formulae": {
            "grounded_claim_ratio": "grounded_claim_count / factual_claim_count",
            "unsupported_claim_ratio": "unsupported_claim_count / factual_claim_count",
            "citation_correctness": "valid_citation_count / citation_count",
            "evidence_coverage": "evidence_covered_claim_count / support_requiring_claim_count",
            "contradiction_rate": "contradiction_count / factual_claim_count",
            "inconclusive_correctness": "correct_inconclusive_count / inconclusive_case_count",
        },
    }


def _human_rows(sample_predictions: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "case_id": prediction["case_id"],
            "scenario": prediction["scenario"],
            "prediction_artifact": "final_evaluation_predictions.json",
            "annotator_id": "",
            "status": "pending",
            **{column: "" for column in _HUMAN_SCORE_COLUMNS},
            "comments": "",
        }
        for prediction in sample_predictions
    ]


def write_human_annotation_template(
    path: Path, predictions: list[dict[str, Any]], *, sample_size: int
) -> Path:
    """Create, but never overwrite, a small independent human-evaluation form."""

    if path.is_file():
        return path
    eligible = [item for item in predictions if item.get("case_type") == "comparison" and item.get("final", {}).get("status") == "completed"]
    fields = ["case_id", "scenario", "prediction_artifact", "annotator_id", "status", *_HUMAN_SCORE_COLUMNS, "comments"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_human_rows(eligible[:sample_size]))
    return path


def aggregate_human_annotations(path: Path) -> dict[str, Any]:
    rubric = {
        "relevance_1_to_5": "Does the answer address the requested comparison criterion?",
        "clarity_1_to_5": "Is the Persian answer understandable and compact?",
        "source_separation_1_to_5": "Are facts, aggregate statistics, review evidence, and inference distinct?",
        "recommendation_usefulness_1_to_5": "Is any conditional recommendation useful without overstating certainty?",
        "uncertainty_appropriateness_1_to_5": "Does it abstain or qualify claims appropriately?",
        "citation_usefulness_1_to_5": "Are review IDs/excerpts useful for audit?",
        "semantic_citation_support_1_to_5": "Does the cited review actually support its attached claim?",
    }
    if not path.is_file():
        return {
            "status": "pending_human_annotation",
            "rubric_version": "human-answer-quality-v1",
            "rubric": rubric,
            "template_row_count": 0,
            "completed_row_count": 0,
            "valid_completed_row_count": 0,
            "annotation_path": str(path),
            "limitation": "The immutable template exists separately; copy it to the annotation path, complete it independently, and rerun the evaluator. No score is fabricated while annotations are absent.",
        }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows if (row.get("status") or "").strip().lower() == "completed"]
    scores: dict[str, list[float]] = {column: [] for column in _HUMAN_SCORE_COLUMNS}
    invalid_rows: list[str] = []
    for row in completed:
        try:
            values = {column: float(row[column]) for column in _HUMAN_SCORE_COLUMNS}
            if any(value < 1 or value > 5 for value in values.values()):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            invalid_rows.append(row.get("case_id", "unknown"))
            continue
        for column, value in values.items():
            scores[column].append(value)
    valid_count = len(completed) - len(invalid_rows)
    return {
        "status": "completed" if valid_count else "pending_human_annotation",
        "rubric_version": "human-answer-quality-v1",
        "rubric": rubric,
        "template_row_count": len(rows),
        "completed_row_count": len(completed),
        "valid_completed_row_count": valid_count,
        "invalid_completed_case_ids": invalid_rows,
        "mean_scores": {
            column: (sum(values) / len(values) if values else None)
            for column, values in scores.items()
        },
        "semantic_support_accuracy_proxy": (
            sum(score >= 4 for score in scores["semantic_citation_support_1_to_5"])
            / len(scores["semantic_citation_support_1_to_5"])
            if scores["semantic_citation_support_1_to_5"]
            else None
        ),
        "limitation": "Human fields remain pending until independent annotators complete the frozen CSV. No model or deterministic score is substituted for a human judgment.",
    }


def _run_resolver_case(case: FinalEvaluationCase, resolver: ProductResolver) -> tuple[dict[str, Any], dict[str, float]]:
    started = perf_counter()
    outcome = resolver.resolve(case.resolver_reference or "")
    latency = (perf_counter() - started) * 1000
    matches = outcome.status == case.expected_status
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "scenario": case.scenario,
        "question": case.question,
        "expected_status": case.expected_status,
        "observed_status": outcome.status,
        "matches_expected_status": matches,
        "result": outcome.model_dump(mode="json"),
        "baseline": {"status": "not_applicable"},
        "final": {"status": "completed", "generation_mode": "not_applicable"},
    }, {"product_resolution": latency, "statistics_lookup": 0.0, "retrieval": 0.0, "deterministic_comparison": 0.0, "generation": 0.0, "grounding_validation": 0.0, "total": latency}


def _run_comparison_case(
    case: FinalEvaluationCase,
    *,
    store: ComparisonDataStore,
    engine: ProductComparisonEngine,
    retriever: ProductionEvidenceRetriever,
    validator: DeterministicGroundingValidator,
    settings: Settings,
    with_llm: bool,
    generator: StructuredComparisonGenerator | None,
) -> tuple[dict[str, Any], dict[str, float], GroundingValidationResult | None, Any | None]:
    result, inputs, timings = _comparison_inputs_and_result(
        case,
        store=store,
        engine=engine,
        retriever=retriever,
        top_k=settings.final_evaluation.default_evidence_top_k,
    )
    context = build_generation_context(result, settings.generation, user_question=case.question)
    baseline_answer = deterministic_template_answer(result, context)
    baseline_rendered = render_persian_answer(baseline_answer)
    timings["generation"] = 0.0
    timings["grounding_validation"] = 0.0
    api = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "cache_hits": 0}
    final_answer = baseline_answer
    final_rendered = baseline_rendered
    final_mode = "deterministic_template_no_llm"
    generation_error: str | None = None
    if with_llm and generator is not None:
        started = perf_counter()
        try:
            outcome = generator.generate(result, user_question=case.question)
            final_answer = outcome.answer
            final_rendered = outcome.rendered_persian
            final_mode = "structured_llm"
            api = {
                "api_calls": 0 if outcome.metadata.cache_hit else 1,
                "input_tokens": outcome.metadata.input_tokens or 0,
                "output_tokens": outcome.metadata.output_tokens or 0,
                "estimated_cost_usd": outcome.metadata.estimated_cost_usd or 0.0,
                "cache_hits": int(outcome.metadata.cache_hit),
            }
        except Exception as error:  # The stored error contains type only, never provider text/secrets.
            generation_error = type(error).__name__
            final_mode = "structured_llm_failed"
        timings["generation"] = (perf_counter() - started) * 1000

    started = perf_counter()
    grounding = validator.validate(final_answer, context) if generation_error is None else None
    timings["grounding_validation"] = (perf_counter() - started) * 1000
    timings["total"] = sum(timings.values())
    normal_inconclusive = any(item.status == "inconclusive" for item in result.criterion_decisions)
    relaxed_engine = ProductComparisonEngine(
        replace(
            settings.comparison,
            practical_percentage_point_difference=0.0,
            practical_price_relative_difference=0.0,
            practical_product_rate_point_difference=0.0,
        )
    )
    requests = [CriterionRequest(name=name, evidence_query=case.evidence_queries.get(name)) for name in case.criteria]
    policy = PreferencePolicy(weights=case.preference_weights) if case.preference_weights else None
    relaxed = relaxed_engine.compare(inputs, requests, policy)
    relaxed_inconclusive = any(item.status == "inconclusive" for item in relaxed.criterion_decisions)
    invariants = {
        "all_evidence_has_requested_product_id": all(
            item.product_id == evidence_set.product_id
            for attachment in result.retrieved_evidence
            for evidence_set in attachment.evidence_sets
            for item in evidence_set.evidence_items
        ),
        "no_duplicate_evidence_review_ids_per_set": all(
            len({item.review_id for item in evidence_set.evidence_items}) == len(evidence_set.evidence_items)
            for attachment in result.retrieved_evidence
            for evidence_set in attachment.evidence_sets
        ),
        "full_statistics_are_structurally_separate_from_evidence": bool(result.aggregate_statistics) and all(
            not hasattr(evidence_set, "recommended_percentage")
            for attachment in result.retrieved_evidence
            for evidence_set in attachment.evidence_sets
        ),
    }
    prediction = {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "scenario": case.scenario,
        "question": case.question,
        "expects_inconclusive": case.expects_inconclusive,
        "observed_inconclusive": normal_inconclusive,
        "comparison": result.model_dump(mode="json"),
        "baseline": {
            "status": "completed",
            "retrieval_method": retriever.method,
            "generation_mode": "deterministic_template_unvalidated",
            "answer": baseline_answer.model_dump(mode="json"),
            "rendered_persian": baseline_rendered,
        },
        "final": {
            "status": "failed" if generation_error else "completed",
            "retrieval_method": retriever.method,
            "generation_mode": final_mode,
            "answer": None if generation_error else final_answer.model_dump(mode="json"),
            "rendered_persian": None if generation_error else final_rendered,
            "grounding_validation": None if grounding is None else grounding.model_dump(mode="json"),
            "generation_error_type": generation_error,
        },
        "api_usage": api,
        "invariants": invariants,
        "ablation": {
            "normal_inconclusive": normal_inconclusive,
            "zero_practical_threshold_inconclusive": relaxed_inconclusive,
            "forced_winner_introduced_by_zero_threshold": normal_inconclusive and not relaxed_inconclusive,
        },
    }
    return prediction, timings, grounding, final_answer


def _latency_summary(per_case: list[dict[str, Any]], *, resolver_build_ms: float | None) -> dict[str, Any]:
    components = ("product_resolution", "statistics_lookup", "retrieval", "deterministic_comparison", "generation", "grounding_validation", "total")
    values: dict[str, list[float]] = {}
    for component in components:
        if component == "total":
            values[component] = [
                float(item["latency_ms"].get(component, 0.0))
                for item in per_case
                if item.get("case_type") == "comparison"
            ]
        else:
            values[component] = [
                float(item["latency_ms"].get(component, 0.0))
                for item in per_case
                if float(item["latency_ms"].get(component, 0.0)) > 0
            ]
    return {
        "measurement": "wall-clock milliseconds in the final evaluator process; component distributions exclude non-applicable cases, and total is comparison end-to-end latency only. Retrieval includes product-scoped BM25 indexing when cold.",
        "resolver_index_build_ms": resolver_build_ms,
        "per_component_ms": {
            component: {"count": len(items), "p50": _percentile(items, 0.50), "p95": _percentile(items, 0.95), "p99": _percentile(items, 0.99)}
            for component, items in values.items()
        },
        "per_case": per_case,
        "peak_process_memory_bytes": peak_process_memory_bytes(),
    }


def _inconclusive_analysis(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [item for item in predictions if item.get("case_type") == "comparison" and item.get("expects_inconclusive")]
    observed = [item for item in predictions if item.get("case_type") == "comparison"]
    true_positive = sum(item.get("observed_inconclusive") for item in labelled)
    false_forced = sum(not item.get("observed_inconclusive") for item in labelled)
    conservative = sum(
        item.get("observed_inconclusive") and not item.get("expects_inconclusive")
        for item in observed
    )
    ablations = [item.get("ablation", {}) for item in observed]
    return {
        "label_policy": "Expected abstentions are deterministic policy probes selected from sparse/no-review, practical-near-tie, or conflicted-metadata inputs; they are not claims of human preference ground truth.",
        "expected_inconclusive_case_count": len(labelled),
        "correct_inconclusive_decisions": true_positive,
        "false_forced_winners": false_forced,
        "overly_conservative_inconclusive_decisions": conservative,
        "policy_probe_inconclusive_correctness": _ratio(true_positive, len(labelled)),
        "zero_threshold_ablation": {
            "cases": len(ablations),
            "normal_inconclusive_cases": sum(item.get("normal_inconclusive", False) for item in ablations),
            "zero_threshold_inconclusive_cases": sum(item.get("zero_practical_threshold_inconclusive", False) for item in ablations),
            "forced_winners_introduced": sum(item.get("forced_winner_introduced_by_zero_threshold", False) for item in ablations),
        },
    }


def _validator_probe(
    answer: GeneratedComparisonAnswer | None,
    context: GenerationContext | None,
    validator: DeterministicGroundingValidator,
) -> dict[str, Any]:
    """A small controlled ablation: invalid numeric output must not pass."""

    if answer is None or context is None or not answer.direct_facts:
        return {"status": "not_run", "reason": "no direct fact available in completed final answer"}
    mutated = answer.model_copy(deep=True)
    claim = mutated.direct_facts[0]
    if isinstance(claim.value, bool):
        claim.value = not claim.value
    elif isinstance(claim.value, int):
        claim.value += 1
    elif isinstance(claim.value, float):
        claim.value += 0.01
    else:
        claim.value = f"{claim.value}-invented"
    result = validator.validate(mutated, context)
    return {
        "status": "completed",
        "mutation": "one deterministic direct-fact value changed after generation",
        "validator_rejected_mutation": not result.valid,
        "reason_codes": [item.reason_code for item in result.claim_results if item.status != "grounded"],
        "limitation": "This controlled probe establishes numeric-claim detection, not semantic entailment quality for arbitrary natural-language claims.",
    }


def _failure_analysis(
    *,
    settings: Settings,
    predictions: list[dict[str, Any]],
    human: dict[str, Any],
    validator_probe: dict[str, Any],
) -> dict[str, Any]:
    retrieval = _load_json(settings.paths.reranker_evaluation_report)
    entries: list[dict[str, Any]] = []
    for method, result in retrieval.get("methods", {}).items():
        if result.get("status") == "unavailable":
            entries.append(
                {
                    "failure_class": "dense_or_reranker_resource_unavailable",
                    "severity": "high",
                    "input": method,
                    "observed_output": result.get("reason"),
                    "expected_behavior": "The required method is reported unavailable; it is not replaced or scored on a different corpus.",
                    "likely_root_cause": "Pinned BGE model/resource prerequisite is not available on the frozen host.",
                    "proposed_fix": "Run the same pinned model and frozen benchmark on a host meeting the recorded RAM/accelerator requirements, then regenerate a new versioned benchmark.",
                    "fix_attempted": False,
                }
            )
    for prediction in predictions:
        if prediction.get("case_type") == "resolver" and not prediction.get("matches_expected_status"):
            entries.append(
                {
                    "failure_class": "product_resolution",
                    "severity": "high",
                    "input": prediction.get("question"),
                    "observed_output": prediction.get("observed_status"),
                    "expected_behavior": prediction.get("expected_status"),
                    "likely_root_cause": "Frozen resolver case did not honour ambiguity or model-token protection.",
                    "proposed_fix": "Inspect candidate tokens and thresholds; do not auto-select a product until the case is corrected.",
                    "fix_attempted": False,
                }
            )
        if prediction.get("case_type") == "comparison":
            comparison = prediction.get("comparison", {})
            for warning in comparison.get("warnings", []):
                entries.append(
                    {
                        "failure_class": warning.get("code", "comparison_warning").lower(),
                        "severity": "medium",
                        "input": prediction.get("case_id"),
                        "observed_output": warning.get("message"),
                        "expected_behavior": "Surface the limitation and avoid an unsupported winner.",
                        "likely_root_cause": "Observed property of pinned product metadata or retrieved evidence.",
                        "proposed_fix": "Require a cleaner source snapshot or collect additional independent evidence; do not silently normalize the conflict away.",
                        "fix_attempted": False,
                    }
                )
    if human.get("status") != "completed":
        entries.append(
            {
                "failure_class": "human_answer_quality_not_yet_annotated",
                "severity": "high",
                "input": HUMAN_TEMPLATE_NAME,
                "observed_output": human.get("status"),
                "expected_behavior": "Independent annotators complete the frozen rubric before claims about answer usefulness or semantic citation support are made.",
                "likely_root_cause": "No human annotations were supplied to the local workspace.",
                "proposed_fix": "Assign at least two independent Persian-speaking annotators, record disagreements, and rerun the final evaluator.",
                "fix_attempted": False,
            }
        )
    entries.append(
        {
            "failure_class": "grounding_validator_ablation",
            "severity": "informational",
            "input": "controlled mutated direct fact",
            "observed_output": validator_probe,
            "expected_behavior": "An altered numeric fact is rejected deterministically.",
            "likely_root_cause": "Not a production failure; controlled validation probe.",
            "proposed_fix": "Keep the validator enabled and expand independently judged semantic-support annotations.",
            "fix_attempted": True,
        }
    )
    return {
        "schema_version": "failure-taxonomy-v1",
        "entries": entries,
        "known_limitations": [
            "Frozen retrieval qrels are deterministic lexical seed labels pending human relevance review.",
            "Lexical overlap in the grounding validator is a conservative signal, not proof of review-claim entailment.",
            "No LLM generation metrics are claimed unless --with-llm is explicitly run with a configured API credential.",
        ],
    }


def _retrieval_metrics(settings: Settings) -> dict[str, Any]:
    report = _load_json(settings.paths.reranker_evaluation_report)
    return {
        "source_artifact": str(settings.paths.reranker_evaluation_report),
        "source_schema_version": report.get("schema_version"),
        "selected_production_retriever": report.get("production_selection", {}).get("selected_method"),
        "methods": report.get("methods", {}),
        "label_limitations": report.get("label_limitations"),
        "metric_note": report.get("metric_note"),
    }


def _end_to_end_metrics(predictions: list[dict[str, Any]], *, with_llm: bool) -> dict[str, Any]:
    comparison_predictions = [item for item in predictions if item.get("case_type") == "comparison"]
    final_completed = [item for item in comparison_predictions if item.get("final", {}).get("status") == "completed"]
    invariant_failures = [
        item["case_id"]
        for item in final_completed
        if not all(item.get("invariants", {}).values())
    ]
    baseline_modes = Counter(item.get("baseline", {}).get("generation_mode") for item in comparison_predictions)
    final_modes = Counter(item.get("final", {}).get("generation_mode") for item in comparison_predictions)
    return {
        "unit_of_analysis": "one frozen evaluation case; answer-level claims are evaluated separately in grounding_metrics.json",
        "comparison_case_count": len(comparison_predictions),
        "resolver_case_count": len(predictions) - len(comparison_predictions),
        "baseline": {
            "description": "pre-resolved products + deterministic full statistics + frozen BM25 evidence + compact template answer; no post-generation grounding validation",
            "completed_case_count": len(comparison_predictions),
            "generation_modes": dict(baseline_modes),
        },
        "final": {
            "description": "frozen selected retrieval + deterministic comparison + structured answer + Phase 11 grounding validation",
            "completed_case_count": len(final_completed),
            "generation_modes": dict(final_modes),
            "all_structural_invariants_passed": not invariant_failures,
            "invariant_failure_case_ids": invariant_failures,
            "llm_execution_requested": with_llm,
        },
        "comparison_interpretation": "In no-LLM mode both systems use the same deterministic template to isolate the validator and pipeline boundaries; this run does not claim a natural-language quality gain over an LLM baseline.",
    }


def _cost_metrics(predictions: list[dict[str, Any]], *, with_llm: bool) -> dict[str, Any]:
    sums = Counter()
    for prediction in predictions:
        for key, value in prediction.get("api_usage", {}).items():
            sums[key] += value or 0
    comparison_count = sum(item.get("case_type") == "comparison" for item in predictions)
    return {
        "llm_execution_requested": with_llm,
        "api_calls": sums["api_calls"],
        "input_tokens": sums["input_tokens"],
        "output_tokens": sums["output_tokens"],
        "total_estimated_cost_usd": sums["estimated_cost_usd"],
        "cache_hits": sums["cache_hits"],
        "cache_hit_rate": _ratio(sums["cache_hits"], sums["api_calls"] + sums["cache_hits"]),
        "average_cost_per_comparison_usd": _ratio(sums["estimated_cost_usd"], comparison_count),
        "interpretation": "Zero API calls/tokens in no-LLM mode are an intentional cost-control setting, not a measured LLM quality result.",
    }


def _markdown_report(
    *,
    manifest: dict[str, Any],
    cases: list[FinalEvaluationCase],
    retrieval: dict[str, Any],
    grounding: dict[str, Any],
    human: dict[str, Any],
    inconclusive: dict[str, Any],
    latency: dict[str, Any],
    cost: dict[str, Any],
    failures: dict[str, Any],
) -> str:
    methods = retrieval.get("methods", {})
    table = ["| روش | وضعیت | Recall@K | Precision@K | MRR | NDCG@K | warm p50/p95 | Storage | Peak RAM |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("bm25", "bge_m3_dense", "hybrid_rrf", "hybrid_bge_reranker"):
        item = methods.get(name, {})
        scores = item.get("metrics_at_k") or {}
        latency_item = item.get("latency_ms") or {}
        def value(name: str) -> str:
            raw = scores.get(name)
            return "-" if raw is None else f"{float(raw):.4f}"
        warm = "-" if not latency_item else f"{float(latency_item.get('warm_p50', 0)):.3f}/{float(latency_item.get('warm_p95', 0)):.3f} ms"
        storage_bytes = item.get("storage_bytes")
        memory_bytes = item.get("peak_process_memory_bytes")
        storage = "-" if storage_bytes is None else f"{float(storage_bytes) / 1024**2:.1f} MiB"
        memory = "-" if memory_bytes is None else f"{float(memory_bytes) / 1024**2:.1f} MiB"
        table.append(f"| {name} | {item.get('status', 'missing')} | {value('recall')} | {value('precision')} | {value('mrr')} | {value('ndcg')} | {warm} | {storage} | {memory} |")
    component_lines = []
    for name, values in latency["per_component_ms"].items():
        component_lines.append(f"- {name}: p50={values['p50']}, p95={values['p95']} ms")
    scenario_counts = Counter(case.scenario for case in cases)
    return "\n".join(
        [
            "# Final evaluation report",
            "",
            "## Frozen system",
            "",
            f"- Dataset revision: `{manifest['dataset']['revision']}`",
            f"- Production retriever: `{manifest['retrieval']['selected_production_retriever']}`",
            f"- Generation model/config: `{manifest['generation']['provider']}` / `{manifest['generation']['model']}`; prompt `{manifest['generation']['prompt_version']}`.",
            f"- Grounding validator: `{manifest['grounding']['validator_version']}` with `{manifest['grounding']['unsupported_claim_action']}` policy.",
            "",
            "## Evaluation set",
            "",
            f"- {len(cases)} frozen Persian cases: {sum(case.case_type == 'comparison' for case in cases)} comparisons and {sum(case.case_type == 'resolver' for case in cases)} resolver cases.",
            f"- Scenarios: {', '.join(sorted(scenario_counts))}.",
            "",
            "## Frozen retrieval benchmark",
            "",
            *table,
            "",
            f"Label limitation: {retrieval.get('label_limitations')}",
            "",
            "## Grounding",
            "",
            f"- grounded claim ratio: {grounding.get('grounded_claim_ratio')}",
            f"- unsupported claim ratio: {grounding.get('unsupported_claim_ratio')}",
            f"- citation correctness: {grounding.get('citation_correctness')}",
            f"- evidence coverage: {grounding.get('evidence_coverage')}",
            f"- contradiction rate: {grounding.get('contradiction_rate')}",
            f"- inconclusive correctness: {grounding.get('inconclusive_correctness')}",
            "",
            "## Human answer quality",
            "",
            f"- Status: `{human.get('status')}`; valid completed forms: {human.get('valid_completed_row_count', 0)}.",
            "- The CSV rubric is intentionally not auto-filled: no LLM or deterministic proxy is represented as human judgment.",
            "",
            "## Inconclusive policy",
            "",
            f"- Correct policy-probe abstentions: {inconclusive['correct_inconclusive_decisions']}/{inconclusive['expected_inconclusive_case_count']}; false forced winners: {inconclusive['false_forced_winners']}; overly conservative cases: {inconclusive['overly_conservative_inconclusive_decisions']}.",
            f"- Removing practical thresholds introduced {inconclusive['zero_threshold_ablation']['forced_winners_introduced']} forced winners in this frozen set.",
            "",
            "## Performance and cost",
            "",
            *component_lines,
            f"- Peak evaluator process memory: {latency.get('peak_process_memory_bytes')} bytes.",
            f"- API calls/tokens/cost: {cost['api_calls']} / {cost['input_tokens']} input + {cost['output_tokens']} output / ${cost['total_estimated_cost_usd']:.6f}.",
            "",
            "## Failure analysis and limitations",
            "",
            f"- {len(failures['entries'])} structured entries are in `failure_analysis.json`.",
            *[f"- {item}" for item in failures["known_limitations"]],
            "",
            "## Reproduction",
            "",
            "```powershell",
            "digikala-evaluate-final --config config/default.toml --no-llm",
            "# Optional and cost-bearing, with OPENAI_API_KEY configured:",
            "digikala-evaluate-final --config config/default.toml --with-llm --max-llm-cases 5",
            "```",
        ]
    ) + "\n"


def run_final_evaluation(
    settings: Settings,
    *,
    output_root: Path | None = None,
    with_llm: bool = False,
    max_llm_cases: int | None = None,
    rebuild_evaluation_set: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the frozen final workflow and persist only versioned Phase 12 artifacts."""

    root = output_root or settings.paths.final_evaluation_root
    if root is None:
        raise ValueError("final_evaluation_root must be configured")
    root.mkdir(parents=True, exist_ok=True)
    cases, case_path, case_hash = load_or_create_evaluation_cases(settings, root, rebuild=rebuild_evaluation_set)
    manifest = build_final_system_manifest(settings, case_set_sha256=case_hash)
    manifest_path = _write_json(root / "final_system_manifest.json", manifest)
    if dry_run:
        plan = {
            "status": "dry_run",
            "case_count": len(cases),
            "comparison_case_count": sum(case.case_type == "comparison" for case in cases),
            "resolver_case_count": sum(case.case_type == "resolver" for case in cases),
            "with_llm": with_llm,
            "artifact_root": str(root),
            "case_set": str(case_path),
            "manifest": str(manifest_path),
        }
        _write_json(root / "dry_run_plan.json", plan)
        return plan
    comparison_cases = [case for case in cases if case.case_type == "comparison"]
    allowed_llm_cases = max_llm_cases if max_llm_cases is not None else settings.final_evaluation.max_llm_cases
    if with_llm and len(comparison_cases) > allowed_llm_cases:
        raise ValueError(
            f"--with-llm would run {len(comparison_cases)} comparison cases, above the explicit max_llm_cases={allowed_llm_cases}; reduce the frozen set or raise the cap deliberately"
        )

    store = ComparisonDataStore.from_settings(settings)
    engine = ProductComparisonEngine(settings.comparison)
    retriever = ProductionEvidenceRetriever.from_settings(settings)
    validator = DeterministicGroundingValidator.from_settings(settings)
    generator = StructuredComparisonGenerator.from_settings(settings) if with_llm else None
    resolver: ProductResolver | None = None
    resolver_build_ms: float | None = None
    predictions: list[dict[str, Any]] = []
    latency_records: list[dict[str, Any]] = []
    grounding_results: list[GroundingValidationResult] = []
    first_context: GenerationContext | None = None
    first_answer: GeneratedComparisonAnswer | None = None
    for case in cases:
        try:
            if case.case_type == "resolver":
                if resolver is None:
                    started = perf_counter()
                    if settings.paths.canonical_products is None:
                        raise ValueError("canonical_products must be configured for resolver evaluation")
                    resolver = ProductResolver.from_parquet(str(settings.paths.canonical_products), settings.resolution)
                    resolver_build_ms = (perf_counter() - started) * 1000
                prediction, timings = _run_resolver_case(case, resolver)
                grounding = None
            else:
                prediction, timings, grounding, final_answer = _run_comparison_case(
                    case,
                    store=store,
                    engine=engine,
                    retriever=retriever,
                    validator=validator,
                    settings=settings,
                    with_llm=with_llm,
                    generator=generator,
                )
                if grounding is not None:
                    grounding_results.append(grounding)
                if first_context is None and final_answer is not None:
                    # Rebuild the exact bounded context from serialized comparison once only for the probe.
                    # It is kept local to avoid mixing Top-K evidence with aggregate statistics.
                    from .comparison import ComparisonResult

                    parsed_result = ComparisonResult.model_validate(prediction["comparison"])
                    first_context = build_generation_context(parsed_result, settings.generation, user_question=case.question)
                    first_answer = final_answer
        except Exception as error:
            prediction = {
                "case_id": case.case_id,
                "case_type": case.case_type,
                "scenario": case.scenario,
                "question": case.question,
                "expected_status": case.expected_status,
                "final": {"status": "failed", "error_type": type(error).__name__},
                "baseline": {"status": "not_run"},
                "api_usage": {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "cache_hits": 0},
            }
            timings = {"product_resolution": 0.0, "statistics_lookup": 0.0, "retrieval": 0.0, "deterministic_comparison": 0.0, "generation": 0.0, "grounding_validation": 0.0, "total": 0.0}
        predictions.append(prediction)
        latency_records.append({"case_id": case.case_id, "case_type": case.case_type, "scenario": case.scenario, "latency_ms": timings})

    grounding = aggregate_grounding_results(grounding_results)
    human_template = write_human_annotation_template(root / HUMAN_TEMPLATE_NAME, predictions, sample_size=settings.final_evaluation.human_sample_size)
    human = aggregate_human_annotations(root / HUMAN_ANNOTATIONS_NAME)
    inconclusive = _inconclusive_analysis(predictions)
    validator_probe = _validator_probe(first_answer, first_context, validator)
    failures = _failure_analysis(settings=settings, predictions=predictions, human=human, validator_probe=validator_probe)
    retrieval = _retrieval_metrics(settings)
    latency = _latency_summary(latency_records, resolver_build_ms=resolver_build_ms)
    cost = _cost_metrics(predictions, with_llm=with_llm)
    end_to_end = _end_to_end_metrics(predictions, with_llm=with_llm)
    demos = {
        "selection_policy": "Real final-evaluation outputs only; no canned answers.",
        "scenarios": [
            item
            for item in predictions
            if item.get("scenario") in {
                "clear_direct_price_winner",
                "review_evidence_with_ids",
                "conflicting_review_population",
                "sparse_review_product",
                "ambiguous_product_name",
                "multi_product_direct_data",
            }
        ],
    }
    paths = {
        "manifest": manifest_path,
        "case_set": case_path,
        "retrieval_metrics": _write_json(root / "retrieval_metrics.json", retrieval),
        "grounding_metrics": _write_json(root / "grounding_metrics.json", grounding),
        "end_to_end_metrics": _write_json(root / "end_to_end_metrics.json", end_to_end),
        "latency_metrics": _write_json(root / "latency_metrics.json", latency),
        "cost_metrics": _write_json(root / "cost_metrics.json", cost),
        "inconclusive_analysis": _write_json(root / "inconclusive_analysis.json", inconclusive),
        "failure_analysis": _write_json(root / "failure_analysis.json", failures),
        "predictions": _write_json(root / "final_evaluation_predictions.json", {"schema_version": FINAL_SCHEMA_VERSION, "predictions": predictions}),
        "demo_scenarios": _write_json(root / "demo_scenarios.json", demos),
        "human_template": human_template,
    }
    report = _markdown_report(
        manifest=manifest,
        cases=cases,
        retrieval=retrieval,
        grounding=grounding,
        human=human,
        inconclusive=inconclusive,
        latency=latency,
        cost=cost,
        failures=failures,
    )
    paths["report"] = root / "final_evaluation_report.md"
    paths["report"].write_text(report, encoding="utf-8")
    return {
        "status": "completed_with_pending_human_annotation" if human.get("status") != "completed" else "completed",
        "output_root": str(root),
        "paths": {name: str(path) for name, path in paths.items()},
        "case_count": len(cases),
        "grounding": grounding,
        "human": human,
        "cost": cost,
        "selected_production_retriever": retrieval.get("selected_production_retriever"),
    }
