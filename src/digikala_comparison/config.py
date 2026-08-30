"""Typed, file-based settings for the Phase 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DatasetSettings:
    revision: str
    source_url: str
    repository: str = ""


@dataclass(frozen=True)
class PathSettings:
    raw_products: Path
    raw_comments: Path
    processed_products: Path
    processed_comments: Path
    quality_report: Path
    product_statistics: Path | None = None
    statistics_report: Path | None = None
    canonical_products: Path | None = None
    duplicate_conflict_report: Path | None = None
    resolution_validation_report: Path | None = None
    retrieval_corpus: Path | None = None
    retrieval_corpus_report: Path | None = None
    retrieval_queries: Path | None = None
    retrieval_qrels: Path | None = None
    retrieval_results: Path | None = None
    retrieval_evaluation_report: Path | None = None
    dense_sorted_corpus: Path | None = None
    dense_index_root: Path | None = None
    dense_manifest: Path | None = None
    dense_product_ranges: Path | None = None
    dense_ranked_results: Path | None = None
    dense_evaluation_report: Path | None = None
    dense_comparison_report: Path | None = None
    dense_failure_analysis: Path | None = None
    dense_resource_report: Path | None = None
    hybrid_splits: Path | None = None
    hybrid_tuning_report: Path | None = None
    hybrid_ranked_results: Path | None = None
    hybrid_evaluation_report: Path | None = None
    hybrid_analysis_report: Path | None = None
    hybrid_selection_report: Path | None = None
    reranker_cache_root: Path | None = None
    reranker_resource_report: Path | None = None
    reranker_tuning_report: Path | None = None
    reranker_ranked_results: Path | None = None
    reranker_evaluation_report: Path | None = None
    reranker_analysis_report: Path | None = None
    reranker_failure_analysis: Path | None = None
    reranker_selection_report: Path | None = None
    frozen_development_queries: Path | None = None
    frozen_development_qrels: Path | None = None
    frozen_test_queries: Path | None = None
    frozen_test_qrels: Path | None = None
    retrieval_experiment_manifest: Path | None = None
    retrieval_benchmark_markdown: Path | None = None
    generation_cache_root: Path | None = None
    generation_trace_root: Path | None = None
    generation_output_root: Path | None = None
    grounding_audit_root: Path | None = None
    final_evaluation_root: Path | None = None
    final_evaluation_v2_root: Path | None = None


@dataclass(frozen=True)
class ResolutionSettings:
    fuzzy_score_threshold: float = 90.0
    ambiguity_score_margin: float = 3.0
    max_fuzzy_candidates: int = 50


@dataclass(frozen=True)
class BM25Settings:
    k1: float = 1.2
    b: float = 0.75
    default_top_k: int = 10
    candidate_depth: int = 100
    minimum_normalized_text_length: int = 3


@dataclass(frozen=True)
class DenseSettings:
    model_id: str = "BAAI/bge-m3"
    model_revision: str = ""
    backend: str = "faiss.IndexFlatIP"
    device: str = "auto"
    batch_size: int = 4
    checkpoint_documents: int = 2048
    max_length: int = 512
    embedding_dimension: int = 1024
    normalize_embeddings: bool = True
    use_fp16: bool = False
    product_index_cache_size: int = 32
    pilot_documents: int = 8


@dataclass(frozen=True)
class HybridSettings:
    bm25_candidate_depth: int = 100
    dense_candidate_depth: int = 100
    rrf_k: int = 60
    final_top_k: int = 10
    tuning_rrf_k: tuple[int, ...] = (20, 40, 60, 80)
    minimum_ndcg_gain: float = 0.02
    maximum_warm_p95_multiplier: float = 2.0


@dataclass(frozen=True)
class RerankerSettings:
    """Pinned cross-encoder settings for Phase 7 only."""

    model_id: str = "BAAI/bge-reranker-v2-m3"
    model_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    device: str = "auto"
    batch_size: int = 4
    max_length: int = 512
    query_max_length: int = 64
    use_fp16: bool = False
    candidate_depths: tuple[int, ...] = (20, 50, 100)
    candidate_depth: int = 50
    final_top_k: int = 10
    # The pinned safetensors file is 2,271,071,852 bytes. The headroom is a
    # conservative preflight guard against CPU OOM during Transformers load.
    model_weights_bytes: int = 2_271_071_852
    minimum_available_ram_bytes: int = 6 * 1024**3
    bootstrap_iterations: int = 1_000
    minimum_ndcg_gain: float = 0.02
    maximum_warm_p95_ms: float = 2_000.0
    long_text_character_threshold: int = 500


@dataclass(frozen=True)
class ComparisonSettings:
    """Predeclared practical thresholds for deterministic product comparison."""

    minimum_percentage_denominator: int = 20
    practical_percentage_point_difference: float = 0.05
    minimum_product_rate_count: int = 10
    practical_product_rate_point_difference: float = 2.0
    practical_price_relative_difference: float = 0.05
    minimum_retrieved_evidence_items: int = 3
    require_stable_field_metadata: bool = True


@dataclass(frozen=True)
class GenerationSettings:
    """Cost-aware, bounded Phase 10 structured-generation settings."""

    provider: str = "openai_responses"
    model: str = "gpt-5.6-luna"
    api_key_environment_variable: str = "OPENAI_API_KEY"
    api_base_url: str | None = None
    cost_estimation_available: bool = True
    temperature: float = 0.0
    max_output_tokens: int = 1200
    timeout_seconds: float = 30.0
    input_token_cost_per_million_usd: float = 0.20
    output_token_cost_per_million_usd: float = 1.20
    prompt_version: str = "comparison-generation-v1"
    schema_version: str = "generated-comparison-answer-v1"
    max_evidence_items_per_set: int = 3
    max_evidence_characters_per_item: int = 400
    max_total_evidence_characters: int = 6000
    max_user_text_characters: int = 1000
    enable_cache: bool = True
    persist_development_traces: bool = False

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("generation provider must be non-empty")
        if not self.model.strip() and self.provider != "metis_openai_compatible":
            raise ValueError("generation model must be non-empty")
        if self.provider == "metis_openai_compatible" and not (self.api_base_url or "").strip():
            raise ValueError("Metis generation requires api_base_url")
        if self.api_base_url is not None and not self.api_base_url.startswith(("https://", "http://")):
            raise ValueError("generation api_base_url must be an HTTP(S) URL")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("generation temperature must be between 0 and 2")
        if self.max_output_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("generation output token limit and timeout must be positive")
        if self.input_token_cost_per_million_usd < 0 or self.output_token_cost_per_million_usd < 0:
            raise ValueError("generation token costs cannot be negative")
        if min(
            self.max_evidence_items_per_set,
            self.max_evidence_characters_per_item,
            self.max_total_evidence_characters,
            self.max_user_text_characters,
        ) <= 0:
            raise ValueError("generation context-budget values must be positive")


@dataclass(frozen=True)
class GroundingSettings:
    """Deterministic Phase 11 grounding-validation policy and tolerances."""

    validator_version: str = "grounding-validator-v1"
    unsupported_claim_action: str = "reject"
    numeric_absolute_tolerance: float = 0.0005
    minimum_review_lexical_overlap_tokens: int = 1
    max_regeneration_attempts: int = 1

    def __post_init__(self) -> None:
        if self.unsupported_claim_action not in {
            "reject",
            "remove_unsupported",
            "rewrite_regenerate",
        }:
            raise ValueError("unsupported_claim_action must be reject, remove_unsupported, or rewrite_regenerate")
        if self.numeric_absolute_tolerance < 0:
            raise ValueError("numeric_absolute_tolerance cannot be negative")
        if self.minimum_review_lexical_overlap_tokens < 0:
            raise ValueError("minimum_review_lexical_overlap_tokens cannot be negative")
        if self.max_regeneration_attempts < 0:
            raise ValueError("max_regeneration_attempts cannot be negative")


@dataclass(frozen=True)
class FinalEvaluationSettings:
    """Explicit, cost-safe controls for the Phase 12 final evaluation."""

    evaluation_version: str = "final-evaluation-v1"
    case_set_version: str = "persian-comparison-cases-v1"
    human_rubric_version: str = "human-answer-quality-v1"
    default_evidence_top_k: int = 3
    human_sample_size: int = 5
    max_llm_cases: int = 5

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.evaluation_version,
                self.case_set_version,
                self.human_rubric_version,
            )
        ):
            raise ValueError("final evaluation version identifiers must be non-empty")
        if min(self.default_evidence_top_k, self.human_sample_size, self.max_llm_cases) <= 0:
            raise ValueError("final evaluation counts must be positive")


@dataclass(frozen=True)
class FinalEvaluationV2Settings:
    """Version identifiers for the staged final-system evaluation v2."""

    evaluation_version: str = "final-system-evaluation-v2"
    case_schema_version: str = "final-evaluation-case-v2"
    manifest_schema_version: str = "final-evaluation-manifest-v2"
    human_rubric_version: str = "human-answer-quality-v2"

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.evaluation_version,
                self.case_schema_version,
                self.manifest_schema_version,
                self.human_rubric_version,
            )
        ):
            raise ValueError("final evaluation v2 version identifiers must be non-empty")


@dataclass(frozen=True)
class NormalizationSettings:
    unicode_form: str
    replace_arabic_yeh: bool
    replace_arabic_kaf: bool
    collapse_whitespace: bool
    treat_string_nan_as_null: bool


@dataclass(frozen=True)
class ReviewEligibilitySettings:
    require_nonempty_normalized_text: bool
    minimum_normalized_text_length: int
    require_buyer: bool
    allowed_recommendation_status: tuple[str, ...]
    minimum_helpfulness_votes: int = 1


@dataclass(frozen=True)
class Settings:
    dataset: DatasetSettings
    paths: PathSettings
    random_seed: int
    normalization: NormalizationSettings
    review_eligibility: ReviewEligibilitySettings
    download_chunk_bytes: int = 1048576
    resolution: ResolutionSettings = field(default_factory=ResolutionSettings)
    bm25: BM25Settings = field(default_factory=BM25Settings)
    dense: DenseSettings = field(default_factory=DenseSettings)
    hybrid: HybridSettings = field(default_factory=HybridSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    comparison: ComparisonSettings = field(default_factory=ComparisonSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    grounding: GroundingSettings = field(default_factory=GroundingSettings)
    final_evaluation: FinalEvaluationSettings = field(default_factory=FinalEvaluationSettings)
    final_evaluation_v2: FinalEvaluationV2Settings = field(default_factory=FinalEvaluationV2Settings)

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "Settings":
        path = Path(config_path).resolve()
        with path.open("rb") as handle:
            raw = tomllib.load(handle)

        # Paths in the project configuration are repository-relative. Locating
        # the root by pyproject.toml also keeps an alternate config file usable.
        base_dir = next(
            (
                candidate
                for candidate in (path.parent, *path.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            path.parent,
        )

        def resolve_path(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else (base_dir / candidate)

        paths = raw["paths"]
        normalization = raw["normalization"]
        eligibility = raw["review_eligibility"]
        return cls(
            dataset=DatasetSettings(**raw["dataset"]),
            paths=PathSettings(
                raw_products=resolve_path(paths["raw_products"]),
                raw_comments=resolve_path(paths["raw_comments"]),
                processed_products=resolve_path(paths["processed_products"]),
                processed_comments=resolve_path(paths["processed_comments"]),
                quality_report=resolve_path(paths["quality_report"]),
                product_statistics=resolve_path(paths["product_statistics"]),
                statistics_report=resolve_path(paths["statistics_report"]),
                canonical_products=resolve_path(paths["canonical_products"]),
                duplicate_conflict_report=resolve_path(paths["duplicate_conflict_report"]),
                resolution_validation_report=resolve_path(
                    paths["resolution_validation_report"]
                ),
                retrieval_corpus=resolve_path(paths["retrieval_corpus"]),
                retrieval_corpus_report=resolve_path(paths["retrieval_corpus_report"]),
                retrieval_queries=resolve_path(paths["retrieval_queries"]),
                retrieval_qrels=resolve_path(paths["retrieval_qrels"]),
                retrieval_results=resolve_path(paths["retrieval_results"]),
                retrieval_evaluation_report=resolve_path(
                    paths["retrieval_evaluation_report"]
                ),
                dense_sorted_corpus=resolve_path(paths["dense_sorted_corpus"]),
                dense_index_root=resolve_path(paths["dense_index_root"]),
                dense_manifest=resolve_path(paths["dense_manifest"]),
                dense_product_ranges=resolve_path(paths["dense_product_ranges"]),
                dense_ranked_results=resolve_path(paths["dense_ranked_results"]),
                dense_evaluation_report=resolve_path(paths["dense_evaluation_report"]),
                dense_comparison_report=resolve_path(paths["dense_comparison_report"]),
                dense_failure_analysis=resolve_path(paths["dense_failure_analysis"]),
                dense_resource_report=resolve_path(paths["dense_resource_report"]),
                hybrid_splits=resolve_path(paths["hybrid_splits"]),
                hybrid_tuning_report=resolve_path(paths["hybrid_tuning_report"]),
                hybrid_ranked_results=resolve_path(paths["hybrid_ranked_results"]),
                hybrid_evaluation_report=resolve_path(paths["hybrid_evaluation_report"]),
                hybrid_analysis_report=resolve_path(paths["hybrid_analysis_report"]),
                hybrid_selection_report=resolve_path(paths["hybrid_selection_report"]),
                reranker_cache_root=resolve_path(paths["reranker_cache_root"]),
                reranker_resource_report=resolve_path(paths["reranker_resource_report"]),
                reranker_tuning_report=resolve_path(paths["reranker_tuning_report"]),
                reranker_ranked_results=resolve_path(paths["reranker_ranked_results"]),
                reranker_evaluation_report=resolve_path(paths["reranker_evaluation_report"]),
                reranker_analysis_report=resolve_path(paths["reranker_analysis_report"]),
                reranker_failure_analysis=resolve_path(paths["reranker_failure_analysis"]),
                reranker_selection_report=resolve_path(paths["reranker_selection_report"]),
                frozen_development_queries=resolve_path(paths["frozen_development_queries"]),
                frozen_development_qrels=resolve_path(paths["frozen_development_qrels"]),
                frozen_test_queries=resolve_path(paths["frozen_test_queries"]),
                frozen_test_qrels=resolve_path(paths["frozen_test_qrels"]),
                retrieval_experiment_manifest=resolve_path(paths["retrieval_experiment_manifest"]),
                retrieval_benchmark_markdown=resolve_path(paths["retrieval_benchmark_markdown"]),
                generation_cache_root=resolve_path(paths.get("generation_cache_root", "data/generation/cache")),
                generation_trace_root=resolve_path(paths.get("generation_trace_root", "data/generation/traces")),
                generation_output_root=resolve_path(paths.get("generation_output_root", "data/generation/outputs")),
                grounding_audit_root=resolve_path(paths.get("grounding_audit_root", "data/generation/grounding_audits")),
                final_evaluation_root=resolve_path(paths.get("final_evaluation_root", "data/evaluations/final_v1")),
                final_evaluation_v2_root=resolve_path(paths.get("final_evaluation_v2_root", "data/evaluations/final_v2")),
            ),
            random_seed=int(raw["runtime"]["random_seed"]),
            normalization=NormalizationSettings(**normalization),
            review_eligibility=ReviewEligibilitySettings(
                require_nonempty_normalized_text=bool(
                    eligibility["require_nonempty_normalized_text"]
                ),
                minimum_normalized_text_length=int(
                    eligibility["minimum_normalized_text_length"]
                ),
                require_buyer=bool(eligibility["require_buyer"]),
                allowed_recommendation_status=tuple(
                    eligibility["allowed_recommendation_status"]
                ),
                minimum_helpfulness_votes=int(
                    eligibility.get("minimum_helpfulness_votes", 1)
                ),
            ),
            download_chunk_bytes=int(raw["runtime"].get("download_chunk_bytes", 1048576)),
            resolution=ResolutionSettings(**raw.get("resolution", {})),
            bm25=BM25Settings(**raw.get("bm25", {})),
            dense=DenseSettings(**raw.get("dense", {})),
            hybrid=HybridSettings(
                **{
                    **raw.get("hybrid", {}),
                    "tuning_rrf_k": tuple(raw.get("hybrid", {}).get("tuning_rrf_k", (20, 40, 60, 80))),
                }
            ),
            reranker=RerankerSettings(
                **{
                    **raw.get("reranker", {}),
                    "candidate_depths": tuple(
                        raw.get("reranker", {}).get("candidate_depths", (20, 50, 100))
                    ),
                }
            ),
            comparison=ComparisonSettings(**raw.get("comparison", {})),
            generation=GenerationSettings(**raw.get("generation", {})),
            grounding=GroundingSettings(**raw.get("grounding", {})),
            final_evaluation=FinalEvaluationSettings(**raw.get("final_evaluation", {})),
            final_evaluation_v2=FinalEvaluationV2Settings(**raw.get("final_evaluation_v2", {})),
        )
