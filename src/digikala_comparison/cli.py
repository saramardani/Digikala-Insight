"""Small explicit command-line interface for Phase 1 and Phase 2 operations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

from .config import Settings
from .acquisition import download_pinned_dataset
from .errors import DatasetDownloadError, DatasetPathError
from .pipeline import (
    build_canonical_products_artifact,
    build_statistics_artifact,
    generate_quality_report,
    run_preprocessing,
)
from .resolver import ProductResolver
from .resolution_validation import run_resolution_validation
from .bm25 import ProductScopedBM25
from .retrieval_pipeline import (
    build_retrieval_corpus,
    create_frozen_benchmark,
    evaluate_bm25,
)
from .dense_evaluation import evaluate_dense, write_resource_estimate
from .dense_index import DenseFaissRetriever, DenseIndexPaths, build_dense_embeddings
from .hybrid import HybridRRFRetriever
from .hybrid_evaluation import evaluate_hybrid
from .reranker import HybridBgeReranker
from .reranker_evaluation import evaluate_reranker
from .retrieval_freeze import run_retrieval_benchmark
from .evidence import ProductionEvidenceRetriever
from .comparison import CriterionRequest, PreferencePolicy, ProductComparisonService
from .generation import (
    GroundingValidationError,
    LLMProviderError,
    MetisOpenAICompatibleProvider,
    StructuredComparisonGenerator,
    SYSTEM_PROMPT,
)
from .grounding import (
    DeterministicGroundingValidator,
    GroundingAuditRecord,
    GroundingAuditStore,
)
from .final_evaluation import run_final_evaluation
from .final_evaluation_v2 import initialize_final_evaluation_v2
from .product_search_evaluation import evaluate_product_search, freeze_product_search_cases
from .review_qa_evaluation import evaluate_review_qa_evidence, freeze_review_qa_cases
from .comparison_evaluation_v2 import evaluate_comparisons, freeze_comparison_cases
from .manager_analysis_evaluation import evaluate_manager_analysis, freeze_manager_analysis_cases
from .final_report_v2 import build_final_report_v2
from .human_evaluation import aggregate_human_evaluation_bundle


def _parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Digikala product-comparison pipeline: {command}"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.toml"),
        help="Path to the TOML settings file.",
    )
    return parser


def preprocess_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("preprocess").parse_args(argv)
    try:
        artifacts = run_preprocessing(Settings.from_toml(args.config))
    except DatasetPathError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


def quality_report_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("quality-report").parse_args(argv)
    try:
        settings = Settings.from_toml(args.config)
        generate_quality_report(settings)
    except DatasetPathError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"quality_report: {settings.paths.quality_report}")
    return 0


def download_data_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("download-data")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing raw files after download."
    )
    args = parser.parse_args(argv)
    try:
        results = download_pinned_dataset(Settings.from_toml(args.config), force=args.force)
    except DatasetDownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for result in results:
        print(
            f"{result.filename}: {result.status}; path={result.path}; "
            f"size_bytes={result.size_bytes}; revision={result.revision}"
        )
    return 0


def build_statistics_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("build-statistics").parse_args(argv)
    try:
        artifacts = build_statistics_artifact(Settings.from_toml(args.config))
    except (DatasetPathError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


def build_canonical_products_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("build-canonical-products").parse_args(argv)
    try:
        artifacts = build_canonical_products_artifact(Settings.from_toml(args.config))
    except (DatasetPathError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


def resolve_product_main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("resolve-product")
    parser.add_argument("reference", nargs="?", help="Product title or numeric product ID.")
    parser.add_argument("--id", dest="product_id", help="Explicit product ID.")
    parser.add_argument("--brand", help="Optional exact brand constraint.")
    parser.add_argument("--category", help="Optional exact category constraint.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON result.")
    args = parser.parse_args(argv)
    if not args.reference and not args.product_id:
        parser.error("provide a reference or --id")
    settings = Settings.from_toml(args.config)
    if settings.paths.canonical_products is None or not settings.paths.canonical_products.is_file():
        print("error: canonical_products.parquet is required; run build-canonical-products", file=sys.stderr)
        return 2
    reference: object
    if args.product_id:
        reference = {"product_id": args.product_id}
    elif args.brand or args.category:
        reference = {"title": args.reference, "brand": args.brand, "category": args.category}
    else:
        reference = args.reference
    result = ProductResolver.from_parquet(
        str(settings.paths.canonical_products), settings.resolution
    ).resolve(reference)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"status: {result.status}; selected_product_id: {result.selected_product_id}")
        print(f"reason: {result.reason}")
        for candidate in result.candidates:
            print(f"- {candidate.product_id}: {candidate.title_fa} ({candidate.score:.1f})")
    return 0


def evaluate_resolution_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("evaluate-resolution").parse_args(argv)
    try:
        report = run_resolution_validation(Settings.from_toml(args.config))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    settings = Settings.from_toml(args.config)
    print(f"resolution_validation_report: {settings.paths.resolution_validation_report}")
    print(f"exact_resolution_accuracy: {report['exact_resolution_accuracy']}")
    return 0


def build_retrieval_corpus_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("build-retrieval-corpus").parse_args(argv)
    try:
        artifacts = build_retrieval_corpus(Settings.from_toml(args.config))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


def retrieve_bm25_main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("retrieve-bm25")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_toml(args.config)
    if settings.paths.retrieval_corpus is None or not settings.paths.retrieval_corpus.is_file():
        print("error: retrieval corpus missing; run build-retrieval-corpus", file=sys.stderr)
        return 2
    results = ProductScopedBM25(settings.paths.retrieval_corpus, settings.bm25).retrieve(
        args.product_id, args.query, args.top_k
    )
    if args.json:
        print(json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            excerpt = (result.review_text_raw or result.indexed_text_normalized).replace("\n", " ")[:180]
            print(f"{result.rank}. score={result.score:.4f} review_id={result.review_id} product_id={result.product_id}\n   {excerpt}")
    return 0


def evaluate_retrieval_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("evaluate-retrieval")
    parser.add_argument("--method", choices=["bm25"], default="bm25")
    parser.add_argument("--build-benchmark", action="store_true")
    parser.add_argument("--force-benchmark", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_toml(args.config)
    try:
        if args.build_benchmark or args.force_benchmark:
            created = create_frozen_benchmark(settings, force=args.force_benchmark)
            print(f"benchmark: {created}")
        report = evaluate_bm25(settings)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"evaluation_report: {settings.paths.retrieval_evaluation_report}")
    print(f"metrics_at_k: {report['metrics_at_k']}")
    return 0


def build_dense_index_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("build-dense-index")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="Embed a small real-corpus pilot (the default).")
    mode.add_argument("--full", action="store_true", help="Build the complete BGE-M3 vector corpus.")
    parser.add_argument("--max-documents", type=int, help="Override the configured pilot document count.")
    parser.add_argument(
        "--accept-cpu-full-estimate",
        action="store_true",
        help="Required with --full on a CPU-only host after inspecting the pilot estimate.",
    )
    args = parser.parse_args(argv)
    settings = Settings.from_toml(args.config)
    root = settings.paths.dense_index_root
    if root is None:
        print("error: dense_index_root must be configured", file=sys.stderr)
        return 2
    full = args.full
    if full:
        try:
            import torch
            cpu_only = not torch.cuda.is_available()
        except ImportError:
            cpu_only = True
        if cpu_only and not args.accept_cpu_full_estimate:
            print(
                "error: --full on this CPU-only host requires --accept-cpu-full-estimate; run --pilot first",
                file=sys.stderr,
            )
            return 2
        output_root, limit = root, None
    else:
        output_root = root.parent / f"{root.name}_pilot"
        limit = args.max_documents or settings.dense.pilot_documents
    try:
        result = build_dense_embeddings(settings, output_root=output_root, max_documents=limit)
        manifest = result["manifest"]
        estimate = write_resource_estimate(settings, manifest)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"dense_manifest: {result['paths'].manifest}")
    print(f"dense_status: {manifest['status']}; vectors={manifest['completed_documents']}; dimension={manifest.get('dimension')}")
    print(f"resource_estimate: {settings.paths.dense_resource_report}")
    print(f"estimated_full_embedding_days: {estimate['estimated_full_embedding_days']}")
    return 0


def retrieve_dense_main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("retrieve-dense")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_toml(args.config)
    try:
        results = DenseFaissRetriever.from_settings(settings).retrieve(args.product_id, args.query, args.top_k)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            excerpt = (result.review_text_raw or result.indexed_text_normalized).replace("\n", " ")[:180]
            print(f"{result.rank}. dense_score={result.score:.6f} review_id={result.review_id} product_id={result.product_id}\n   {excerpt}")
    return 0


def evaluate_dense_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("evaluate-dense").parse_args(argv)
    settings = Settings.from_toml(args.config)
    try:
        report = evaluate_dense(settings)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"dense_evaluation_report: {settings.paths.dense_evaluation_report}")
    print(f"status: {report['status']}")
    print(f"metrics_at_k: {report['metrics_at_k']}")
    return 0


def retrieve_hybrid_main(argv: Sequence[str] | None = None) -> int:
    """Retrieve product-scoped evidence with configured BM25 + dense RRF."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("retrieve-hybrid")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_toml(args.config)
    try:
        results = HybridRRFRetriever.from_settings(settings).retrieve(
            args.product_id, args.query, args.top_k
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            excerpt = (result.review_text_raw or result.indexed_text_normalized).replace("\n", " ")[:180]
            print(
                f"{result.rank}. fused={result.fused_score:.6f} review_id={result.review_id} "
                f"product_id={result.product_id} bm25_rank={result.bm25_rank} "
                f"dense_rank={result.dense_rank}\n   {excerpt}"
            )
    return 0


def evaluate_hybrid_main(argv: Sequence[str] | None = None) -> int:
    """Benchmark BM25, BGE-M3 dense, and their RRF fusion on frozen test IDs."""
    args = _parser("evaluate-hybrid").parse_args(argv)
    settings = Settings.from_toml(args.config)
    try:
        report = evaluate_hybrid(settings)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"hybrid_evaluation_report: {settings.paths.hybrid_evaluation_report}")
    print(f"status: {report['status']}")
    print(f"production_selection: {report['production_selection']['selected_method']}")
    for method, result in report["methods"].items():
        print(f"{method}: {result['status']}; metrics_at_k={result['metrics_at_k']}")
    return 0


def retrieve_reranked_main(argv: Sequence[str] | None = None) -> int:
    """Retrieve hybrid candidates and rerank them with the pinned BGE model."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("retrieve-reranked")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        results = HybridBgeReranker.from_settings(Settings.from_toml(args.config)).retrieve(
            args.product_id, args.query, args.top_k
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            excerpt = (result.review_text_raw or result.indexed_text_normalized).replace("\n", " ")[:180]
            print(
                f"{result.final_rank}. reranker={result.reranker_score:.6f} "
                f"review_id={result.review_id} product_id={result.product_id} "
                f"fused_rank={result.fused_rank}\n   {excerpt}"
            )
    return 0


def evaluate_reranker_main(argv: Sequence[str] | None = None) -> int:
    """Benchmark BM25, dense, hybrid, and hybrid + BGE reranker."""
    args = _parser("evaluate-reranker").parse_args(argv)
    settings = Settings.from_toml(args.config)
    try:
        report = evaluate_reranker(settings)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"reranker_evaluation_report: {settings.paths.reranker_evaluation_report}")
    print(f"status: {report['status']}")
    print(f"production_selection: {report['production_selection']['selected_method']}")
    for method, result in report["methods"].items():
        print(f"{method}: {result['status']}; metrics_at_k={result['metrics_at_k']}")
    return 0


def retrieval_benchmark_main(argv: Sequence[str] | None = None) -> int:
    """Run the frozen four-way benchmark and publish reproducibility artifacts."""
    parser = _parser("retrieval-benchmark")
    parser.add_argument("--all", action="store_true", help="Run all four required retrieval methods.")
    parser.add_argument("--force-freeze", action="store_true", help="Rewrite the frozen manifest from the current pinned inputs.")
    args = parser.parse_args(argv)
    if not args.all:
        parser.error("--all is required; this command always reports the four-method benchmark")
    settings = Settings.from_toml(args.config)
    try:
        result = run_retrieval_benchmark(settings, force_freeze=args.force_freeze)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    benchmark = result["benchmark"]
    print(f"benchmark_report: {settings.paths.reranker_evaluation_report}")
    print(f"experiment_manifest: {settings.paths.retrieval_experiment_manifest}")
    print(f"markdown_summary: {settings.paths.retrieval_benchmark_markdown}")
    print(f"status: {benchmark['status']}")
    print(f"production_selection: {benchmark['production_selection']['selected_method']}")
    return 0


def retrieve_evidence_main(argv: Sequence[str] | None = None) -> int:
    """Retrieve frozen production evidence with review-level provenance."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("retrieve-evidence")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--criterion", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        evidence = ProductionEvidenceRetriever.from_settings(Settings.from_toml(args.config)).retrieve_evidence(
            args.product_id, args.criterion, args.query, args.top_k
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(evidence.model_dump_json(indent=2))
    else:
        print(
            f"method={evidence.retrieval_method}; status={evidence.retrieval_status}; "
            f"returned={evidence.retrieved_count}/{evidence.requested_top_k}; "
            f"eligible_product_reviews={evidence.eligible_product_review_count}"
        )
        for item in evidence.evidence_items:
            excerpt = (item.raw_evidence_text or "").replace("\n", " ")[:180]
            print(f"{item.rank}. score={item.final_score:.6f} review_id={item.review_id}\n   {excerpt}")
    return 0


def compare_structured_main(argv: Sequence[str] | None = None) -> int:
    """Inspect a deterministic, provenance-preserving multi-product comparison."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("compare-structured")
    parser.add_argument("--product-id", action="append", required=True, help="Resolved product ID; repeat for each product.")
    parser.add_argument("--criterion", action="append", required=True, help="Comparison criterion; repeat as needed.")
    parser.add_argument("--evidence-query", action="append", default=[], metavar="CRITERION=QUERY", help="Optional query override for a review-based criterion.")
    parser.add_argument("--weight", action="append", default=[], metavar="CRITERION=WEIGHT", help="Optional explicit overall-preference weight.")
    parser.add_argument("--top-k", type=int, default=10, help="Evidence Top-K for review-based criteria.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON (the default output).")
    args = parser.parse_args(argv)
    if len(args.product_id) < 2:
        parser.error("at least two --product-id values are required")
    if len(args.product_id) != len(set(args.product_id)):
        parser.error("--product-id values must be unique")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        evidence_queries = _key_value_arguments(args.evidence_query, "--evidence-query")
        weights = {key: float(value) for key, value in _key_value_arguments(args.weight, "--weight").items()}
        criteria = [CriterionRequest(name=name, evidence_query=evidence_queries.get(name)) for name in args.criterion]
        policy = PreferencePolicy(weights=weights) if weights else None
        result = ProductComparisonService.from_settings(Settings.from_toml(args.config)).compare_product_ids(
            args.product_id, criteria, evidence_top_k=args.top_k, preference_policy=policy
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    # The command's public contract is structured JSON; --json is retained for
    # consistency with the other debug commands and future formatting options.
    print(result.model_dump_json(indent=2))
    return 0


def generate_comparison_main(argv: Sequence[str] | None = None) -> int:
    """Generate a validated Persian rendering of a deterministic comparison."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("generate-comparison")
    parser.add_argument("--product-id", action="append", required=True, help="Resolved product ID; repeat for each product.")
    parser.add_argument("--criterion", action="append", required=True, help="Comparison criterion; repeat as needed.")
    parser.add_argument("--evidence-query", action="append", default=[], metavar="CRITERION=QUERY")
    parser.add_argument("--weight", action="append", default=[], metavar="CRITERION=WEIGHT")
    parser.add_argument("--question", help="Optional user question to include in the bounded model context.")
    parser.add_argument("--priority", action="append", default=[], help="Optional user priority; repeat as needed.")
    parser.add_argument("--top-k", type=int, default=10, help="Evidence Top-K for review-based criteria.")
    parser.add_argument("--dry-run", action="store_true", help="Print the structured model input without calling the API.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the optional development response cache.")
    args = parser.parse_args(argv)
    if len(args.product_id) < 2:
        parser.error("at least two --product-id values are required")
    if len(args.product_id) != len(set(args.product_id)):
        parser.error("--product-id values must be unique")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        settings = Settings.from_toml(args.config)
        evidence_queries = _key_value_arguments(args.evidence_query, "--evidence-query")
        weights = {
            key: float(value)
            for key, value in _key_value_arguments(args.weight, "--weight").items()
        }
        criteria = [
            CriterionRequest(name=name, evidence_query=evidence_queries.get(name))
            for name in args.criterion
        ]
        policy = PreferencePolicy(weights=weights) if weights else None
        result = ProductComparisonService.from_settings(settings).compare_product_ids(
            args.product_id,
            criteria,
            evidence_top_k=args.top_k,
            preference_policy=policy,
        )
        generator = StructuredComparisonGenerator.from_settings(settings)
        if args.dry_run:
            context = generator.dry_run_input(
                result, user_question=args.question, user_priorities=args.priority
            )
            print(
                json.dumps(
                    {
                        "system_prompt": SYSTEM_PROMPT,
                        "generation_context": context.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        outcome = generator.generate(
            result,
            user_question=args.question,
            user_priorities=args.priority,
            use_cache=not args.no_cache,
        )
    except (FileNotFoundError, RuntimeError, ValueError, LLMProviderError, GroundingValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output_path = _write_generation_text_output(settings.paths.generation_output_root, outcome)
    print(_final_generation_summary(result))
    print(f"full_output: {output_path}")
    return 0


def _write_generation_text_output(root: Path | None, outcome: object) -> Path:
    """Persist the complete accepted answer for audit without console noise."""

    if root is None:
        raise ValueError("generation_output_root must be configured")
    rendered = getattr(outcome, "rendered_persian")
    metadata_json = getattr(outcome, "model_dump_json")(indent=2)
    fingerprint = getattr(outcome, "context_fingerprint")[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"comparison_{timestamp}_{fingerprint}.txt"
    content = f"{rendered}\n\n--- structured metadata ---\n{metadata_json}\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def _final_generation_summary(result: object) -> str:
    """Render only the deterministic overall decision for the terminal."""

    overall = getattr(result, "overall")
    winner_ids = list(getattr(overall, "winner_product_ids", []))
    product_labels = {
        product.product_id: product.title_fa
        for product in getattr(result, "products")
    }
    if winner_ids:
        labels = "، ".join(
            f"{product_labels.get(product_id, product_id)} ({product_id})"
            for product_id in winner_ids
        )
        return f"نتیجه نهایی: {labels} بر اساس اولویت‌ها و محاسبات قطعی سیستم پیشنهاد می‌شود."
    status = getattr(overall, "status", "inconclusive")
    reason = getattr(overall, "reason_code", "NO_AUTHORIZED_WINNER")
    return f"نتیجه نهایی: برندهٔ قطعی اعلام نشد ({status}: {reason})."


def list_llm_models_main(argv: Sequence[str] | None = None) -> int:
    """List account-enabled Metis model IDs without creating a completion."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser("list-llm-models").parse_args(argv)
    try:
        settings = Settings.from_toml(args.config)
        if settings.generation.provider != MetisOpenAICompatibleProvider.name:
            raise ValueError("digikala-list-llm-models requires provider = metis_openai_compatible")
        model_ids = MetisOpenAICompatibleProvider().list_models(settings.generation)
    except (FileNotFoundError, ValueError, LLMProviderError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"provider": settings.generation.provider, "models": model_ids}, ensure_ascii=False, indent=2))
    return 0


def validate_grounding_main(argv: Sequence[str] | None = None) -> int:
    """Validate a saved structured answer against its exact generation context."""
    parser = _parser("validate-grounding")
    parser.add_argument("--context", type=Path, required=True, help="Saved GenerationContext JSON from a dry run or trace.")
    parser.add_argument("--answer", type=Path, required=True, help="Saved GeneratedComparisonAnswer JSON.")
    parser.add_argument(
        "--action",
        choices=["reject", "remove_unsupported", "rewrite_regenerate"],
        help="Explicit unsupported-claim handling policy; defaults to [grounding].unsupported_claim_action.",
    )
    parser.add_argument("--audit-output", type=Path, help="Optional explicit audit JSON destination.")
    args = parser.parse_args(argv)
    try:
        from .generation import GeneratedComparisonAnswer, GenerationContext

        settings = Settings.from_toml(args.config)
        context = GenerationContext.model_validate_json(args.context.read_text(encoding="utf-8"))
        answer = GeneratedComparisonAnswer.model_validate_json(args.answer.read_text(encoding="utf-8"))
        validator = DeterministicGroundingValidator.from_settings(settings)
        policy = validator.enforce(
            answer,
            context,
            action=args.action or settings.grounding.unsupported_claim_action,  # type: ignore[arg-type]
        )
        final = policy.final_validation or policy.initial_validation
        audit = GroundingAuditRecord(
            validator_version=final.validator_version,
            context_fingerprint=_sha256_json(context.model_dump(mode="json")),
            original_answer=policy.original_answer,
            initial_validation=policy.initial_validation,
            final_answer=policy.final_answer,
            final_validation=policy.final_validation,
            action_taken=policy.action_taken,
        )
        if args.audit_output:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
            audit_path = args.audit_output
        elif settings.paths.grounding_audit_root:
            audit_path = GroundingAuditStore(settings.paths.grounding_audit_root).write(audit)
        else:
            audit_path = None
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(final.model_dump_json(indent=2))
    if audit_path:
        print(f"grounding_audit: {audit_path}")
    return 0 if final.valid else 2


def evaluate_final_main(argv: Sequence[str] | None = None) -> int:
    """Run the versioned Phase 12 evaluation without hidden API calls."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("evaluate-final")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-llm",
        action="store_true",
        help="Use the deterministic template plus real grounding validation (the default, cost-free mode).",
    )
    mode.add_argument(
        "--with-llm",
        action="store_true",
        help="Explicitly permit configured LLM calls for the frozen comparison cases.",
    )
    parser.add_argument("--max-llm-cases", type=int, help="Explicit cost cap for --with-llm.")
    parser.add_argument("--output-root", type=Path, help="Versioned Phase 12 artifact directory.")
    parser.add_argument(
        "--rebuild-evaluation-set",
        action="store_true",
        help="Deliberately replace the frozen case-set selection; never needed for normal reproduction.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Freeze/inspect the case set and manifest without executing cases.")
    args = parser.parse_args(argv)
    if args.max_llm_cases is not None and args.max_llm_cases <= 0:
        parser.error("--max-llm-cases must be positive")
    try:
        result = run_final_evaluation(
            Settings.from_toml(args.config),
            output_root=args.output_root,
            with_llm=args.with_llm,
            max_llm_cases=args.max_llm_cases,
            rebuild_evaluation_set=args.rebuild_evaluation_set,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def initialize_final_evaluation_v2_main(argv: Sequence[str] | None = None) -> int:
    """Initialize the immutable, staged final-evaluation v2 workspace."""
    parser = _parser("initialize-final-evaluation-v2")
    parser.add_argument("--output-root", type=Path, help="Artifact directory; defaults to [paths].final_evaluation_v2_root.")
    args = parser.parse_args(argv)
    try:
        paths = initialize_final_evaluation_v2(Settings.from_toml(args.config), output_root=args.output_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({name: str(path) for name, path in paths.items()}, ensure_ascii=False, indent=2))
    return 0


def freeze_product_search_evaluation_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("freeze-product-search-evaluation")
    parser.add_argument("--output-root", type=Path, help="Final-evaluation v2 artifact directory.")
    args = parser.parse_args(argv)
    try:
        path = freeze_product_search_cases(Settings.from_toml(args.config), output_root=args.output_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"case_set": str(path), "status": "product_search_cases_frozen"}, ensure_ascii=False, indent=2))
    return 0


def evaluate_product_search_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("evaluate-product-search")
    parser.add_argument("--output-root", type=Path, help="Final-evaluation v2 artifact directory.")
    args = parser.parse_args(argv)
    try:
        result = evaluate_product_search(Settings.from_toml(args.config), output_root=args.output_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def freeze_review_qa_evaluation_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("freeze-review-qa-evaluation")
    parser.add_argument("--output-root", type=Path, help="Final-evaluation v2 artifact directory.")
    args = parser.parse_args(argv)
    try:
        path = freeze_review_qa_cases(Settings.from_toml(args.config), output_root=args.output_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"case_set": str(path), "status": "review_qa_cases_frozen"}, ensure_ascii=False, indent=2))
    return 0


def evaluate_review_qa_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("evaluate-review-qa")
    parser.add_argument("--output-root", type=Path, help="Final-evaluation v2 artifact directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Evidence depth; must be positive.")
    args = parser.parse_args(argv)
    try:
        result = evaluate_review_qa_evidence(Settings.from_toml(args.config), output_root=args.output_root, top_k=args.top_k)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def freeze_comparison_evaluation_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("freeze-comparison-evaluation")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        path = freeze_comparison_cases(Settings.from_toml(args.config), output_root=args.output_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"case_set": str(path), "status": "comparison_cases_frozen"}, ensure_ascii=False, indent=2))
    return 0


def evaluate_comparison_evaluation_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("evaluate-comparison")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--evidence-top-k", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = evaluate_comparisons(Settings.from_toml(args.config), output_root=args.output_root, evidence_top_k=args.evidence_top_k)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def freeze_manager_analysis_evaluation_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("freeze-manager-analysis-evaluation"); parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try: path = freeze_manager_analysis_cases(Settings.from_toml(args.config), output_root=args.output_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr); return 2
    print(json.dumps({"case_set": str(path), "status": "manager_analysis_cases_frozen"}, ensure_ascii=False, indent=2)); return 0


def evaluate_manager_analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("evaluate-manager-analysis"); parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try: result = evaluate_manager_analysis(Settings.from_toml(args.config), output_root=args.output_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr); return 2
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


def report_final_evaluation_v2_main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("report-final-evaluation-v2")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prediction-metrics", type=Path, help="Held-out prediction metrics JSON containing macro_f1.")
    args = parser.parse_args(argv)
    try: result = build_final_report_v2(Settings.from_toml(args.config), output_root=args.output_root, prediction_metrics_path=args.prediction_metrics)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr); return 2
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


def aggregate_human_evaluation_main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser("aggregate-human-evaluation")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        root = initialize_final_evaluation_v2(Settings.from_toml(args.config), output_root=args.output_root)["root"]
        result = aggregate_human_evaluation_bundle(root)
        output = root / "human_evaluation_metrics.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr); return 2
    print(json.dumps({"metrics": result, "path": str(output)}, ensure_ascii=False, indent=2)); return 0


def _key_value_arguments(values: Sequence[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"{option} values must use CRITERION=VALUE")
        if key.strip() in result:
            raise ValueError(f"duplicate {option} criterion: {key.strip()}")
        result[key.strip()] = value.strip()
    return result


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
