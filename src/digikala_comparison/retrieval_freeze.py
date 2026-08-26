"""Frozen retrieval benchmark artifacts and reproducible experiment manifest."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import polars as pl

from .config import Settings
from .dense_embedding import runtime_resources
from .hybrid_evaluation import _required, freeze_hybrid_splits
from .reranker_evaluation import evaluate_reranker
from .retrieval_text import PERSIAN_LEXICAL_TOKENIZER_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _copy_frozen_partition(
    *,
    query_source: Path,
    qrels_source: Path,
    query_ids: list[str],
    query_target: Path,
    qrels_target: Path,
) -> dict[str, Any]:
    identifiers = set(query_ids)
    source_lines = query_source.read_text(encoding="utf-8").splitlines()
    selected_lines = [line for line in source_lines if line and str(json.loads(line)["query_id"]) in identifiers]
    selected_ids = [str(json.loads(line)["query_id"]) for line in selected_lines]
    if set(selected_ids) != identifiers or len(selected_ids) != len(identifiers):
        raise ValueError("frozen split IDs do not match the retrieval query artifact")
    query_target.parent.mkdir(parents=True, exist_ok=True)
    query_target.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    qrels = pl.read_csv(qrels_source)
    selected_qrels = qrels.filter(pl.col("query_id").cast(pl.String).is_in(sorted(identifiers)))
    qrels_target.parent.mkdir(parents=True, exist_ok=True)
    selected_qrels.write_csv(qrels_target)
    return {
        "query_path": str(query_target),
        "qrels_path": str(qrels_target),
        "query_count": len(selected_ids),
        "qrel_count": selected_qrels.height,
        "queries_sha256": _sha256(query_target),
        "qrels_sha256": _sha256(qrels_target),
    }


def _markdown_summary(benchmark: dict[str, Any], manifest: dict[str, Any]) -> str:
    def display_bytes(value: int | None) -> str:
        if value is None:
            return "-"
        return f"{value / 1024 / 1024:.1f} MiB"

    lines = [
        "# Frozen retrieval benchmark",
        "",
        f"Corpus version: `{manifest['corpus']['version']}`",
        "",
        "| Method | Status | Recall@K | Precision@K | MRR | NDCG@K | warm p50/p95 | Storage | Peak process memory |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in benchmark["methods"].items():
        metrics = method["metrics_at_k"]
        latency = method.get("latency_ms") or {}
        p50 = latency.get("end_to_end_warm_p50", latency.get("warm_p50"))
        p95 = latency.get("end_to_end_warm_p95", latency.get("warm_p95"))
        metrics_text = ["-" if metrics[key] is None else f"{metrics[key]:.4f}" for key in ("recall", "precision", "mrr", "ndcg")]
        latency_text = "-" if p50 is None else f"{p50:.3f} / {p95:.3f} ms"
        lines.append(
            f"| {name} | {method['status']} | {' | '.join(metrics_text)} | {latency_text} | "
            f"{display_bytes(method.get('storage_bytes'))} | {display_bytes(method.get('peak_process_memory_bytes'))} |"
        )
    selected = manifest["selected_production_retriever"]
    lines.extend(
        [
            "",
            f"Selected production retriever: `{selected['selected_method']}` ({selected['status']}).",
            "",
            "The relevance labels are deterministic lexical seed labels pending human review; this report is a reproducibility artifact, not a final relevance claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def freeze_retrieval_experiment(
    settings: Settings, benchmark: dict[str, Any], *, force: bool = False
) -> dict[str, Any]:
    """Version split copies and immutable inputs for the production evidence API."""
    manifest_path = _required(settings.paths.retrieval_experiment_manifest, "retrieval_experiment_manifest")
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    split = freeze_hybrid_splits(settings)
    query_source = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_source = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    development = _copy_frozen_partition(
        query_source=query_source,
        qrels_source=qrels_source,
        query_ids=list(split["development_query_ids"]),
        query_target=_required(settings.paths.frozen_development_queries, "frozen_development_queries"),
        qrels_target=_required(settings.paths.frozen_development_qrels, "frozen_development_qrels"),
    )
    test = _copy_frozen_partition(
        query_source=query_source,
        qrels_source=qrels_source,
        query_ids=list(split["test_query_ids"]),
        query_target=_required(settings.paths.frozen_test_queries, "frozen_test_queries"),
        qrels_target=_required(settings.paths.frozen_test_qrels, "frozen_test_qrels"),
    )
    corpus = _required(settings.paths.retrieval_corpus, "retrieval_corpus")
    corpus_report_path = _required(settings.paths.retrieval_corpus_report, "retrieval_corpus_report")
    corpus_report = json.loads(corpus_report_path.read_text(encoding="utf-8"))
    selection = benchmark["production_selection"]
    manifest = {
        "schema_version": "frozen-retrieval-experiment-v1",
        "dataset": asdict(settings.dataset),
        "random_seed": settings.random_seed,
        "corpus": {
            "path": str(corpus),
            "sha256": _sha256(corpus),
            "size_bytes": corpus.stat().st_size,
            "version": corpus_report["corpus_version"],
            "document_unit": corpus_report["document_unit"],
            "eligible_retrieval_reviews": corpus_report["eligible_retrieval_reviews"],
            "eligibility_policy": corpus_report["eligibility_policy"],
            "field_composition_policy": corpus_report["field_composition_policy"],
            "indexed_fields": corpus_report["indexed_fields"],
        },
        "tokenizer": {
            "name": "digikala_comparison.retrieval_text.tokenize_persian_lexical",
            "version": PERSIAN_LEXICAL_TOKENIZER_VERSION,
            "policy": corpus_report["tokenization"],
            "normalization": asdict(settings.normalization),
        },
        "partitions": {"development": development, "test": test},
        "source_split": split,
        "configuration": {
            "bm25": asdict(settings.bm25),
            "dense": asdict(settings.dense),
            "hybrid": asdict(settings.hybrid),
            "reranker": asdict(settings.reranker),
        },
        "selected_production_retriever": selection,
        "benchmark": {
            "report_path": str(_required(settings.paths.reranker_evaluation_report, "reranker_evaluation_report")),
            "ranked_results_path": benchmark["ranked_results_path"],
            "status": benchmark["status"],
            "methods": {
                name: {
                    "status": method["status"],
                    "metrics_at_k": method["metrics_at_k"],
                    "latency_ms": method.get("latency_ms"),
                    "storage_bytes": method.get("storage_bytes"),
                    "peak_process_memory_bytes": method.get("peak_process_memory_bytes"),
                }
                for name, method in benchmark["methods"].items()
            },
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                package: _package_version(package)
                for package in ("polars", "numpy", "FlagEmbedding", "faiss-cpu", "pydantic", "pyarrow")
            },
        },
        "hardware_at_freeze": runtime_resources(),
        "label_policy": benchmark["label_limitations"],
    }
    _write_json(manifest_path, manifest)
    markdown_path = _required(settings.paths.retrieval_benchmark_markdown, "retrieval_benchmark_markdown")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown_summary(benchmark, manifest), encoding="utf-8")
    # Return the persisted JSON form.  This avoids a mismatch between Python
    # tuples in dataclass configuration and JSON lists loaded by later runs.
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_frozen_retrieval_experiment(settings: Settings) -> dict[str, Any]:
    path = _required(settings.paths.retrieval_experiment_manifest, "retrieval_experiment_manifest")
    if not path.is_file():
        raise FileNotFoundError("frozen retrieval manifest is missing; run digikala-retrieval-benchmark --all first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_retrieval_benchmark(settings: Settings, *, force_freeze: bool = False) -> dict[str, Any]:
    """Cleanly execute the four-method benchmark and publish frozen artifacts."""
    benchmark = evaluate_reranker(settings)
    # A clean benchmark run must publish a manifest/Markdown pair describing
    # that exact run.  The immutable inputs remain content-addressed by hashes;
    # only observed latency/resource measurements are refreshed.
    manifest = freeze_retrieval_experiment(settings, benchmark, force=True)
    return {"benchmark": benchmark, "manifest": manifest}
