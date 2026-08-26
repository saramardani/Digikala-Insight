"""Controlled BGE-M3/BM25 retrieval evaluation and resource reporting."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import polars as pl

from .config import Settings
from .dense_embedding import runtime_resources
from .dense_index import DenseFaissRetriever, DenseIndexPaths
from .retrieval_metrics import retrieval_metrics_at_k
from .runtime import peak_process_memory_bytes


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _required(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} must be configured")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benchmark(settings: Settings) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    query_path = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_path = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    queries = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line]
    labels: dict[str, dict[str, int]] = {}
    for row in pl.read_csv(qrels_path).to_dicts():
        labels.setdefault(str(row["query_id"]), {})[str(row["review_id"])] = int(row["relevance_grade"])
    return queries, labels


def write_resource_estimate(settings: Settings, manifest: dict[str, Any]) -> dict[str, Any]:
    """Extrapolate a full CPU run only from a real, persisted pilot measurement."""
    path = _required(settings.paths.dense_resource_report, "dense_resource_report")
    full_count = int(manifest["full_corpus_document_count"])
    rate = manifest.get("embedding_documents_per_second")
    dimension = manifest.get("dimension")
    estimate_seconds = full_count / rate if rate else None
    vector_bytes = full_count * int(dimension) * 4 if dimension else None
    report = {
        "status": "estimate_from_pilot",
        "model": manifest.get("model"),
        "pilot_document_count": manifest.get("expected_documents"),
        "pilot_is_full_corpus": manifest.get("is_full_corpus"),
        "pilot_embedding_runtime_seconds": manifest.get("embedding_runtime_seconds"),
        "pilot_documents_per_second": rate,
        "full_corpus_document_count": full_count,
        "estimated_full_embedding_seconds": estimate_seconds,
        "estimated_full_embedding_days": estimate_seconds / 86400 if estimate_seconds else None,
        "estimated_raw_float32_vector_bytes": vector_bytes,
        "estimated_raw_float32_vector_gib": vector_bytes / (1024**3) if vector_bytes else None,
        "observed_peak_process_memory_bytes": manifest.get("peak_process_memory_bytes"),
        "resources": manifest.get("resources", runtime_resources()),
        "decision": (
            "Do not start a full CPU index automatically. Run it only on a machine that "
            "has adequate RAM/disk and accepts the estimate above, or use a GPU-capable host."
        ),
    }
    _write_json(path, report)
    return report


def write_resource_limitation(
    settings: Settings, *, reason: str, observed_loader_peak_memory_bytes: int | None = None
) -> dict[str, Any]:
    """Persist a transparent block when even a real BGE-M3 pilot cannot load."""
    path = _required(settings.paths.dense_resource_report, "dense_resource_report")
    corpus = _required(settings.paths.retrieval_corpus, "retrieval_corpus")
    document_count = int(pl.scan_parquet(corpus).select(pl.len()).collect()[0, 0])
    vector_bytes = document_count * settings.dense.embedding_dimension * 4
    report = {
        "status": "resource_limited_before_pilot_throughput",
        "reason": reason,
        "model": {
            "model_id": settings.dense.model_id,
            "model_revision": settings.dense.model_revision,
            "backend": settings.dense.backend,
            "embedding_dimension": settings.dense.embedding_dimension,
            "device": settings.dense.device,
            "dtype": "float16" if settings.dense.use_fp16 else "float32",
            "batch_size": settings.dense.batch_size,
            "max_length": settings.dense.max_length,
            "normalize_embeddings": settings.dense.normalize_embeddings,
        },
        "full_corpus_document_count": document_count,
        "estimated_raw_float32_vector_bytes": vector_bytes,
        "estimated_raw_float32_vector_gib": vector_bytes / (1024**3),
        "estimated_full_embedding_seconds": None,
        "estimated_full_embedding_days": None,
        "time_estimate_reason": "No throughput is reported because BGE-M3 could not finish initialization safely on this host; fabricating a CPU estimate would be misleading.",
        "observed_loader_peak_memory_bytes": observed_loader_peak_memory_bytes,
        "resources": runtime_resources(),
        "required_next_environment": "GPU-capable host, or substantially more free RAM plus a successful measured pilot before any full CPU build.",
    }
    _write_json(path, report)
    return report


def _unavailable_dense_report(settings: Settings, reason: str) -> dict[str, Any]:
    report_path = _required(settings.paths.dense_evaluation_report, "dense_evaluation_report")
    query_path = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_path = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    report = {
        "method": "bge-m3-dense",
        "status": "unavailable",
        "reason": reason,
        "corpus_version": f"{settings.dataset.revision}:bm25-corpus-v1",
        "configuration": {
            "model_id": settings.dense.model_id,
            "model_revision": settings.dense.model_revision,
            "backend": settings.dense.backend,
            "device": settings.dense.device,
            "dtype": "float16" if settings.dense.use_fp16 else "float32",
            "batch_size": settings.dense.batch_size,
            "max_length": settings.dense.max_length,
            "embedding_dimension": settings.dense.embedding_dimension,
            "normalize_embeddings": settings.dense.normalize_embeddings,
            "top_k": settings.bm25.default_top_k,
            "candidate_depth": settings.bm25.candidate_depth,
        },
        "frozen_benchmark": {
            "queries_sha256": _sha256(query_path),
            "qrels_sha256": _sha256(qrels_path),
            "qrels_policy": "same Phase 4 seed labels; zero-qrel queries are excluded from aggregate metrics",
            "label_limitations": "seed lexical qrels pending human review are not final relevance ground truth",
        },
        "metrics_at_k": {"recall": None, "precision": None, "mrr": None, "ndcg": None},
        "latency_ms": None,
        "storage_bytes": None,
        "resources": runtime_resources(),
    }
    _write_json(report_path, report)
    return report


def evaluate_dense(settings: Settings) -> dict[str, Any]:
    """Evaluate a *full* BGE-M3 index against the unchanged Phase 4 qrels."""
    paths = DenseIndexPaths.from_settings(settings)
    if not paths.manifest.is_file():
        resource_path = _required(settings.paths.dense_resource_report, "dense_resource_report")
        if resource_path.is_file():
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            reason = f"no dense manifest; {resource.get('reason', 'build a full BGE-M3 index first')}"
        else:
            reason = "no dense manifest; build a full BGE-M3 index first"
        report = _unavailable_dense_report(settings, reason)
        write_controlled_comparison(settings, report)
        return report
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("is_full_corpus"):
        report = _unavailable_dense_report(
            settings,
            "dense index is absent, incomplete, or only a pilot; a controlled full-corpus comparison is unavailable",
        )
        write_controlled_comparison(settings, report)
        return report
    if manifest.get("corpus_version") != f"{settings.dataset.revision}:bm25-corpus-v1":
        report = _unavailable_dense_report(settings, "dense manifest corpus version differs from Phase 4 corpus")
        write_controlled_comparison(settings, report)
        return report

    queries, labels = _benchmark(settings)
    retriever = DenseFaissRetriever.from_settings(settings)
    result_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    cold_latencies: list[float] = []
    warm_latencies: list[float] = []
    started = perf_counter()
    for query in queries:
        cold, cold_ms = retriever.timed_retrieve(query["product_id"], query["query"], settings.bm25.candidate_depth)
        warm, warm_ms = retriever.timed_retrieve(query["product_id"], query["query"], settings.bm25.candidate_depth)
        cold_latencies.append(cold_ms)
        warm_latencies.append(warm_ms)
        qrels = labels.get(query["query_id"], {})
        metrics = retrieval_metrics_at_k(
            [result.review_id for result in warm], qrels, settings.bm25.default_top_k
        )
        per_query.append(
            {
                "query_id": query["query_id"],
                "known_relevance_count": len(qrels),
                "cold_latency_ms": cold_ms,
                "warm_latency_ms": warm_ms,
                **metrics,
            }
        )
        result_rows.extend(
            {
                "method": "bge-m3-dense",
                "query_id": query["query_id"],
                "rank": result.rank,
                "score": result.score,
                "review_id": result.review_id,
                "product_id": result.product_id,
            }
            for result in cold
        )
    result_path = _required(settings.paths.dense_ranked_results, "dense_ranked_results")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        result_rows,
        schema={
            "method": pl.String,
            "query_id": pl.String,
            "rank": pl.Int64,
            "score": pl.Float64,
            "review_id": pl.String,
            "product_id": pl.String,
        },
    ).write_parquet(result_path)
    evaluable = [row for row in per_query if row["known_relevance_count"] > 0]
    aggregate = {
        metric: sum(row[metric] for row in evaluable if row[metric] is not None) / len(evaluable)
        if evaluable
        else None
        for metric in ("recall", "precision", "mrr", "ndcg")
    }
    query_path = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_path = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    report = {
        "method": "bge-m3-dense",
        "status": "available",
        "corpus_version": manifest["corpus_version"],
        "configuration": manifest["model"],
        "dimension": manifest["dimension"],
        "vector_count": manifest["completed_documents"],
        "storage_bytes": manifest["index_storage_bytes"],
        "index_strategy": "persisted BGE-M3 vectors with contiguous product ranges; FAISS IndexFlatIP is built only for the requested product and cached",
        "frozen_benchmark": {
            "queries_sha256": _sha256(query_path),
            "qrels_sha256": _sha256(qrels_path),
            "qrels_policy": "same Phase 4 seed labels; zero-qrel queries are excluded from aggregate metrics",
            "label_limitations": "seed lexical qrels pending human review are not final relevance ground truth",
        },
        "query_count": len(queries),
        "evaluable_query_count": len(evaluable),
        "metrics_at_k": aggregate,
        "latency_ms": {
            "cold_p50": median(cold_latencies),
            "cold_p95": sorted(cold_latencies)[round((len(cold_latencies) - 1) * 0.95)],
            "warm_p50": median(warm_latencies),
            "warm_p95": sorted(warm_latencies)[round((len(warm_latencies) - 1) * 0.95)],
        },
        "cache_statistics": retriever.cache_statistics(),
        "performance": {
            "runtime_seconds": perf_counter() - started,
            "peak_process_memory_bytes": peak_process_memory_bytes(),
            "resources": runtime_resources(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "ranked_results_path": str(result_path),
        "per_query": per_query,
    }
    _write_json(_required(settings.paths.dense_evaluation_report, "dense_evaluation_report"), report)
    write_controlled_comparison(settings, report)
    return report


def _best_relevant_rank(rows: list[dict[str, Any]], labels: dict[str, int]) -> int | None:
    ranks = [int(row["rank"]) for row in rows if str(row["review_id"]) in labels]
    return min(ranks) if ranks else None


def write_controlled_comparison(settings: Settings, dense_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist a side-by-side report; unavailable is never treated as a loss."""
    bm25_path = _required(settings.paths.retrieval_evaluation_report, "retrieval_evaluation_report")
    dense_path = _required(settings.paths.dense_evaluation_report, "dense_evaluation_report")
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    dense = dense_report or (json.loads(dense_path.read_text(encoding="utf-8")) if dense_path.is_file() else None)
    query_path = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_path = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    full_invariants = {
        "corpus_version": f"{settings.dataset.revision}:bm25-corpus-v1",
        "queries_sha256": _sha256(query_path),
        "qrels_sha256": _sha256(qrels_path),
        "top_k": settings.bm25.default_top_k,
        "candidate_depth": settings.bm25.candidate_depth,
        "document_unit": "one Phase 4 eligible review with review_id/product_id provenance",
    }
    report = {
        "status": "available" if dense and dense.get("status") == "available" else "dense_unavailable",
        "invariants": full_invariants,
        "methods": {
            "bm25": {
                "status": bm25.get("status"),
                "metrics_at_k": bm25.get("metrics_at_k"),
                "latency_ms": bm25.get("latency_ms"),
                "storage_bytes": None,
                "peak_process_memory_bytes": bm25.get("performance", {}).get("peak_process_memory_bytes"),
            },
            "bge_m3_dense": {
                "status": dense.get("status") if dense else "unavailable",
                "metrics_at_k": dense.get("metrics_at_k") if dense else None,
                "latency_ms": dense.get("latency_ms") if dense else None,
                "storage_bytes": dense.get("storage_bytes") if dense else None,
                "peak_process_memory_bytes": (dense or {}).get("performance", {}).get("peak_process_memory_bytes"),
                "peak_vram_bytes": (dense or {}).get("performance", {}).get("resources", {}).get("cuda_peak_allocated_bytes"),
                "reason": (dense or {}).get("reason"),
            },
        },
        "conclusion": (
            "No quality conclusion: BGE-M3 full-corpus index is unavailable."
            if not dense or dense.get("status") != "available"
            else "Metrics are comparative retrieval measurements only; qrels remain seed labels pending human review."
        ),
    }
    _write_json(_required(settings.paths.dense_comparison_report, "dense_comparison_report"), report)
    write_failure_analysis(settings, dense)
    return report


def write_failure_analysis(settings: Settings, dense_report: dict[str, Any] | None) -> dict[str, Any]:
    """Record deterministic ranking disagreements; semantic plausibility needs review."""
    output = _required(settings.paths.dense_failure_analysis, "dense_failure_analysis")
    if not dense_report or dense_report.get("status") != "available":
        report = {
            "status": "unavailable",
            "reason": "full dense benchmark is unavailable, so dense/BM25 failure cases cannot be identified fairly",
            "dense_wins": [],
            "bm25_wins": [],
            "both_fail": [],
            "plausible_but_unsupported_candidates": [],
        }
        _write_json(output, report)
        return report
    queries, labels_by_query = _benchmark(settings)
    bm25_rows = pl.read_parquet(_required(settings.paths.retrieval_results, "retrieval_results")).to_dicts()
    dense_rows = pl.read_parquet(_required(settings.paths.dense_ranked_results, "dense_ranked_results")).to_dicts()
    by_query: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method, rows in (("bm25", bm25_rows), ("bge_m3_dense", dense_rows)):
        for row in rows:
            by_query.setdefault(str(row["query_id"]), {}).setdefault(method, []).append(row)
    groups = {"dense_wins": [], "bm25_wins": [], "both_fail": [], "plausible_but_unsupported_candidates": []}
    for query in queries:
        query_id = str(query["query_id"])
        labels = labels_by_query.get(query_id, {})
        if not labels:
            continue
        bm25 = by_query.get(query_id, {}).get("bm25", [])
        dense = by_query.get(query_id, {}).get("bge_m3_dense", [])
        bm25_rank = _best_relevant_rank(bm25, labels)
        dense_rank = _best_relevant_rank(dense, labels)
        example = {
            "query_id": query_id,
            "product_id": query["product_id"],
            "query": query["query"],
            "bm25_first_relevant_rank": bm25_rank,
            "dense_first_relevant_rank": dense_rank,
            "bm25_top": bm25[:3],
            "dense_top": dense[:3],
        }
        if dense_rank is not None and (bm25_rank is None or dense_rank < bm25_rank):
            groups["dense_wins"].append(example)
        elif bm25_rank is not None and (dense_rank is None or bm25_rank < dense_rank):
            groups["bm25_wins"].append(example)
        elif dense_rank is None and bm25_rank is None:
            groups["both_fail"].append(example)
        if dense and str(dense[0]["review_id"]) not in labels:
            groups["plausible_but_unsupported_candidates"].append(
                {
                    **example,
                    "judgment": "requires_human_review; a non-qrel semantic hit is not automatically evidence",
                }
            )
    report = {"status": "available", **groups}
    _write_json(output, report)
    return report
