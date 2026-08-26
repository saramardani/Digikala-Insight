"""Leakage-safe RRF tuning and BM25/dense/hybrid benchmark reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import polars as pl

from .bm25 import ProductScopedBM25, RetrievedReview
from .config import Settings
from .dense_evaluation import _sha256
from .dense_index import DenseFaissRetriever
from .hybrid import HybridRRFRetriever
from .retrieval_metrics import retrieval_metrics_at_k
from .runtime import peak_process_memory_bytes


_RANKED_RESULT_SCHEMA = {
    "method": pl.String,
    "query_id": pl.String,
    "product_id": pl.String,
    "review_id": pl.String,
    "rank": pl.Int64,
    "score": pl.Float64,
    "bm25_score": pl.Float64,
    "bm25_rank": pl.Int64,
    "dense_score": pl.Float64,
    "dense_rank": pl.Int64,
    "fused_score": pl.Float64,
    "fused_rank": pl.Int64,
    "reranker_score": pl.Float64,
    "final_rank": pl.Int64,
    "indexed_text_normalized": pl.String,
    "review_text_raw": pl.String,
    "title_raw": pl.String,
    "advantages_items": pl.List(pl.String),
    "disadvantages_items": pl.List(pl.String),
    "is_buyer": pl.Boolean,
    "recommendation_status": pl.String,
    "review_rate": pl.Float64,
    "likes": pl.Float64,
    "dislikes": pl.Float64,
}


def _required(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} must be configured")
    return path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _queries_and_qrels(settings: Settings) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    queries_path = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_path = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line]
    labels: dict[str, dict[str, int]] = {}
    for row in pl.read_csv(qrels_path).to_dicts():
        labels.setdefault(str(row["query_id"]), {})[str(row["review_id"])] = int(row["relevance_grade"])
    return queries, labels


def freeze_hybrid_splits(settings: Settings) -> dict[str, Any]:
    """Freeze an overlay split without editing Phase 4 queries or qrels.

    Assignment is intentionally based on query evidence type/identifier, not
    on system outputs or relevance grades.  This small seed benchmark has two
    evaluable queries in each split; labels remain untouched.
    """
    path = _required(settings.paths.hybrid_splits, "hybrid_splits")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    queries, _ = _queries_and_qrels(settings)
    ids = {str(query["query_id"]) for query in queries}
    development = ["bm25_seed_01", "bm25_seed_03", "bm25_seed_04"]
    test = ["bm25_seed_02", "bm25_seed_05", "bm25_seed_low_evidence"]
    if set(development + test) != ids or len(development) + len(test) != len(ids):
        raise ValueError("the frozen Phase 4 benchmark no longer matches the Phase 6 split definition")
    queries_path = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_path = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    report = {
        "schema_version": "hybrid-split-v1",
        "corpus_version": f"{settings.dataset.revision}:bm25-corpus-v1",
        "queries_sha256": _sha256(queries_path),
        "qrels_sha256": _sha256(qrels_path),
        "assignment_policy": "fixed Phase 6 query-id/evidence-type split; no retrieval outputs or qrel grades were used to choose assignments",
        "development_query_ids": development,
        "test_query_ids": test,
        "frozen": True,
    }
    _write_json(path, report)
    return report


def _split_queries(queries: list[dict[str, Any]], split: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    identifiers = set(split[f"{partition}_query_ids"])
    return [query for query in queries if str(query["query_id"]) in identifiers]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    evaluable = [row for row in rows if row["known_relevance_count"] > 0]
    return {
        metric: (
            sum(float(row[metric]) for row in evaluable if row[metric] is not None) / len(evaluable)
            if evaluable
            else None
        )
        for metric in ("recall", "precision", "mrr", "ndcg")
    }


def _evaluate_retriever(
    *,
    method: str,
    retriever: Any,
    queries: list[dict[str, Any]],
    labels: dict[str, dict[str, int]],
    request_top_k: int,
    metric_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure cold/warm online calls using the same test queries and K."""
    cold_latencies: list[float] = []
    warm_latencies: list[float] = []
    per_query: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    for query in queries:
        started = perf_counter()
        cold = retriever.retrieve(query["product_id"], query["query"], request_top_k)
        cold_latencies.append((perf_counter() - started) * 1000)
        started = perf_counter()
        warm = retriever.retrieve(query["product_id"], query["query"], request_top_k)
        warm_latencies.append((perf_counter() - started) * 1000)
        if any(str(result.product_id) != str(query["product_id"]) for result in warm):
            raise RuntimeError(f"{method} leaked a cross-product candidate")
        qrels = labels.get(str(query["query_id"]), {})
        metrics = retrieval_metrics_at_k([result.review_id for result in warm], qrels, metric_k)
        per_query.append(
            {
                "query_id": query["query_id"],
                "product_id": query["product_id"],
                "category": query.get("category"),
                "evidence_type": query.get("evidence_type"),
                "known_relevance_count": len(qrels),
                "cold_latency_ms": cold_latencies[-1],
                "warm_latency_ms": warm_latencies[-1],
                **metrics,
            }
        )
        for result in cold:
            row = {
                "method": method,
                "query_id": query["query_id"],
                "product_id": result.product_id,
                "review_id": result.review_id,
                "rank": result.rank,
                "score": result.score,
                "bm25_score": getattr(result, "bm25_score", None),
                "bm25_rank": getattr(result, "bm25_rank", None),
                "dense_score": getattr(result, "dense_score", None),
                "dense_rank": getattr(result, "dense_rank", None),
                "fused_score": getattr(result, "fused_score", None),
                "fused_rank": getattr(result, "fused_rank", None),
                "reranker_score": getattr(result, "reranker_score", None),
                "final_rank": getattr(result, "final_rank", None),
                "indexed_text_normalized": result.indexed_text_normalized,
                "review_text_raw": result.review_text_raw,
                "title_raw": result.title_raw,
                "advantages_items": result.advantages_items,
                "disadvantages_items": result.disadvantages_items,
                "is_buyer": result.is_buyer,
                "recommendation_status": result.recommendation_status,
                "review_rate": result.review_rate,
                "likes": result.likes,
                "dislikes": result.dislikes,
            }
            ranked_rows.append(row)
    latency = {
        "cold_p50": median(cold_latencies),
        "cold_p95": sorted(cold_latencies)[round((len(cold_latencies) - 1) * 0.95)],
        "warm_p50": median(warm_latencies),
        "warm_p95": sorted(warm_latencies)[round((len(warm_latencies) - 1) * 0.95)],
    }
    return {"status": "available", "per_query": per_query, "metrics_at_k": _aggregate(per_query), "latency_ms": latency}, ranked_rows


def tune_rrf_on_development(settings: Settings, split: dict[str, Any]) -> dict[str, Any]:
    """Tune only RRF k on the frozen development partition, never test qrels."""
    output = _required(settings.paths.hybrid_tuning_report, "hybrid_tuning_report")
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    dense_report_path = _required(settings.paths.dense_evaluation_report, "dense_evaluation_report")
    dense_report = json.loads(dense_report_path.read_text(encoding="utf-8")) if dense_report_path.is_file() else {}
    base = {
        "schema_version": "hybrid-rrf-tuning-v1",
        "split": "development",
        "development_query_ids": split["development_query_ids"],
        "candidate_rrf_k": list(settings.hybrid.tuning_rrf_k),
        "test_query_ids_not_read_for_tuning": split["test_query_ids"],
    }
    if dense_report.get("status") != "available":
        report = {
            **base,
            "status": "unavailable",
            "reason": f"dense retrieval is unavailable: {dense_report.get('reason', 'no full BGE-M3 index')}",
            "selected_rrf_k": settings.hybrid.rrf_k,
            "selection_policy": "configured robust RRF default retained; no qrel-driven tuning was attempted",
        }
        _write_json(output, report)
        return report
    queries, labels = _queries_and_qrels(settings)
    development = _split_queries(queries, split, "development")
    bm25 = ProductScopedBM25(_required(settings.paths.retrieval_corpus, "retrieval_corpus"), settings.bm25)
    dense = DenseFaissRetriever.from_settings(settings)
    candidates: list[dict[str, Any]] = []
    for rrf_k in settings.hybrid.tuning_rrf_k:
        hybrid = HybridRRFRetriever(bm25, dense, replace(settings.hybrid, rrf_k=rrf_k))
        result, _ = _evaluate_retriever(
            method="hybrid_rrf",
            retriever=hybrid,
            queries=development,
            labels=labels,
            request_top_k=settings.hybrid.final_top_k,
            metric_k=settings.hybrid.final_top_k,
        )
        candidates.append({"rrf_k": rrf_k, "metrics_at_k": result["metrics_at_k"]})
    best = sorted(
        candidates,
        key=lambda item: (
            -(item["metrics_at_k"]["ndcg"] or 0.0),
            -(item["metrics_at_k"]["recall"] or 0.0),
            item["rrf_k"],
        ),
    )[0]
    report = {
        **base,
        "status": "available",
        "candidates": candidates,
        "selected_rrf_k": best["rrf_k"],
        "selection_policy": "maximum development NDCG@K, then Recall@K, then smaller RRF k; test labels were not read",
    }
    _write_json(output, report)
    return report


def _relative(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return (value - baseline) / baseline


def _absolute(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _storage(shared_corpus_bytes: int, additional_artifact_bytes: int) -> dict[str, int]:
    return {
        "shared_corpus_bytes": shared_corpus_bytes,
        "additional_retrieval_artifact_bytes": additional_artifact_bytes,
        "total_storage_bytes": shared_corpus_bytes + additional_artifact_bytes,
    }


def _method_unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "metrics_at_k": {"recall": None, "precision": None, "mrr": None, "ndcg": None},
        "latency_ms": None,
        "per_query": [],
        "storage_bytes": None,
        "peak_process_memory_bytes": None,
    }


def _query_analysis(
    bm25: dict[str, Any], dense: dict[str, Any], hybrid: dict[str, Any], rows: list[dict[str, Any]], split: dict[str, Any]
) -> dict[str, Any]:
    if dense["status"] != "available" or hybrid["status"] != "available":
        return {
            "status": "unavailable",
            "reason": "dense/hybrid test results are unavailable; candidate overlap and regressions cannot be measured fairly",
            "improved_queries": [], "harmed_queries": [], "ties": [], "overlap": [],
            "category_aspect_patterns": [],
        }
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_id"]), {}).setdefault(str(row["method"]), []).append(row)
    bm25_metrics = {row["query_id"]: row for row in bm25["per_query"]}
    hybrid_metrics = {row["query_id"]: row for row in hybrid["per_query"]}
    improved: list[dict[str, Any]] = []
    harmed: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    evidence_patterns: dict[str, dict[str, int]] = {}
    for query_id in split["test_query_ids"]:
        sparse_ids = {str(row["review_id"]) for row in grouped.get(query_id, {}).get("bm25", [])}
        dense_ids = {str(row["review_id"]) for row in grouped.get(query_id, {}).get("bge_m3_dense", [])}
        union = sparse_ids | dense_ids
        overlaps.append({"query_id": query_id, "jaccard_at_candidate_depth": len(sparse_ids & dense_ids) / len(union) if union else None})
        baseline = bm25_metrics[query_id].get("ndcg")
        fused = hybrid_metrics[query_id].get("ndcg")
        item = {"query_id": query_id, "bm25_ndcg": baseline, "hybrid_ndcg": fused}
        if baseline is None or fused is None or baseline == fused:
            ties.append(item)
            outcome = "tied"
        elif fused > baseline:
            improved.append(item)
            outcome = "improved"
        else:
            harmed.append(item)
            outcome = "harmed"
        evidence_type = str(bm25_metrics[query_id].get("evidence_type") or "unknown")
        pattern = evidence_patterns.setdefault(
            evidence_type, {"query_count": 0, "improved_count": 0, "harmed_count": 0, "tied_count": 0}
        )
        pattern["query_count"] += 1
        pattern[f"{outcome}_count"] += 1
    return {
        "status": "available",
        "improved_queries": improved,
        "harmed_queries": harmed,
        "ties": ties,
        "overlap": overlaps,
        "category_aspect_patterns": [
            {"evidence_type": evidence_type, **counts}
            for evidence_type, counts in sorted(evidence_patterns.items())
        ],
        "category_aspect_pattern_note": "No aspect extractor is used. These counts only group the frozen seed queries by their supplied evidence_type and are too small for general claims.",
    }


def _production_selection(settings: Settings, bm25: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    if hybrid["status"] != "available":
        return {
            "selected_method": "bm25",
            "status": "baseline_retained",
            "reason": "hybrid cannot be measured because dense retrieval is unavailable",
            "criteria": {"minimum_ndcg_gain": settings.hybrid.minimum_ndcg_gain, "maximum_warm_p95_multiplier": settings.hybrid.maximum_warm_p95_multiplier},
        }
    hybrid_ndcg = hybrid["metrics_at_k"]["ndcg"]
    bm25_ndcg = bm25["metrics_at_k"]["ndcg"]
    if hybrid_ndcg is None or bm25_ndcg is None:
        return {
            "selected_method": "bm25",
            "status": "baseline_retained",
            "reason": "hybrid quality cannot be measured on evaluable test labels",
            "criteria": {"minimum_ndcg_gain": settings.hybrid.minimum_ndcg_gain, "maximum_warm_p95_multiplier": settings.hybrid.maximum_warm_p95_multiplier},
        }
    ndcg_gain = hybrid_ndcg - bm25_ndcg
    latency_ok = hybrid["latency_ms"]["warm_p95"] <= bm25["latency_ms"]["warm_p95"] * settings.hybrid.maximum_warm_p95_multiplier
    if ndcg_gain >= settings.hybrid.minimum_ndcg_gain and latency_ok:
        return {"selected_method": "hybrid_rrf", "status": "selected", "ndcg_gain": ndcg_gain, "latency_ok": latency_ok, "reason": "quality threshold and latency guard both passed"}
    return {"selected_method": "bm25", "status": "baseline_retained", "ndcg_gain": ndcg_gain, "latency_ok": latency_ok, "reason": "hybrid did not clear both predeclared quality and latency criteria"}


def evaluate_hybrid(settings: Settings) -> dict[str, Any]:
    """Benchmark the three methods on the frozen test split, with no leakage."""
    split = freeze_hybrid_splits(settings)
    tuning = tune_rrf_on_development(settings, split)
    queries, labels = _queries_and_qrels(settings)
    test_queries = _split_queries(queries, split, "test")
    bm25_retriever = ProductScopedBM25(_required(settings.paths.retrieval_corpus, "retrieval_corpus"), settings.bm25)
    bm25, rows = _evaluate_retriever(
        method="bm25", retriever=bm25_retriever, queries=test_queries, labels=labels,
        request_top_k=settings.hybrid.bm25_candidate_depth, metric_k=settings.hybrid.final_top_k,
    )
    corpus = _required(settings.paths.retrieval_corpus, "retrieval_corpus")
    bm25["storage"] = _storage(corpus.stat().st_size, 0)
    bm25["storage_bytes"] = bm25["storage"]["total_storage_bytes"]
    bm25["peak_process_memory_bytes"] = peak_process_memory_bytes()
    dense_report_path = _required(settings.paths.dense_evaluation_report, "dense_evaluation_report")
    dense_previous = json.loads(dense_report_path.read_text(encoding="utf-8")) if dense_report_path.is_file() else {}
    if dense_previous.get("status") != "available":
        reason = f"dense retrieval unavailable: {dense_previous.get('reason', 'no full BGE-M3 index')}"
        dense = _method_unavailable(reason)
        hybrid = _method_unavailable(reason)
    else:
        dense_retriever = DenseFaissRetriever.from_settings(settings)
        dense, dense_rows = _evaluate_retriever(
            method="bge_m3_dense", retriever=dense_retriever, queries=test_queries, labels=labels,
            request_top_k=settings.hybrid.dense_candidate_depth, metric_k=settings.hybrid.final_top_k,
        )
        rows.extend(dense_rows)
        dense_manifest = dense_retriever.manifest
        dense["storage"] = _storage(corpus.stat().st_size, int(dense_manifest["index_storage_bytes"]))
        dense["storage_bytes"] = dense["storage"]["total_storage_bytes"]
        dense["peak_process_memory_bytes"] = peak_process_memory_bytes()
        selected_k = int(tuning["selected_rrf_k"])
        hybrid_retriever = HybridRRFRetriever(bm25_retriever, dense_retriever, replace(settings.hybrid, rrf_k=selected_k))
        hybrid, hybrid_rows = _evaluate_retriever(
            method="hybrid_rrf", retriever=hybrid_retriever, queries=test_queries, labels=labels,
            request_top_k=settings.hybrid.final_top_k, metric_k=settings.hybrid.final_top_k,
        )
        rows.extend(hybrid_rows)
        hybrid["storage"] = _storage(corpus.stat().st_size, int(dense_manifest["index_storage_bytes"]))
        hybrid["storage_bytes"] = hybrid["storage"]["total_storage_bytes"]
        hybrid["peak_process_memory_bytes"] = peak_process_memory_bytes()
    relative = {
        method: {
            metric: {
                "absolute": _absolute(result["metrics_at_k"][metric], bm25["metrics_at_k"][metric]),
                "relative": _relative(result["metrics_at_k"][metric], bm25["metrics_at_k"][metric]),
            }
            for metric in ("recall", "precision", "mrr", "ndcg")
        }
        for method, result in {"bge_m3_dense": dense, "hybrid_rrf": hybrid}.items()
    }
    if rows:
        output = _required(settings.paths.hybrid_ranked_results, "hybrid_ranked_results")
        output.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows, schema=_RANKED_RESULT_SCHEMA).write_parquet(output)
    analysis = _query_analysis(bm25, dense, hybrid, rows, split)
    _write_json(_required(settings.paths.hybrid_analysis_report, "hybrid_analysis_report"), analysis)
    selection = _production_selection(settings, bm25, hybrid)
    _write_json(_required(settings.paths.hybrid_selection_report, "hybrid_selection_report"), selection)
    report = {
        "status": "available" if dense["status"] == "available" else "dense_and_hybrid_unavailable",
        "corpus_version": f"{settings.dataset.revision}:bm25-corpus-v1",
        "split": split,
        "tuning": tuning,
        "configuration": {
            "fusion": "reciprocal_rank_fusion",
            "bm25_candidate_depth": settings.hybrid.bm25_candidate_depth,
            "dense_candidate_depth": settings.hybrid.dense_candidate_depth,
            "rrf_k": tuning["selected_rrf_k"],
            "final_top_k": settings.hybrid.final_top_k,
        },
        "methods": {"bm25": bm25, "bge_m3_dense": dense, "hybrid_rrf": hybrid},
        "improvement_over_bm25": relative,
        "ranked_results_path": str(_required(settings.paths.hybrid_ranked_results, "hybrid_ranked_results")),
        "production_selection": selection,
    }
    _write_json(_required(settings.paths.hybrid_evaluation_report, "hybrid_evaluation_report"), report)
    return report
