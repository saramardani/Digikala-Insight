"""Leakage-safe BGE reranker tuning and four-way retrieval benchmark."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import polars as pl

from .bm25 import ProductScopedBM25
from .config import Settings
from .dense_index import DenseFaissRetriever
from .dense_embedding import runtime_resources
from .hybrid import HybridRRFRetriever
from .hybrid_evaluation import (
    _RANKED_RESULT_SCHEMA,
    _absolute,
    _queries_and_qrels,
    _relative,
    _required,
    _split_queries,
    _storage,
    freeze_hybrid_splits,
    evaluate_hybrid,
)
from .reranker import (
    BgeRerankerV2M3,
    HybridBgeReranker,
    RerankedReview,
    RerankerScorer,
    RerankerUnavailableError,
    write_reranker_resource_report,
)
from .retrieval_metrics import retrieval_metrics_at_k
from .runtime import peak_process_memory_bytes


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _aggregate(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _development_queries_and_labels(
    settings: Settings, split: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Load only development qrels; test relevance grades stay unobserved."""
    query_path = _required(settings.paths.retrieval_queries, "retrieval_queries")
    qrels_path = _required(settings.paths.retrieval_qrels, "retrieval_qrels")
    development_ids = set(str(value) for value in split["development_query_ids"])
    queries = [
        json.loads(line)
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line and str(json.loads(line)["query_id"]) in development_ids
    ]
    rows = (
        pl.scan_csv(qrels_path)
        .filter(pl.col("query_id").cast(pl.String).is_in(sorted(development_ids)))
        .select("query_id", "review_id", "relevance_grade")
        .collect()
        .to_dicts()
    )
    labels: dict[str, dict[str, int]] = {}
    for row in rows:
        labels.setdefault(str(row["query_id"]), {})[str(row["review_id"])] = int(row["relevance_grade"])
    return queries, labels


def _ranked_row(method: str, query_id: str, result: RerankedReview) -> dict[str, Any]:
    return {
        "method": method,
        "query_id": query_id,
        "product_id": result.product_id,
        "review_id": result.review_id,
        "rank": result.rank,
        "score": result.score,
        "bm25_score": result.bm25_score,
        "bm25_rank": result.bm25_rank,
        "dense_score": result.dense_score,
        "dense_rank": result.dense_rank,
        "fused_score": result.fused_score,
        "fused_rank": result.fused_rank,
        "reranker_score": result.reranker_score,
        "final_rank": result.final_rank,
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


def _evaluate_reranker(
    *,
    retriever: HybridBgeReranker,
    queries: list[dict[str, Any]],
    labels: dict[str, dict[str, int]],
    metric_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure end-to-end and reranker-only latency using the same candidates."""
    cold_end_to_end: list[float] = []
    warm_end_to_end: list[float] = []
    cold_reranker: list[float] = []
    warm_reranker: list[float] = []
    warm_candidate_counts: list[int] = []
    rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for query in queries:
        started = perf_counter()
        cold = retriever.retrieve(query["product_id"], query["query"], metric_k)
        cold_end_to_end.append((perf_counter() - started) * 1000)
        cold_reranker.append(float(retriever.last_reranker_latency_ms or 0.0))
        started = perf_counter()
        warm = retriever.retrieve(query["product_id"], query["query"], metric_k)
        warm_end_to_end.append((perf_counter() - started) * 1000)
        warm_reranker.append(float(retriever.last_reranker_latency_ms or 0.0))
        warm_candidate_counts.append(retriever.last_candidate_count)
        if any(str(result.product_id) != str(query["product_id"]) for result in warm):
            raise RuntimeError("reranker benchmark detected a cross-product result")
        qrels = labels.get(str(query["query_id"]), {})
        final_metrics = retrieval_metrics_at_k([result.review_id for result in warm], qrels, metric_k)
        candidate_metrics = retrieval_metrics_at_k(
            retriever.last_candidate_review_ids, qrels, retriever.settings.candidate_depth
        )
        per_query.append(
            {
                "query_id": query["query_id"],
                "product_id": query["product_id"],
                "category": query.get("category"),
                "evidence_type": query.get("evidence_type"),
                "known_relevance_count": len(qrels),
                "candidate_count": retriever.last_candidate_count,
                "candidate_recall_at_n": candidate_metrics["recall"],
                "cold_end_to_end_latency_ms": cold_end_to_end[-1],
                "warm_end_to_end_latency_ms": warm_end_to_end[-1],
                "cold_reranker_latency_ms": cold_reranker[-1],
                "warm_reranker_latency_ms": warm_reranker[-1],
                **final_metrics,
            }
        )
        rows.extend(_ranked_row("hybrid_bge_reranker", str(query["query_id"]), result) for result in cold)
    evaluable = [row for row in per_query if row["known_relevance_count"] > 0]
    warm_reranker_seconds = sum(warm_reranker) / 1000
    return (
        {
            "status": "available",
            "per_query": per_query,
            "metrics_at_k": {
                metric: _aggregate(evaluable, metric) for metric in ("recall", "precision", "mrr", "ndcg")
            },
            "candidate_recall_at_n": _aggregate(evaluable, "candidate_recall_at_n"),
            "latency_ms": {
                "end_to_end_cold_p50": median(cold_end_to_end),
                "end_to_end_cold_p95": _percentile(cold_end_to_end, 0.95),
                "end_to_end_warm_p50": median(warm_end_to_end),
                "end_to_end_warm_p95": _percentile(warm_end_to_end, 0.95),
                "reranker_only_cold_p50": median(cold_reranker),
                "reranker_only_cold_p95": _percentile(cold_reranker, 0.95),
                "reranker_only_warm_p50": median(warm_reranker),
                "reranker_only_warm_p95": _percentile(warm_reranker, 0.95),
            },
            "throughput_candidates_per_second": (
                sum(warm_candidate_counts) / warm_reranker_seconds if warm_reranker_seconds else None
            ),
        },
        rows,
    )


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "metrics_at_k": {"recall": None, "precision": None, "mrr": None, "ndcg": None},
        "candidate_recall_at_n": None,
        "latency_ms": None,
        "throughput_candidates_per_second": None,
        "per_query": [],
        "storage_bytes": None,
        "peak_process_memory_bytes": None,
    }


def tune_reranker_candidate_depth(
    settings: Settings,
    split: dict[str, Any],
    hybrid: HybridRRFRetriever,
    scorer: RerankerScorer,
) -> dict[str, Any]:
    """Select candidate depth using only development query IDs and qrels."""
    output = _required(settings.paths.reranker_tuning_report, "reranker_tuning_report")
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "available":
            return existing
    queries, labels = _development_queries_and_labels(settings, split)
    candidates: list[dict[str, Any]] = []
    for depth in settings.reranker.candidate_depths:
        reranker = HybridBgeReranker(hybrid, scorer, replace(settings.reranker, candidate_depth=depth))
        result, _ = _evaluate_reranker(
            retriever=reranker,
            queries=queries,
            labels=labels,
            metric_k=settings.reranker.final_top_k,
        )
        candidates.append(
            {
                "candidate_depth": depth,
                "metrics_at_k": result["metrics_at_k"],
                "candidate_recall_at_n": result["candidate_recall_at_n"],
                "latency_ms": result["latency_ms"],
                "throughput_candidates_per_second": result["throughput_candidates_per_second"],
            }
        )
    best = sorted(
        candidates,
        key=lambda item: (
            -(item["metrics_at_k"]["ndcg"] or 0.0),
            -(item["metrics_at_k"]["mrr"] or 0.0),
            item["candidate_depth"],
            item["latency_ms"]["reranker_only_warm_p95"] or float("inf"),
        ),
    )[0]
    report = {
        "schema_version": "bge-reranker-depth-tuning-v1",
        "status": "available",
        "split": "development",
        "development_query_ids": split["development_query_ids"],
        "test_query_ids_not_loaded_for_tuning": split["test_query_ids"],
        "candidate_depths": list(settings.reranker.candidate_depths),
        "candidates": candidates,
        "selected_candidate_depth": best["candidate_depth"],
        "selection_policy": "maximum development NDCG@K, then MRR, then smaller candidate depth as the deterministic latency/cost guard, then lower reranker-only warm p95",
        "final_top_k": settings.reranker.final_top_k,
    }
    _write_json(output, report)
    return report


def _bootstrap_interval(values: list[float], iterations: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"sample_size": 0, "mean_delta": None, "ci_95": None, "iterations": iterations, "seed": seed}
    random_source = random.Random(seed)
    samples = [
        sum(random_source.choice(values) for _ in values) / len(values)
        for _ in range(iterations)
    ]
    ordered = sorted(samples)
    return {
        "sample_size": len(values),
        "mean_delta": sum(values) / len(values),
        "ci_95": [ordered[int((iterations - 1) * 0.025)], ordered[int((iterations - 1) * 0.975)]],
        "iterations": iterations,
        "seed": seed,
    }


def _reranker_analysis(
    settings: Settings,
    hybrid: dict[str, Any],
    reranker: dict[str, Any],
) -> dict[str, Any]:
    if hybrid["status"] != "available" or reranker["status"] != "available":
        return {
            "status": "unavailable",
            "reason": "hybrid/reranker test results are unavailable, so deltas and uncertainty cannot be measured fairly",
            "per_query_deltas": [],
            "bootstrap": {},
        }
    hybrid_by_id = {str(row["query_id"]): row for row in hybrid["per_query"]}
    reranker_by_id = {str(row["query_id"]): row for row in reranker["per_query"]}
    deltas = []
    for query_id in sorted(hybrid_by_id):
        before, after = hybrid_by_id[query_id], reranker_by_id[query_id]
        deltas.append(
            {
                "query_id": query_id,
                "hybrid_ndcg": before["ndcg"],
                "reranker_ndcg": after["ndcg"],
                "ndcg_delta": _absolute(after["ndcg"], before["ndcg"]),
                "hybrid_mrr": before["mrr"],
                "reranker_mrr": after["mrr"],
                "mrr_delta": _absolute(after["mrr"], before["mrr"]),
                "candidate_recall_at_n": after["candidate_recall_at_n"],
            }
        )
    return {
        "status": "available",
        "per_query_deltas": deltas,
        "bootstrap": {
            "ndcg_delta": _bootstrap_interval(
                [float(row["ndcg_delta"]) for row in deltas if row["ndcg_delta"] is not None],
                settings.reranker.bootstrap_iterations,
                settings.random_seed,
            ),
            "mrr_delta": _bootstrap_interval(
                [float(row["mrr_delta"]) for row in deltas if row["mrr_delta"] is not None],
                settings.reranker.bootstrap_iterations,
                settings.random_seed + 1,
            ),
        },
        "interpretation": "The frozen seed benchmark is small; confidence intervals are descriptive and do not justify strong claims without human-reviewed qrels.",
    }


def _first(rows: list[dict[str, Any]], method: str, query_id: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["method"] == method and str(row["query_id"]) == query_id]
    return min(candidates, key=lambda row: int(row["rank"])) if candidates else None


def _failure_analysis(
    settings: Settings,
    labels: dict[str, dict[str, int]],
    rows: list[dict[str, Any]],
    reranker: dict[str, Any],
) -> dict[str, Any]:
    if reranker["status"] != "available":
        return {
            "status": "unavailable",
            "reason": "reranker test results are unavailable, so counterfactual failure examples cannot be identified fairly",
            "corrects_lexical_false_positives": [],
            "corrects_dense_semantic_false_positives": [],
            "harms_previously_good_hybrid_ranking": [],
            "long_or_noisy_text_cases_requiring_human_review": [],
        }
    groups: dict[str, list[dict[str, Any]]] = {
        "corrects_lexical_false_positives": [],
        "corrects_dense_semantic_false_positives": [],
        "harms_previously_good_hybrid_ranking": [],
        "long_or_noisy_text_cases_requiring_human_review": [],
    }
    for query in reranker["per_query"]:
        query_id = str(query["query_id"])
        known = set(labels.get(query_id, {}))
        if not known:
            continue
        bm25_top = _first(rows, "bm25", query_id)
        dense_top = _first(rows, "bge_m3_dense", query_id)
        hybrid_top = _first(rows, "hybrid_rrf", query_id)
        reranked_top = _first(rows, "hybrid_bge_reranker", query_id)
        if reranked_top is None:
            continue
        common = {"query_id": query_id, "reranker_top": reranked_top, "known_relevance_count": len(known)}
        if bm25_top and str(bm25_top["review_id"]) not in known and str(reranked_top["review_id"]) in known:
            groups["corrects_lexical_false_positives"].append({**common, "bm25_top": bm25_top})
        if dense_top and str(dense_top["review_id"]) not in known and str(reranked_top["review_id"]) in known:
            groups["corrects_dense_semantic_false_positives"].append({**common, "dense_top": dense_top})
        if hybrid_top and str(hybrid_top["review_id"]) in known and str(reranked_top["review_id"]) not in known:
            groups["harms_previously_good_hybrid_ranking"].append({**common, "hybrid_top": hybrid_top})
        text = str(reranked_top.get("indexed_text_normalized") or "")
        if str(reranked_top["review_id"]) not in known and len(text) >= settings.reranker.long_text_character_threshold:
            groups["long_or_noisy_text_cases_requiring_human_review"].append(
                {
                    **common,
                    "normalized_text_length": len(text),
                    "judgment": "long-text association only; qrels do not establish that length/noise caused the failure",
                }
            )
    return {"status": "available", **groups}


def _production_selection(settings: Settings, methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bm25 = methods["bm25"]
    if bm25["status"] != "available" or bm25["metrics_at_k"]["ndcg"] is None:
        return {"selected_method": "bm25", "status": "baseline_retained", "reason": "BM25 test quality is not measurable"}
    baseline_ndcg = bm25["metrics_at_k"]["ndcg"]
    complexity = {"bm25": 0, "bge_m3_dense": 1, "hybrid_rrf": 2, "hybrid_bge_reranker": 3}
    qualified: list[tuple[str, dict[str, Any]]] = []
    rejected: dict[str, str] = {}
    for name, method in methods.items():
        if name == "bm25":
            continue
        if method["status"] != "available" or method["metrics_at_k"]["ndcg"] is None:
            rejected[name] = method.get("reason", "unavailable")
            continue
        latency = method.get("latency_ms") or {}
        warm_p95 = latency.get("end_to_end_warm_p95", latency.get("warm_p95"))
        gain = method["metrics_at_k"]["ndcg"] - baseline_ndcg
        if gain < settings.reranker.minimum_ndcg_gain:
            rejected[name] = f"NDCG gain {gain:.6f} is below {settings.reranker.minimum_ndcg_gain:.6f}"
        elif warm_p95 is None or warm_p95 > settings.reranker.maximum_warm_p95_ms:
            rejected[name] = f"warm p95 {warm_p95} ms exceeds {settings.reranker.maximum_warm_p95_ms} ms"
        else:
            qualified.append((name, method))
    if not qualified:
        return {
            "selected_method": "bm25",
            "status": "baseline_retained",
            "reason": "no more complex available method passed both predeclared quality and latency thresholds",
            "rejected": rejected,
            "criteria": {
                "minimum_ndcg_gain": settings.reranker.minimum_ndcg_gain,
                "maximum_warm_p95_ms": settings.reranker.maximum_warm_p95_ms,
            },
        }
    selected_name, selected = sorted(
        qualified,
        key=lambda item: (
            -item[1]["metrics_at_k"]["ndcg"],
            -item[1]["metrics_at_k"]["mrr"],
            complexity[item[0]],
        ),
    )[0]
    return {
        "selected_method": selected_name,
        "status": "selected",
        "ndcg_gain_over_bm25": selected["metrics_at_k"]["ndcg"] - baseline_ndcg,
        "reason": "highest measured NDCG among methods that met the predeclared quality and absolute latency criteria; simpler method wins exact metric ties",
        "rejected": rejected,
        "criteria": {
            "minimum_ndcg_gain": settings.reranker.minimum_ndcg_gain,
            "maximum_warm_p95_ms": settings.reranker.maximum_warm_p95_ms,
        },
    }


def _normalise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{name: row.get(name) for name in _RANKED_RESULT_SCHEMA} for row in rows]


def _persist(
    settings: Settings,
    report: dict[str, Any],
    analysis: dict[str, Any],
    failures: dict[str, Any],
    selection: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked_path = _required(settings.paths.reranker_ranked_results, "reranker_ranked_results")
    if rows:
        ranked_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(_normalise_rows(rows), schema=_RANKED_RESULT_SCHEMA).write_parquet(ranked_path)
    report["ranked_results_path"] = str(ranked_path)
    _write_json(_required(settings.paths.reranker_analysis_report, "reranker_analysis_report"), analysis)
    _write_json(_required(settings.paths.reranker_failure_analysis, "reranker_failure_analysis"), failures)
    _write_json(_required(settings.paths.reranker_selection_report, "reranker_selection_report"), selection)
    _write_json(_required(settings.paths.reranker_evaluation_report, "reranker_evaluation_report"), report)
    return report


def _base_rows(settings: Settings) -> list[dict[str, Any]]:
    path = _required(settings.paths.hybrid_ranked_results, "hybrid_ranked_results")
    return pl.read_parquet(path).to_dicts() if path.is_file() else []


def evaluate_reranker(settings: Settings) -> dict[str, Any]:
    """Run the frozen four-way benchmark; no reranker is instantiated without prerequisites."""
    split = freeze_hybrid_splits(settings)
    # This regenerates BM25/dense/hybrid on the same frozen Phase 6 test IDs.
    base = evaluate_hybrid(settings)
    methods = dict(base["methods"])
    resource = write_reranker_resource_report(settings)
    queries, labels = _queries_and_qrels(settings)
    test_queries = _split_queries(queries, split, "test")
    rows = _base_rows(settings)
    reranker: dict[str, Any]
    tuning: dict[str, Any]
    reason: str | None = None
    if methods["hybrid_rrf"]["status"] != "available":
        reason = f"hybrid candidate flow unavailable: {methods['hybrid_rrf'].get('reason', 'dense retrieval unavailable')}"
    elif resource["status"] != "ready":
        reason = f"BGE reranker preflight unavailable: {resource['reason']}"
    if reason is not None:
        reranker = _unavailable(reason)
        tuning = {
            "schema_version": "bge-reranker-depth-tuning-v1",
            "status": "unavailable",
            "reason": reason,
            "development_query_ids": split["development_query_ids"],
            "test_query_ids_not_loaded_for_tuning": split["test_query_ids"],
        }
        _write_json(_required(settings.paths.reranker_tuning_report, "reranker_tuning_report"), tuning)
    else:
        try:
            bm25 = ProductScopedBM25(_required(settings.paths.retrieval_corpus, "retrieval_corpus"), settings.bm25)
            dense = DenseFaissRetriever.from_settings(settings)
            selected_rrf_k = int(base["tuning"]["selected_rrf_k"])
            hybrid = HybridRRFRetriever(bm25, dense, replace(settings.hybrid, rrf_k=selected_rrf_k))
            scorer = BgeRerankerV2M3(settings.reranker, _required(settings.paths.reranker_cache_root, "reranker_cache_root"))
            tuning = tune_reranker_candidate_depth(settings, split, hybrid, scorer)
            selected_depth = int(tuning["selected_candidate_depth"])
            retriever = HybridBgeReranker(hybrid, scorer, replace(settings.reranker, candidate_depth=selected_depth))
            reranker, reranker_rows = _evaluate_reranker(
                retriever=retriever,
                queries=test_queries,
                labels=labels,
                metric_k=settings.reranker.final_top_k,
            )
            rows.extend(reranker_rows)
            cache = _required(settings.paths.reranker_cache_root, "reranker_cache_root")
            model_storage = sum(item.stat().st_size for item in cache.rglob("*") if item.is_file())
            hybrid_storage = methods["hybrid_rrf"].get("storage", {})
            reranker["storage"] = {
                **hybrid_storage,
                "reranker_model_storage_bytes": model_storage,
                "total_storage_bytes": int(hybrid_storage.get("total_storage_bytes", 0)) + model_storage,
            }
            reranker["storage_bytes"] = reranker["storage"]["total_storage_bytes"]
            reranker["peak_process_memory_bytes"] = peak_process_memory_bytes()
            reranker["resources"] = runtime_resources()
            reranker["model"] = scorer.metadata()
        except (FileNotFoundError, RerankerUnavailableError, RuntimeError, ValueError) as error:
            reason = f"BGE reranker unavailable: {error}"
            reranker = _unavailable(reason)
            tuning = {
                "schema_version": "bge-reranker-depth-tuning-v1",
                "status": "unavailable",
                "reason": reason,
                "development_query_ids": split["development_query_ids"],
                "test_query_ids_not_loaded_for_tuning": split["test_query_ids"],
            }
            _write_json(_required(settings.paths.reranker_tuning_report, "reranker_tuning_report"), tuning)
    methods["hybrid_bge_reranker"] = reranker
    analysis = _reranker_analysis(settings, methods["hybrid_rrf"], reranker)
    failures = _failure_analysis(settings, labels, rows, reranker)
    selection = _production_selection(settings, methods)
    selection["frozen_configuration"] = {
        "corpus_version": base["corpus_version"],
        "query_and_qrels_hashes": {
            "queries_sha256": split["queries_sha256"],
            "qrels_sha256": split["qrels_sha256"],
        },
        "random_seed": settings.random_seed,
        "hybrid_rrf_k": base["configuration"]["rrf_k"],
        "reranker_model_id": settings.reranker.model_id,
        "reranker_model_revision": settings.reranker.model_revision,
        "reranker_candidate_depth": tuning.get("selected_candidate_depth", settings.reranker.candidate_depth),
        "final_top_k": settings.reranker.final_top_k,
    }
    improvement = {
        name: {
            metric: {
                "absolute": _absolute(method["metrics_at_k"][metric], methods["bm25"]["metrics_at_k"][metric]),
                "relative": _relative(method["metrics_at_k"][metric], methods["bm25"]["metrics_at_k"][metric]),
            }
            for metric in ("recall", "precision", "mrr", "ndcg")
        }
        for name, method in methods.items()
        if name != "bm25"
    }
    report = {
        "schema_version": "four-way-retrieval-benchmark-v1",
        "status": "available" if reranker["status"] == "available" else "reranker_unavailable",
        "corpus_version": base["corpus_version"],
        "split": split,
        "configuration": {
            "final_top_k": settings.reranker.final_top_k,
            "hybrid": base["configuration"],
            "reranker": {
                "model_id": settings.reranker.model_id,
                "model_revision": settings.reranker.model_revision,
                "candidate_depth": tuning.get("selected_candidate_depth", settings.reranker.candidate_depth),
                "candidate_depths_tuned_on_development": list(settings.reranker.candidate_depths),
                "device": settings.reranker.device,
                "dtype": "float16" if settings.reranker.use_fp16 else "float32",
                "batch_size": settings.reranker.batch_size,
                "max_length": settings.reranker.max_length,
                "query_max_length": settings.reranker.query_max_length,
                "reranker_input": "indexed_text_normalized only; it is the frozen composed normalized review evidence, not product aggregate statistics",
            },
        },
        "resource_preflight": resource,
        "tuning": tuning,
        "methods": methods,
        "improvement_over_bm25": improvement,
        "production_selection": selection,
        "metric_note": "Recall@K is final-evidence recall. candidate_recall_at_n is reported separately for the reranker because reranking changes order, not candidate membership.",
        "label_limitations": "The frozen qrels are deterministic lexical seed candidates pending human review; differences are reproducibility measurements, not final relevance claims.",
    }
    return _persist(settings, report, analysis, failures, selection, rows)
