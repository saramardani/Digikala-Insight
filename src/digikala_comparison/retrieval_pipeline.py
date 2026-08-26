"""Phase 4 corpus, frozen benchmark, and BM25 evaluation orchestration."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import polars as pl

from .bm25 import ProductScopedBM25
from .config import Settings
from .retrieval_metrics import retrieval_metrics_at_k
from .retrieval_text import compose_retrieval_text, tokenize_persian_lexical
from .runtime import peak_process_memory_bytes


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _scalar(frame: pl.LazyFrame, expression: pl.Expr) -> int:
    return int(frame.select(expression).collect()[0, 0])


def unique_retrieval_documents(documents: pl.LazyFrame) -> pl.LazyFrame:
    """Keep one deterministic document for each preserved review identifier.

    The pinned source contains repeated review records.  BM25 documents must
    nevertheless have a one-to-one relationship with ``review_id`` so that
    retrieval evidence and qrels remain unambiguous.  ``keep='first'`` refers
    to the stable row order in the canonical source and is recorded in the
    corpus report rather than silently treating the rows as distinct reviews.
    """
    return documents.unique(subset=["review_id"], keep="first", maintain_order=True)


def retrieval_eligibility_expression(minimum_normalized_body_length: int) -> pl.Expr:
    """Explicit, reusable retrieval-document eligibility rule."""
    valid_review_id = pl.col("review_id").cast(pl.String).str.strip_chars().ne("").fill_null(False)
    valid_product_id = pl.col("product_id").cast(pl.String).str.strip_chars().ne("").fill_null(False)
    useful_body = pl.col("review_text_normalized").is_not_null()
    long_enough = pl.col("review_text_normalized").str.len_chars() >= minimum_normalized_body_length
    return valid_review_id & valid_product_id & useful_body & long_enough & (pl.col("bm25_tokens").list.len() > 0)


def build_retrieval_corpus(settings: Settings) -> dict[str, Path]:
    """Persist eligible review documents; no global satisfaction is calculated here."""
    paths = settings.paths
    if paths.retrieval_corpus is None or paths.retrieval_corpus_report is None:
        raise ValueError("retrieval corpus paths must be configured")
    if not paths.processed_comments.is_file():
        raise FileNotFoundError("comments.parquet is required. Run digikala-preprocess first.")
    started = perf_counter()
    comments = pl.scan_parquet(paths.processed_comments)
    valid_review_id = pl.col("review_id").cast(pl.String).str.strip_chars().ne("").fill_null(False)
    valid_product_id = pl.col("product_id").cast(pl.String).str.strip_chars().ne("").fill_null(False)
    useful_body = pl.col("review_text_normalized").is_not_null()
    long_enough = pl.col("review_text_normalized").str.len_chars() >= settings.bm25.minimum_normalized_text_length
    composed = pl.struct(
        ["title_normalized", "review_text_normalized", "advantages_items", "disadvantages_items"]
    ).map_elements(compose_retrieval_text, return_dtype=pl.String).alias("indexed_text_normalized")
    staged = comments.with_columns(composed).with_columns(
        pl.col("indexed_text_normalized")
        .map_elements(tokenize_persian_lexical, return_dtype=pl.List(pl.String))
        .alias("bm25_tokens")
    )
    eligible = retrieval_eligibility_expression(settings.bm25.minimum_normalized_text_length)
    # Count raw eligibility failures in one scan. The remaining exclusion is
    # derived after writing the corpus, avoiding repeated Python tokenization.
    quality_counts = comments.select(
        [
            pl.len().alias("total"),
            (~valid_review_id).sum().alias("invalid_review_id"),
            (valid_review_id & ~valid_product_id).sum().alias("invalid_product_id"),
            (valid_review_id & valid_product_id & ~useful_body).sum().alias("missing_normalized_body"),
            (valid_review_id & valid_product_id & useful_body & ~long_enough).sum().alias("below_minimum_text_length"),
        ]
    ).collect().to_dicts()[0]
    total = int(quality_counts.pop("total"))
    duplicate_review_ids = comments.group_by("review_id").len().filter(pl.col("len") > 1)
    source_duplicate_review_id_rows = _scalar(
        duplicate_review_ids, (pl.col("len") - 1).sum().fill_null(0)
    )
    corpus_columns = [
        "review_id", "product_id", "indexed_text_normalized", "bm25_tokens",
        "review_text_raw", "title_raw", "advantages_items", "disadvantages_items",
        "is_buyer_bool", "recommendation_status", "review_rate_numeric", "likes_numeric", "dislikes_numeric",
    ]
    paths.retrieval_corpus.parent.mkdir(parents=True, exist_ok=True)
    unique_retrieval_documents(staged.filter(eligible).select(corpus_columns)).sink_parquet(
        paths.retrieval_corpus
    )
    eligible_count = _scalar(pl.scan_parquet(paths.retrieval_corpus), pl.len())
    # Evaluate eligibility only for duplicate-ID groups, not by re-running the
    # expensive tokenizer over the full corpus.  This separates a genuine lack
    # of lexical evidence from a deliberate one-document-per-review-id choice.
    duplicate_staged = (
        comments.join(duplicate_review_ids.select("review_id"), on="review_id", how="inner")
        .with_columns(composed)
        .with_columns(
            pl.col("indexed_text_normalized")
            .map_elements(tokenize_persian_lexical, return_dtype=pl.List(pl.String))
            .alias("bm25_tokens")
        )
        .filter(eligible)
    )
    duplicate_eligible = duplicate_staged.select(
        [pl.len().alias("rows"), pl.col("review_id").n_unique().alias("unique_review_ids")]
    ).collect().to_dicts()[0]
    duplicate_review_id_rows_removed = (
        int(duplicate_eligible["rows"]) - int(duplicate_eligible["unique_review_ids"])
    )
    exclusions = {key: int(value) for key, value in quality_counts.items()}
    exclusions["no_lexical_tokens_after_composition"] = (
        total - eligible_count - sum(exclusions.values()) - duplicate_review_id_rows_removed
    )
    exclusions["duplicate_review_id_rows_removed"] = duplicate_review_id_rows_removed
    report = {
        "dataset": {"revision": settings.dataset.revision, "repository": settings.dataset.repository},
        "corpus_version": f"{settings.dataset.revision}:bm25-corpus-v1",
        "document_unit": "one eligible canonical review",
        "total_canonical_reviews": total,
        "eligible_retrieval_reviews": eligible_count,
        "excluded_reviews_by_first_failure_reason": exclusions,
        "source_duplicate_review_id_rows": source_duplicate_review_id_rows,
        "duplicate_review_id_policy": (
            "one document per review_id; retain the first stable canonical-source row "
            "and report source duplicates explicitly"
        ),
        "eligibility_policy": {
            "valid_review_id": True, "valid_product_id": True,
            "minimum_normalized_body_length": settings.bm25.minimum_normalized_text_length,
            "buyer_only": False,
        },
        "field_composition_policy": "title, body, advantages, and disadvantages are each included at most once; duplicate field values are removed before tokenization",
        "indexed_fields": ["title_normalized", "review_text_normalized", "advantages_items", "disadvantages_items"],
        "tokenization": "Unicode/Persian-character normalization, digit normalization, whitespace/punctuation splitting; no stemming and no stopword removal",
        "output_path": str(paths.retrieval_corpus),
        "output_size_bytes": paths.retrieval_corpus.stat().st_size,
        "performance": {"runtime_seconds": perf_counter() - started, "peak_process_memory_bytes": peak_process_memory_bytes()},
    }
    _write_json(paths.retrieval_corpus_report, report)
    return {"retrieval_corpus": paths.retrieval_corpus, "retrieval_corpus_report": paths.retrieval_corpus_report}


def create_frozen_benchmark(settings: Settings, force: bool = False) -> dict[str, int]:
    """Create a seed benchmark, explicitly marked as pending human qrels review."""
    paths = settings.paths
    required = (paths.retrieval_corpus, paths.retrieval_queries, paths.retrieval_qrels, paths.canonical_products)
    if any(path is None for path in required):
        raise ValueError("retrieval benchmark paths must be configured")
    if not force and paths.retrieval_queries.is_file() and paths.retrieval_qrels.is_file():
        return {"query_count": sum(1 for _ in paths.retrieval_queries.open(encoding="utf-8")), "qrels_count": pl.read_csv(paths.retrieval_qrels).height}
    counts = pl.scan_parquet(paths.retrieval_corpus).group_by("product_id").len(name="review_count")
    products = pl.scan_parquet(paths.canonical_products).select(["product_id", "Category1", "title_fa"])
    high = (
        counts.join(products, on="product_id", how="left")
        .filter(pl.col("review_count") >= 20)
        .sort("review_count", descending=True)
        .group_by("Category1", maintain_order=True)
        .agg([pl.col("product_id").first(), pl.col("title_fa").first(), pl.col("review_count").first()])
        .head(5).collect().to_dicts()
    )
    low = counts.filter(pl.col("review_count") == 1).sort("product_id").head(1).collect().to_dicts()
    query_texts = ["کیفیت ساخت", "مشکلات و ایرادها", "رضایت از خرید", "دوام", "ارزش خرید"]
    queries: list[dict[str, Any]] = []
    for index, row in enumerate(high, start=1):
        queries.append({"query_id": f"bm25_seed_{index:02d}", "product_id": str(row["product_id"]), "query": query_texts[index - 1], "category": row["Category1"], "evidence_type": ["attribute", "negative", "satisfaction", "attribute", "value"][index - 1], "split": "development", "annotation_status": "seed_lexical_labels_pending_human_review"})
    if low:
        queries.append({"query_id": "bm25_seed_low_evidence", "product_id": str(low[0]["product_id"]), "query": "کیفیت", "category": None, "evidence_type": "low_evidence", "split": "development", "annotation_status": "intentionally_unlabeled_low_evidence"})
    qrels: list[dict[str, Any]] = []
    corpus = pl.scan_parquet(paths.retrieval_corpus)
    for query in queries:
        if query["annotation_status"].startswith("intentionally"):
            continue
        tokens = tokenize_persian_lexical(query["query"])
        # These are deliberately broad seed candidates for subsequent human
        # judgment, not claims of final relevance ground truth.
        condition = pl.any_horizontal([pl.col("bm25_tokens").list.contains(token) for token in tokens])
        evidence = corpus.filter((pl.col("product_id") == query["product_id"]) & condition).sort("review_id").head(3).collect()
        for row in evidence.iter_rows(named=True):
            qrels.append({"query_id": query["query_id"], "product_id": query["product_id"], "review_id": str(row["review_id"]), "relevance_grade": 1, "judgment_source": "deterministic_lexical_seed_pending_human_review"})
    paths.retrieval_queries.parent.mkdir(parents=True, exist_ok=True)
    paths.retrieval_queries.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queries), encoding="utf-8")
    pl.DataFrame(
        qrels,
        schema={"query_id": pl.String, "product_id": pl.String, "review_id": pl.String, "relevance_grade": pl.Int64, "judgment_source": pl.String},
    ).unique(subset=["query_id", "review_id"], keep="first").sort(["query_id", "review_id"]).write_csv(paths.retrieval_qrels)
    return {"query_count": len(queries), "qrels_count": len(qrels)}


def evaluate_bm25(settings: Settings) -> dict[str, Any]:
    paths = settings.paths
    required = (paths.retrieval_corpus, paths.retrieval_queries, paths.retrieval_qrels, paths.retrieval_results, paths.retrieval_evaluation_report)
    if any(path is None for path in required):
        raise ValueError("retrieval evaluation paths must be configured")
    if not paths.retrieval_queries.is_file() or not paths.retrieval_qrels.is_file():
        create_frozen_benchmark(settings)
    queries = [json.loads(line) for line in paths.retrieval_queries.read_text(encoding="utf-8").splitlines() if line]
    qrels_rows = pl.read_csv(paths.retrieval_qrels).to_dicts()
    qrels_by_query: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qrels_by_query.setdefault(row["query_id"], {})[str(row["review_id"])] = int(row["relevance_grade"])
    started = perf_counter()
    bm25 = ProductScopedBM25(paths.retrieval_corpus, settings.bm25)
    result_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    warm_latencies: list[float] = []
    cold_latencies: list[float] = []
    index_build_times: list[float] = []
    for query in queries:
        cold, cold_ms = bm25.timed_retrieve(query["product_id"], query["query"], settings.bm25.candidate_depth)
        warm, warm_ms = bm25.timed_retrieve(query["product_id"], query["query"], settings.bm25.candidate_depth)
        cold_latencies.append(cold_ms); warm_latencies.append(warm_ms)
        index_build_times.append(bm25.index_build_times_ms[str(query["product_id"])])
        labels = qrels_by_query.get(query["query_id"], {})
        metrics = retrieval_metrics_at_k([item.review_id for item in warm], labels, settings.bm25.default_top_k)
        per_query.append({"query_id": query["query_id"], "known_relevance_count": len(labels), "cold_latency_ms": cold_ms, "warm_latency_ms": warm_ms, **metrics})
        for item in cold:
            result_rows.append({"method": "bm25", "query_id": query["query_id"], "rank": item.rank, "score": item.score, "review_id": item.review_id, "product_id": item.product_id})
    paths.retrieval_results.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(result_rows, schema={"method": pl.String, "query_id": pl.String, "rank": pl.Int64, "score": pl.Float64, "review_id": pl.String, "product_id": pl.String}).write_parquet(paths.retrieval_results)
    evaluable = [row for row in per_query if row["known_relevance_count"] > 0]
    aggregate = {metric: (sum(row[metric] for row in evaluable if row[metric] is not None) / len(evaluable) if evaluable else None) for metric in ("recall", "precision", "mrr", "ndcg")}
    report = {"method": "bm25", "status": "available", "corpus_version": f"{settings.dataset.revision}:bm25-corpus-v1", "configuration": {"k1": settings.bm25.k1, "b": settings.bm25.b, "top_k": settings.bm25.default_top_k, "candidate_depth": settings.bm25.candidate_depth}, "index_strategy": "single persisted corpus Parquet; BM25 index is built and LRU-cached per requested product, with no millions of index files", "qrels_policy": "queries without known relevant documents are reported but excluded from aggregate ranking metrics", "label_limitations": "qrels are deterministic lexical seed candidates pending human review; metrics are a reproducibility/sanity check, not a claim of final retrieval quality", "query_count": len(queries), "evaluable_query_count": len(evaluable), "metrics_at_k": aggregate, "latency_ms": {"cold_p50": median(cold_latencies), "cold_p95": sorted(cold_latencies)[round((len(cold_latencies)-1)*0.95)], "warm_p50": median(warm_latencies), "warm_p95": sorted(warm_latencies)[round((len(warm_latencies)-1)*0.95)], "product_index_build_p50": median(index_build_times), "product_index_build_p95": sorted(index_build_times)[round((len(index_build_times)-1)*0.95)]}, "cache_statistics": bm25.cache_statistics(), "performance": {"runtime_seconds": perf_counter()-started, "peak_process_memory_bytes": peak_process_memory_bytes(), "python": sys.version, "platform": platform.platform()}, "ranked_results_path": str(paths.retrieval_results), "per_query": per_query}
    _write_json(paths.retrieval_evaluation_report, report)
    return report
