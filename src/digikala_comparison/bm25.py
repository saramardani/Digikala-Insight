"""Small deterministic BM25 implementation with product-scoped cached indexes."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
from pydantic import BaseModel

from .config import BM25Settings
from .retrieval_text import tokenize_persian_lexical


class RetrievedReview(BaseModel):
    review_id: str
    product_id: str
    score: float
    rank: int
    indexed_text_normalized: str
    review_text_raw: str | None
    title_raw: str | None
    advantages_items: list[str] | None
    disadvantages_items: list[str] | None
    is_buyer: bool | None
    recommendation_status: str | None
    review_rate: float | None
    likes: float | None
    dislikes: float | None


@dataclass
class _ProductBM25Index:
    records: list[dict[str, Any]]
    postings: dict[str, list[tuple[int, int]]]
    document_frequency: dict[str, int]
    average_document_length: float


class ProductScopedBM25:
    """One corpus Parquet plus a bounded LRU cache of per-product BM25 indexes."""

    def __init__(self, corpus_path: Path, settings: BM25Settings, cache_size: int = 32):
        self.corpus_path = corpus_path
        self.settings = settings
        self.cache_size = cache_size
        self._cache: OrderedDict[str, _ProductBM25Index] = OrderedDict()
        self.index_build_times_ms: dict[str, float] = {}

    def retrieve(self, product_id: str | int, query: str, top_k: int | None = None) -> list[RetrievedReview]:
        tokens = tokenize_persian_lexical(query)
        if not tokens:
            return []
        index = self._index_for_product(str(product_id))
        if index is None:
            return []
        scores: defaultdict[int, float] = defaultdict(float)
        document_count = len(index.records)
        for term in dict.fromkeys(tokens):
            postings = index.postings.get(term, [])
            if not postings:
                continue
            df = index.document_frequency[term]
            idf = log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            for document_index, term_frequency in postings:
                record = index.records[document_index]
                document_length = len(record["bm25_tokens"])
                denominator = term_frequency + self.settings.k1 * (
                    1 - self.settings.b
                    + self.settings.b * document_length / index.average_document_length
                )
                scores[document_index] += idf * (
                    term_frequency * (self.settings.k1 + 1) / denominator
                )
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], str(index.records[item[0]]["review_id"])),
        )[: top_k or self.settings.default_top_k]
        return [
            RetrievedReview(
                review_id=str(index.records[position]["review_id"]),
                product_id=str(index.records[position]["product_id"]),
                score=round(float(score), 6),
                rank=rank,
                indexed_text_normalized=index.records[position]["indexed_text_normalized"],
                review_text_raw=index.records[position]["review_text_raw"],
                title_raw=index.records[position]["title_raw"],
                advantages_items=index.records[position]["advantages_items"],
                disadvantages_items=index.records[position]["disadvantages_items"],
                is_buyer=index.records[position]["is_buyer_bool"],
                recommendation_status=index.records[position]["recommendation_status"],
                review_rate=index.records[position]["review_rate_numeric"],
                likes=index.records[position]["likes_numeric"],
                dislikes=index.records[position]["dislikes_numeric"],
            )
            for rank, (position, score) in enumerate(ranked, start=1)
        ]

    def _index_for_product(self, product_id: str) -> _ProductBM25Index | None:
        cached = self._cache.pop(product_id, None)
        if cached is not None:
            self._cache[product_id] = cached
            return cached
        started = perf_counter()
        records = (
            pl.scan_parquet(self.corpus_path)
            .filter(pl.col("product_id") == product_id)
            .collect()
            .to_dicts()
        )
        if not records:
            self.index_build_times_ms[product_id] = (perf_counter() - started) * 1000
            return None
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for position, record in enumerate(records):
            tokens = record["bm25_tokens"] or []
            record["bm25_tokens"] = tokens
            total_length += len(tokens)
            frequencies = Counter(tokens)
            for term, frequency in frequencies.items():
                postings[term].append((position, frequency))
                document_frequency[term] += 1
        index = _ProductBM25Index(
            records=records,
            postings=dict(postings),
            document_frequency=dict(document_frequency),
            average_document_length=max(total_length / len(records), 1.0),
        )
        self._cache[product_id] = index
        self.index_build_times_ms[product_id] = (perf_counter() - started) * 1000
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return index

    def cache_statistics(self) -> dict[str, int]:
        return {
            "cached_product_indexes": len(self._cache),
            "cached_documents": sum(len(index.records) for index in self._cache.values()),
            "cached_unique_terms": sum(len(index.postings) for index in self._cache.values()),
        }

    def timed_retrieve(self, product_id: str | int, query: str, top_k: int) -> tuple[list[RetrievedReview], float]:
        started = perf_counter()
        return self.retrieve(product_id, query, top_k), (perf_counter() - started) * 1000
