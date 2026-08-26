"""Resumable BGE-M3 vector storage and product-scoped FAISS retrieval."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .bm25 import RetrievedReview
from .config import DenseSettings, Settings
from .dense_embedding import BgeM3DenseEmbedder, DenseEmbedder, runtime_resources
from .product_identity import normalize_product_text
from .runtime import peak_process_memory_bytes


_SOURCE_COLUMNS = [
    "review_id",
    "product_id",
    "indexed_text_normalized",
    "review_text_raw",
    "title_raw",
    "advantages_items",
    "disadvantages_items",
    "is_buyer_bool",
    "recommendation_status",
    "review_rate_numeric",
    "likes_numeric",
    "dislikes_numeric",
]


@dataclass(frozen=True)
class DenseIndexPaths:
    root: Path
    sorted_corpus: Path
    manifest: Path
    product_ranges: Path

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @classmethod
    def from_settings(cls, settings: Settings, root: Path | None = None) -> "DenseIndexPaths":
        if root is None:
            root = _required_path(settings.paths.dense_index_root, "dense_index_root")
            sorted_corpus = _required_path(settings.paths.dense_sorted_corpus, "dense_sorted_corpus")
            manifest = _required_path(settings.paths.dense_manifest, "dense_manifest")
            product_ranges = _required_path(settings.paths.dense_product_ranges, "dense_product_ranges")
        else:
            sorted_corpus = root / "dense_retrieval_reviews_sorted.parquet"
            manifest = root / "manifest.json"
            product_ranges = root / "product_ranges.parquet"
        return cls(root=root, sorted_corpus=sorted_corpus, manifest=manifest, product_ranges=product_ranges)


def _required_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} must be configured")
    return path


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus_version(settings: Settings) -> str:
    return f"{settings.dataset.revision}:bm25-corpus-v1"


def _count_rows(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect()[0, 0])


def prepare_dense_source(
    settings: Settings,
    paths: DenseIndexPaths,
    max_documents: int | None = None,
) -> Path:
    """Create an atomically-published product/review-ID ordered corpus view.

    It contains exactly the Phase 4 eligible documents and its existing text
    composition.  Ordering lets every product occupy one contiguous vector-ID
    range without creating a file per product.
    """
    source = _required_path(settings.paths.retrieval_corpus, "retrieval_corpus")
    if not source.is_file():
        raise FileNotFoundError("retrieval_reviews.parquet is required. Run Phase 4 first.")
    if paths.sorted_corpus.is_file():
        return paths.sorted_corpus
    paths.sorted_corpus.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.sorted_corpus.with_suffix(".partial.parquet")
    frame = pl.scan_parquet(source).select(_SOURCE_COLUMNS)
    if max_documents is not None:
        if max_documents <= 0:
            raise ValueError("max_documents must be positive")
        frame = frame.head(max_documents)
    provenance = frame.select([pl.len().alias("rows"), pl.col("review_id").n_unique().alias("unique_ids")]).collect()
    if int(provenance[0, "rows"]) != int(provenance[0, "unique_ids"]):
        raise ValueError("dense indexing refuses duplicate review_id values; rebuild the Phase 4 corpus")
    frame.sort(["product_id", "review_id"]).sink_parquet(temporary)
    os.replace(temporary, paths.sorted_corpus)
    return paths.sorted_corpus


def _iter_records(
    source: Path, *, batch_size: int, skip_rows: int, max_documents: int | None
) -> Iterator[list[dict[str, Any]]]:
    """Yield bounded batches from Parquet without accumulating all reviews."""
    skipped = 0
    yielded = 0
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=_SOURCE_COLUMNS):
        if skipped < skip_rows:
            amount = min(batch.num_rows, skip_rows - skipped)
            batch = batch.slice(amount)
            skipped += amount
        if batch.num_rows == 0:
            continue
        if max_documents is not None:
            remaining = max_documents - yielded
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        records = batch.to_pylist()
        yielded += len(records)
        if records:
            yield records
        if max_documents is not None and yielded >= max_documents:
            break


def _save_chunk(
    paths: DenseIndexPaths,
    chunk_id: int,
    vector_start: int,
    vectors: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically persist matching vector and metadata chunks before manifesting."""
    if len(records) != vectors.shape[0]:
        raise ValueError("vector/metadata chunk row counts differ")
    if len({str(record["review_id"]) for record in records}) != len(records):
        raise ValueError("duplicate review_id inside a dense embedding checkpoint")
    paths.vectors_dir.mkdir(parents=True, exist_ok=True)
    paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    name = f"chunk-{chunk_id:06d}"
    vector_path = paths.vectors_dir / f"{name}.npy"
    metadata_path = paths.metadata_dir / f"{name}.parquet"
    vector_temp = paths.vectors_dir / f"{name}.partial.npy"
    metadata_temp = paths.metadata_dir / f"{name}.partial.parquet"
    np.save(vector_temp, np.ascontiguousarray(vectors, dtype=np.float32), allow_pickle=False)
    metadata = pl.DataFrame(records).with_columns(
        pl.Series("vector_id", np.arange(vector_start, vector_start + len(records), dtype=np.uint64))
    )
    metadata.write_parquet(metadata_temp)
    os.replace(vector_temp, vector_path)
    os.replace(metadata_temp, metadata_path)
    return {
        "chunk_id": chunk_id,
        "vector_start": vector_start,
        "count": len(records),
        "vector_file": str(vector_path.relative_to(paths.root)),
        "metadata_file": str(metadata_path.relative_to(paths.root)),
    }


def _build_product_ranges(paths: DenseIndexPaths) -> int:
    """Persist the deterministic product -> contiguous vector-ID mapping."""
    metadata_glob = str(paths.metadata_dir / "chunk-*.parquet")
    ranges = (
        pl.scan_parquet(metadata_glob)
        .group_by("product_id")
        .agg(
            [
                pl.col("vector_id").min().alias("vector_start"),
                pl.col("vector_id").max().alias("vector_end"),
                pl.len().alias("vector_count"),
            ]
        )
        .with_columns((pl.col("vector_end") - pl.col("vector_start") + 1).alias("span"))
        .collect()
    )
    non_contiguous = ranges.filter(pl.col("span") != pl.col("vector_count"))
    if non_contiguous.height:
        raise RuntimeError("dense source ordering did not produce contiguous product vector ranges")
    paths.product_ranges.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.product_ranges.with_suffix(".partial.parquet")
    ranges.drop("span").sort("product_id").write_parquet(temporary)
    os.replace(temporary, paths.product_ranges)
    return ranges.height


def build_dense_embeddings(
    settings: Settings,
    *,
    output_root: Path | None = None,
    max_documents: int | None = None,
    embedder: DenseEmbedder | None = None,
) -> dict[str, Any]:
    """Create or resume a durable BGE-M3 vector corpus in bounded checkpoints."""
    paths = DenseIndexPaths.from_settings(settings, output_root)
    source = prepare_dense_source(settings, paths, max_documents=max_documents)
    expected_documents = _count_rows(source)
    full_documents = _count_rows(_required_path(settings.paths.retrieval_corpus, "retrieval_corpus"))
    is_full_corpus = max_documents is None and expected_documents == full_documents
    if paths.manifest.is_file():
        manifest = _read_json(paths.manifest)
        if manifest["expected_documents"] != expected_documents:
            raise ValueError("existing dense manifest has a different source/document count")
        if manifest["model"]["model_id"] != settings.dense.model_id or manifest["model"]["model_revision"] != settings.dense.model_revision:
            raise ValueError("existing dense manifest was created with a different BGE-M3 pin")
    else:
        manifest = {
            "schema_version": "bge-m3-dense-v1",
            "status": "embedding",
            "corpus_version": _corpus_version(settings),
            "source_corpus": str(source),
            "full_corpus_document_count": full_documents,
            "expected_documents": expected_documents,
            "is_full_corpus": is_full_corpus,
            "model": {
                "model_id": settings.dense.model_id,
                "model_revision": settings.dense.model_revision,
                "backend": settings.dense.backend,
                "max_length": settings.dense.max_length,
                "embedding_dimension": settings.dense.embedding_dimension,
                "normalize_embeddings": settings.dense.normalize_embeddings,
            },
            "checkpoint_documents": settings.dense.checkpoint_documents,
            "chunks": [],
            "completed_documents": 0,
        }
        _write_json_atomic(paths.manifest, manifest)
    if manifest["status"] == "complete":
        return {"paths": paths, "manifest": manifest, "resumed": True}

    completed = int(manifest["completed_documents"])
    if completed > expected_documents:
        raise RuntimeError("dense manifest completed count exceeds source rows")
    active_embedder = embedder or BgeM3DenseEmbedder(settings.dense, paths.root / "model_cache")
    manifest["model"].update(active_embedder.metadata())
    started = perf_counter()
    pending_records: list[dict[str, Any]] = []
    pending_vectors: list[np.ndarray] = []
    encoded_documents = 0
    dimension = manifest.get("dimension")
    for records in _iter_records(
        source,
        batch_size=settings.dense.batch_size,
        skip_rows=completed,
        max_documents=None,
    ):
        texts = [str(record["indexed_text_normalized"]) for record in records]
        vectors = np.asarray(active_embedder.encode_passages(texts), dtype=np.float32)
        if vectors.shape[0] != len(records):
            raise ValueError("embedder returned a different number of passage vectors")
        if dimension is None:
            dimension = int(vectors.shape[1])
            manifest["dimension"] = dimension
            if dimension != settings.dense.embedding_dimension:
                raise ValueError(
                    f"BGE-M3 returned dimension {dimension}, expected {settings.dense.embedding_dimension}"
                )
        if vectors.shape[1] != dimension:
            raise ValueError("embedder changed vector dimension within one corpus")
        pending_records.extend(records)
        pending_vectors.append(vectors)
        encoded_documents += len(records)
        if len(pending_records) >= settings.dense.checkpoint_documents:
            merged = np.concatenate(pending_vectors, axis=0)
            chunk = _save_chunk(
                paths,
                len(manifest["chunks"]),
                completed,
                merged,
                pending_records,
            )
            manifest["chunks"].append(chunk)
            completed += len(pending_records)
            manifest["completed_documents"] = completed
            manifest["status"] = "embedding"
            _write_json_atomic(paths.manifest, manifest)
            print(f"dense checkpoint {len(manifest['chunks'])}: {completed}/{expected_documents}")
            pending_records = []
            pending_vectors = []
    if pending_records:
        merged = np.concatenate(pending_vectors, axis=0)
        chunk = _save_chunk(paths, len(manifest["chunks"]), completed, merged, pending_records)
        manifest["chunks"].append(chunk)
        completed += len(pending_records)
        manifest["completed_documents"] = completed
        _write_json_atomic(paths.manifest, manifest)
    if completed != expected_documents:
        raise RuntimeError("embedding finished without covering every expected document")
    product_count = _build_product_ranges(paths)
    elapsed = perf_counter() - started
    vector_bytes = sum((paths.root / chunk["vector_file"]).stat().st_size for chunk in manifest["chunks"])
    metadata_bytes = sum((paths.root / chunk["metadata_file"]).stat().st_size for chunk in manifest["chunks"])
    manifest.update(
        {
            "status": "complete",
            "product_count": product_count,
            "embedding_runtime_seconds": elapsed,
            "embedding_documents_per_second": encoded_documents / elapsed if elapsed else None,
            "vector_storage_bytes": vector_bytes,
            "metadata_storage_bytes": metadata_bytes,
            "index_storage_bytes": vector_bytes + metadata_bytes + paths.product_ranges.stat().st_size,
            "peak_process_memory_bytes": peak_process_memory_bytes(),
            "resources": runtime_resources(),
        }
    )
    _write_json_atomic(paths.manifest, manifest)
    return {"paths": paths, "manifest": manifest, "resumed": False}


@dataclass
class _DenseProductIndex:
    index: Any
    records: list[dict[str, Any]]


class DenseFaissRetriever:
    """Exact cosine/IP retrieval over one product's persisted BGE-M3 vectors."""

    def __init__(self, paths: DenseIndexPaths, settings: DenseSettings, embedder: DenseEmbedder):
        if not paths.manifest.is_file() or not paths.product_ranges.is_file():
            raise FileNotFoundError("a completed dense index and product_ranges.parquet are required")
        self.paths = paths
        self.settings = settings
        self.embedder = embedder
        self.manifest = _read_json(paths.manifest)
        if self.manifest.get("status") != "complete":
            raise RuntimeError("dense index is incomplete; resume embedding before retrieval")
        self._ranges = {
            str(row["product_id"]): row
            for row in pl.read_parquet(paths.product_ranges).to_dicts()
        }
        self._cache: OrderedDict[str, _DenseProductIndex] = OrderedDict()

    @classmethod
    def from_settings(cls, settings: Settings) -> "DenseFaissRetriever":
        paths = DenseIndexPaths.from_settings(settings)
        if not paths.manifest.is_file() or not paths.product_ranges.is_file():
            raise FileNotFoundError("a completed dense index and product_ranges.parquet are required")
        return cls(paths, settings.dense, BgeM3DenseEmbedder(settings.dense, paths.root / "model_cache"))

    def _load_product_index(self, product_id: str) -> _DenseProductIndex | None:
        cached = self._cache.pop(product_id, None)
        if cached is not None:
            self._cache[product_id] = cached
            return cached
        row = self._ranges.get(product_id)
        if row is None:
            return None
        vector_start, vector_end = int(row["vector_start"]), int(row["vector_end"])
        vectors: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for chunk in self.manifest["chunks"]:
            chunk_start = int(chunk["vector_start"])
            chunk_end = chunk_start + int(chunk["count"]) - 1
            overlap_start, overlap_end = max(vector_start, chunk_start), min(vector_end, chunk_end)
            if overlap_start > overlap_end:
                continue
            local_start, count = overlap_start - chunk_start, overlap_end - overlap_start + 1
            array = np.load(self.paths.root / chunk["vector_file"], mmap_mode="r", allow_pickle=False)
            vectors.append(np.ascontiguousarray(array[local_start : local_start + count], dtype=np.float32))
            metadata = pl.read_parquet(self.paths.root / chunk["metadata_file"]).slice(local_start, count)
            records.extend(metadata.to_dicts())
        if not records or len(records) != int(row["vector_count"]):
            raise RuntimeError("dense product range does not map to complete metadata")
        if any(str(record["product_id"]) != product_id for record in records):
            raise RuntimeError("product filtering invariant failed for dense vectors")
        matrix = np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)
        import faiss

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        product_index = _DenseProductIndex(index=index, records=records)
        self._cache[product_id] = product_index
        if len(self._cache) > self.settings.product_index_cache_size:
            self._cache.popitem(last=False)
        return product_index

    def retrieve(
        self, product_id: str | int, query: str, top_k: int | None = None
    ) -> list[RetrievedReview]:
        normalized_query = normalize_product_text(query)
        if not normalized_query:
            return []
        index = self._load_product_index(str(product_id))
        if index is None:
            return []
        query_vector = np.asarray(self.embedder.encode_queries([normalized_query]), dtype=np.float32)
        expected_dimension = int(self.manifest["dimension"])
        if query_vector.shape != (1, expected_dimension):
            raise ValueError("query embedding dimension does not match the persisted dense index")
        count = min(top_k or 10, len(index.records))
        scores, positions = index.index.search(np.ascontiguousarray(query_vector), count)
        results: list[RetrievedReview] = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0], strict=True), start=1):
            record = index.records[int(position)]
            results.append(
                RetrievedReview(
                    review_id=str(record["review_id"]),
                    product_id=str(record["product_id"]),
                    score=float(score),
                    rank=rank,
                    indexed_text_normalized=str(record["indexed_text_normalized"]),
                    review_text_raw=record.get("review_text_raw"),
                    title_raw=record.get("title_raw"),
                    advantages_items=record.get("advantages_items"),
                    disadvantages_items=record.get("disadvantages_items"),
                    is_buyer=record.get("is_buyer_bool"),
                    recommendation_status=record.get("recommendation_status"),
                    review_rate=record.get("review_rate_numeric"),
                    likes=record.get("likes_numeric"),
                    dislikes=record.get("dislikes_numeric"),
                )
            )
        return results

    def timed_retrieve(self, product_id: str | int, query: str, top_k: int) -> tuple[list[RetrievedReview], float]:
        started = perf_counter()
        return self.retrieve(product_id, query, top_k), (perf_counter() - started) * 1000

    def cache_statistics(self) -> dict[str, int]:
        return {
            "cached_product_indexes": len(self._cache),
            "cached_documents": sum(len(item.records) for item in self._cache.values()),
        }
