"""BGE-M3 dense embedding adapter and environment/resource helpers."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
import psutil

from .config import DenseSettings


class DenseEmbedder(Protocol):
    """Small seam that keeps model downloads out of unit tests."""

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def metadata(self) -> dict[str, object]: ...


def runtime_resources() -> dict[str, object]:
    """Return reproducible host and accelerator measurements when available."""
    result: dict[str, object] = {
        "platform": platform.platform(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            result["cuda_total_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    except ImportError:
        result["cuda_available"] = False
    return result


class BgeM3DenseEmbedder:
    """Official FlagEmbedding BGE-M3 dense-only adapter.

    The Phase 5 backend intentionally requests only ``dense_vecs``.  Sparse
    weights and ColBERT vectors are left for later, explicitly out-of-scope
    hybrid/reranker phases.
    """

    def __init__(self, settings: DenseSettings, cache_dir: Path):
        self.settings = settings
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(settings.device)
        if settings.use_fp16 and self.device == "cpu":
            raise ValueError("BGE-M3 fp16 is disabled on CPU; set use_fp16=false.")

        from FlagEmbedding import BGEM3FlagModel
        from huggingface_hub import snapshot_download

        started = perf_counter()
        snapshot = snapshot_download(
            repo_id=settings.model_id,
            revision=settings.model_revision or None,
            cache_dir=str(cache_dir),
        )
        self.snapshot_path = Path(snapshot)
        self.model = BGEM3FlagModel(
            str(self.snapshot_path),
            normalize_embeddings=settings.normalize_embeddings,
            use_fp16=settings.use_fp16,
            devices=self.device,
            batch_size=settings.batch_size,
            query_max_length=settings.max_length,
            passage_max_length=settings.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        self.load_seconds = perf_counter() - started

    @staticmethod
    def _resolve_device(requested: str) -> str:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("torch is required by FlagEmbedding") from exc
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("dense.device=cuda was requested but CUDA is unavailable")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("dense.device must be auto, cpu, or cuda")
        return requested

    def _normalise(self, vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("BGE-M3 returned an invalid dense embedding matrix")
        if self.settings.normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("BGE-M3 returned a zero-norm embedding")
            array = array / norms
        return np.ascontiguousarray(array, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        result = self.model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            max_length=self.settings.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._normalise(result["dense_vecs"])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # BGE-M3's official API exposes a retrieval-query method.  No LLM
        # rewriting or mutable query preprocessing is applied.
        result = self.model.encode_queries(
            list(texts),
            batch_size=self.settings.batch_size,
            max_length=self.settings.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._normalise(result["dense_vecs"])

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.settings.model_id,
            "model_revision": self.settings.model_revision,
            "adapter": "FlagEmbedding.BGEM3FlagModel",
            "flag_embedding_version": importlib.metadata.version("FlagEmbedding"),
            "device": self.device,
            "dtype": "float16" if self.settings.use_fp16 else "float32",
            "batch_size": self.settings.batch_size,
            "max_length": self.settings.max_length,
            "normalize_embeddings": self.settings.normalize_embeddings,
            "dense_only": True,
            "snapshot_path": str(self.snapshot_path),
            "model_load_seconds": self.load_seconds,
        }
