from __future__ import annotations

from io import BytesIO

import pytest

from digikala_comparison import acquisition
from digikala_comparison.acquisition import RemoteFile, download_pinned_dataset, download_remote_file
from digikala_comparison.config import (
    DatasetSettings,
    NormalizationSettings,
    PathSettings,
    ReviewEligibilitySettings,
    Settings,
)
from digikala_comparison.errors import DatasetDownloadError


def _settings(tmp_path) -> Settings:
    return Settings(
        dataset=DatasetSettings("revision", "https://example.invalid", "owner/repository"),
        paths=PathSettings(
            raw_products=tmp_path / "raw" / "digikala-products.csv",
            raw_comments=tmp_path / "raw" / "digikala-comments.csv",
            processed_products=tmp_path / "processed" / "products.parquet",
            processed_comments=tmp_path / "processed" / "comments.parquet",
            quality_report=tmp_path / "reports" / "quality.json",
        ),
        random_seed=42,
        normalization=NormalizationSettings("NFC", True, True, True, True),
        review_eligibility=ReviewEligibilitySettings(
            True, 1, False, ("recommended", "not_recommended", "no_idea")
        ),
    )


class _Response(BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_incomplete_download_leaves_partial_file(monkeypatch, tmp_path) -> None:
    remote = RemoteFile("products.csv", "https://example.invalid/file", "revision", 3)
    monkeypatch.setattr(acquisition.shutil, "which", lambda _name: None)
    monkeypatch.setattr(acquisition, "urlopen", lambda *args, **kwargs: _Response(b"ab"))

    with pytest.raises(DatasetDownloadError, match="Incomplete download"):
        download_remote_file(remote, tmp_path / "products.csv", chunk_bytes=1)

    assert (tmp_path / "products.csv.part").read_bytes() == b"ab"


def test_existing_verified_files_are_not_redownloaded(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.paths.raw_products.parent.mkdir(parents=True)
    settings.paths.raw_products.write_bytes(b"product")
    settings.paths.raw_comments.write_bytes(b"comment")
    remotes = {
        "digikala-products.csv": RemoteFile(
            "digikala-products.csv", "https://example.invalid/products", "revision", 7
        ),
        "digikala-comments.csv": RemoteFile(
            "digikala-comments.csv", "https://example.invalid/comments", "revision", 7
        ),
    }
    monkeypatch.setattr(acquisition, "resolve_pinned_file", lambda _, name: remotes[name])
    (settings.paths.raw_products.parent / acquisition.MANIFEST_NAME).write_text(
        '{"files":{"digikala-products.csv":{"revision":"revision","size_bytes":7},'
        '"digikala-comments.csv":{"revision":"revision","size_bytes":7}}}',
        encoding="utf-8",
    )

    results = download_pinned_dataset(settings)

    assert [result.status for result in results] == ["already_verified", "already_verified"]


def test_force_requests_download_even_when_file_exists(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)
    remotes = {
        "digikala-products.csv": RemoteFile(
            "digikala-products.csv", "https://example.invalid/products", "revision", 1
        ),
        "digikala-comments.csv": RemoteFile(
            "digikala-comments.csv", "https://example.invalid/comments", "revision", 1
        ),
    }
    monkeypatch.setattr(acquisition, "resolve_pinned_file", lambda _, name: remotes[name])
    calls: list[str] = []

    def fake_download(remote, destination, chunk_bytes, force):
        calls.append(remote.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x")
        return acquisition.DownloadResult(
            remote.filename, str(destination), 1, remote.revision, "downloaded"
        )

    monkeypatch.setattr(acquisition, "download_remote_file", fake_download)
    download_pinned_dataset(settings, force=True)

    assert calls == ["digikala-products.csv", "digikala-comments.csv"]
