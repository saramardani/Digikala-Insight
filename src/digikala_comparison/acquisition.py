"""Streaming acquisition of the exact Hugging Face dataset revision."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from .config import Settings
from .errors import DatasetDownloadError, PinnedRevisionError

REQUIRED_FILES = ("digikala-products.csv", "digikala-comments.csv")
MANIFEST_NAME = ".digikala_dataset_manifest.json"
CURL_RANGE_BYTES = 8 * 1024 * 1024
CURL_PARALLEL_REQUESTS = 4


@dataclass(frozen=True)
class RemoteFile:
    filename: str
    download_url: str
    revision: str
    expected_size: int | None


@dataclass(frozen=True)
class DownloadResult:
    filename: str
    path: str
    size_bytes: int
    revision: str
    status: str


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _header_int(headers: Any, name: str) -> int | None:
    value = headers.get(name)
    return int(value) if value is not None else None


def resolve_pinned_file(settings: Settings, filename: str) -> RemoteFile:
    if not settings.dataset.repository:
        raise DatasetDownloadError("dataset.repository must be configured for downloads")
    endpoint = (
        f"https://huggingface.co/datasets/{settings.dataset.repository}/resolve/"
        f"{settings.dataset.revision}/{filename}?download=true"
    )
    request = Request(endpoint, method="GET")
    opener = build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=60)
    except HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise DatasetDownloadError(
                f"Unable to resolve {filename} at pinned revision "
                f"{settings.dataset.revision}: HTTP {error.code}"
            ) from error
        headers = error.headers
        location = headers.get("Location")
        resolved_revision = headers.get("X-Repo-Commit")
    except URLError as error:
        raise DatasetDownloadError(f"Network error while resolving {filename}: {error}") from error
    else:
        headers = response.headers
        location = response.geturl()
        resolved_revision = headers.get("X-Repo-Commit")
        response.close()

    if resolved_revision != settings.dataset.revision:
        raise PinnedRevisionError(
            f"{filename} resolved to {resolved_revision!r}, not configured revision "
            f"{settings.dataset.revision!r}."
        )
    if not location:
        raise DatasetDownloadError(f"No download location returned for {filename}")
    return RemoteFile(
        filename=filename,
        download_url=location,
        revision=resolved_revision,
        expected_size=_header_int(headers, "X-Linked-Size")
        or _header_int(headers, "Content-Length"),
    )


def _manifest_path(raw_dir: Path) -> Path:
    return raw_dir / MANIFEST_NAME


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetDownloadError(
            f"Existing manifest is invalid: {path}. Use --force after inspecting it."
        ) from error


def _existing_is_verified(path: Path, remote: RemoteFile, manifest: dict[str, Any]) -> bool:
    entry = manifest.get("files", {}).get(remote.filename, {})
    return (
        path.is_file()
        and path.stat().st_size == remote.expected_size
        and entry.get("revision") == remote.revision
        and entry.get("size_bytes") == remote.expected_size
    )


def download_remote_file(
    remote: RemoteFile, destination: Path, chunk_bytes: int, force: bool = False
) -> DownloadResult:
    """Stream one remote file to a partial path and atomically publish it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    if partial.exists() and not force:
        raise DatasetDownloadError(
            f"Incomplete partial file exists: {partial}. Re-run with --force to restart."
        )
    if force and partial.exists():
        partial.unlink()

    # curl handles the Hugging Face Xet CDN's long-lived responses reliably on
    # Windows and Unix-like systems.  It writes directly to the partial file,
    # so an interrupted transfer can never be mistaken for a published CSV.
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl is not None:
        received = _download_with_curl(curl, remote, partial)
    else:
        received = _download_with_urlopen(remote, partial, chunk_bytes)

    expected_size = remote.expected_size
    if expected_size is not None and received != expected_size:
        raise DatasetDownloadError(
            f"Incomplete download for {remote.filename}: received {received} bytes, "
            f"expected {expected_size}. Partial data remains at {partial}."
        )
    os.replace(partial, destination)
    return DownloadResult(
        filename=remote.filename,
        path=str(destination),
        size_bytes=received,
        revision=remote.revision,
        status="downloaded",
    )


def _download_with_urlopen(remote: RemoteFile, partial: Path, chunk_bytes: int) -> int:
    """Portable fallback when curl is unavailable."""
    received = 0
    try:
        with urlopen(remote.download_url, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(chunk_bytes):
                handle.write(chunk)
                received += len(chunk)
    except (OSError, URLError) as error:
        raise DatasetDownloadError(
            f"Download interrupted for {remote.filename}; partial data remains at {partial}. "
            "Re-run with --force to restart."
        ) from error
    return received


def _download_with_curl(curl: str, remote: RemoteFile, partial: Path) -> int:
    """Request bounded byte ranges to avoid stalled full Xet CDN responses."""
    if remote.expected_size is None:
        raise DatasetDownloadError(
            f"The pinned source did not report a size for {remote.filename}; "
            "cannot safely perform ranged download."
        )
    received = 0
    while received < remote.expected_size:
        ranges: list[tuple[int, int]] = []
        next_start = received
        for _ in range(CURL_PARALLEL_REQUESTS):
            if next_start >= remote.expected_size:
                break
            size = min(CURL_RANGE_BYTES, remote.expected_size - next_start)
            ranges.append((next_start, size))
            next_start += size
        try:
            with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
                chunks = list(
                    executor.map(
                        lambda item: _fetch_curl_range(curl, remote, *item), ranges
                    )
                )
        except DatasetDownloadError as error:
            raise DatasetDownloadError(
                f"{error} Partial data remains at {partial}; re-run with --force to restart."
            ) from error
        # Bounded responses are held only for this batch (at most 32 MiB), then
        # appended in deterministic byte order to the partial file.
        with partial.open("ab") as handle:
            for start, chunk in sorted(chunks):
                if start != received:
                    raise DatasetDownloadError(
                        f"Out-of-order range while downloading {remote.filename}."
                    )
                handle.write(chunk)
                received += len(chunk)
    return received


def _fetch_curl_range(
    curl: str, remote: RemoteFile, start: int, expected_chunk: int
) -> tuple[int, bytes]:
    byte_range = f"{start}-{start + expected_chunk - 1}"
    command = [
        curl,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "120",
        "--range",
        byte_range,
        remote.download_url,
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except OSError as error:
        raise DatasetDownloadError(
            f"Unable to start curl for {remote.filename}: {error}"
        ) from error
    if completed.returncode != 0 or len(completed.stdout) != expected_chunk:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        detail = detail or "no diagnostic output"
        raise DatasetDownloadError(
            f"Download interrupted for {remote.filename} range {byte_range}; "
            f"curl exit code {completed.returncode}: {detail}."
        )
    return start, completed.stdout


def download_pinned_dataset(settings: Settings, force: bool = False) -> list[DownloadResult]:
    raw_dir = settings.paths.raw_products.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(raw_dir)
    manifest = {} if force else _read_manifest(manifest_path)
    results: list[DownloadResult] = []

    filenames_to_paths = {
        "digikala-products.csv": settings.paths.raw_products,
        "digikala-comments.csv": settings.paths.raw_comments,
    }
    for filename in REQUIRED_FILES:
        remote = resolve_pinned_file(settings, filename)
        destination = filenames_to_paths[filename]
        if not force and _existing_is_verified(destination, remote, manifest):
            results.append(
                DownloadResult(
                    filename=filename,
                    path=str(destination),
                    size_bytes=destination.stat().st_size,
                    revision=remote.revision,
                    status="already_verified",
                )
            )
        else:
            if destination.exists() and not force:
                raise DatasetDownloadError(
                    f"Existing file cannot be verified for the pinned revision: {destination}. "
                    "Use --force to replace it."
                )
            results.append(
                download_remote_file(
                    remote, destination, settings.download_chunk_bytes, force=force
                )
            )

    manifest_payload = {
        "repository": settings.dataset.repository,
        "revision": settings.dataset.revision,
        "files": {
            result.filename: {
                "revision": result.revision,
                "size_bytes": result.size_bytes,
                "path": result.path,
            }
            for result in results
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return results
