"""Explicit failures for data-contract violations."""


class DatasetPathError(FileNotFoundError):
    """Raised when a configured dataset file is absent."""


class RequiredColumnsError(ValueError):
    """Raised when an input CSV does not satisfy the documented schema."""


class DatasetDownloadError(RuntimeError):
    """Raised when a pinned dataset file cannot be downloaded or verified."""


class PinnedRevisionError(DatasetDownloadError):
    """Raised when Hugging Face does not resolve the configured revision."""
