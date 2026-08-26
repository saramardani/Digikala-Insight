"""Build full-population product statistics from canonical Parquet."""

from __future__ import annotations

import sys

from digikala_comparison.cli import build_statistics_main


if __name__ == "__main__":
    raise SystemExit(build_statistics_main(sys.argv[1:]))
