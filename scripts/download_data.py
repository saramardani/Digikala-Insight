"""Download the full pinned Digikala CSV dataset."""

from __future__ import annotations

import sys

from digikala_comparison.cli import download_data_main


if __name__ == "__main__":
    raise SystemExit(download_data_main(sys.argv[1:]))
