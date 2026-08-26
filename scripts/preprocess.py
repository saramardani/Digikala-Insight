"""Run the installed Phase 1 preprocessing command without shell aliases."""

from __future__ import annotations

import sys

from digikala_comparison.cli import preprocess_main


if __name__ == "__main__":
    raise SystemExit(preprocess_main(sys.argv[1:]))
