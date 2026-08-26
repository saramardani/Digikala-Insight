"""Run the installed Phase 1 quality-report command without shell aliases."""

from __future__ import annotations

import sys

from digikala_comparison.cli import quality_report_main


if __name__ == "__main__":
    raise SystemExit(quality_report_main(sys.argv[1:]))
