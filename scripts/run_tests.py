"""Run the Phase 1 test suite with the current Python interpreter."""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", *sys.argv[1:]]))
