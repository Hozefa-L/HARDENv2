"""Pytest entry point for the benchmark integrity checks.

Runs the same checks as `python scripts/verify_data.py` and fails the test
if any of them fail.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_benchmark_integrity():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_data.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"integrity checks failed:\n{result.stdout}\n{result.stderr}"
