"""Stage 7 - verify raw data availability before the pipeline runs.

Exits non-zero (and Airflow marks the task failed) if either raw dataset is
missing, truncated or corrupted; otherwise prints row counts and exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts -> airflow -> services -> project
RAW = PROJECT_ROOT / "data" / "raw"

EXPECTED = {
    "winequality-red.csv": 1599,
    "winequality-white.csv": 4898,
}


def main() -> int:
    ok = True
    for filename, expected_rows in EXPECTED.items():
        path = RAW / filename
        if not path.exists():
            print(f"[FAIL] missing raw file: {path}")
            ok = False
            continue
        with open(path, "rb") as f:
            lines = sum(1 for _ in f)
        rows = lines - 1  # header
        if rows != expected_rows:
            print(f"[FAIL] {filename}: expected {expected_rows} rows, found {rows}")
            ok = False
        else:
            print(f"[PASS] {filename}: {rows} rows (expected {expected_rows})")
    print("RAW DATA VERIFICATION:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
