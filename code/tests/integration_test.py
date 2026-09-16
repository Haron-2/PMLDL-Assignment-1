"""Stage 8 - end-to-end integration test of the whole MLOps pipeline.

Runs every stage back-to-back exactly as documented and asserts exit codes
plus key outputs:

    [1/6] raw data verification   (services/airflow/scripts/verify_raw.py)
    [2/6] preprocessing           (code/datasets/preprocess.py)   + 13 checks
    [3/6] training + MLflow       (code/models/train.py)           + metrics sanity
    [4/6] independent evaluation  (code/models/evaluate.py)        + figure
    [5/6] API integration tests   (code/deployment/api/test_api.py) + 7 tests
    [6/6] deployment configs      (docker compose config for both stacks)

Requires: models/ + data/ regenerable from data/raw (all steps re-run here),
Docker for step 6 (skipped gracefully with a WARN if Docker is unavailable).

Usage (from anywhere):
    .venv/Scripts/python.exe code/tests/integration_test.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = sys.executable

STEPS = [
    ("[1/6] raw data verification",
     [VENV_PY, str(PROJECT_ROOT / "services" / "airflow" / "scripts" / "verify_raw.py")],
     ["RAW DATA VERIFICATION: PASSED"]),
    ("[2/6] preprocessing",
     [VENV_PY, str(PROJECT_ROOT / "code" / "datasets" / "preprocess.py")],
     ["all 13 checks passed", "STAGE 2 PREPROCESSING: DONE"]),
    ("[3/6] training + MLflow tracking",
     [VENV_PY, str(PROJECT_ROOT / "code" / "models" / "train.py")],
     ["STAGE 3 TRAINING: DONE"]),
    ("[4/6] independent evaluation",
     [VENV_PY, str(PROJECT_ROOT / "code" / "models" / "evaluate.py")],
     ["PASS: saved artifact reproduces the logged metrics exactly",
      "STAGE 3 EVALUATION: DONE"]),
    ("[5/6] API integration tests (real uvicorn)",
     [VENV_PY, str(PROJECT_ROOT / "code" / "deployment" / "api" / "test_api.py")],
     ["7/7 API tests passed"]),
    ("[6/6] deployment configs (docker compose config)",
     ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), "config", "--quiet"],
     []),
]

results: list[tuple[str, bool, str]] = []


def run_step(name: str, cmd: list[str], expect_markers: list[str]) -> None:
    print(f"\n--- {name} ---")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(PROJECT_ROOT), timeout=900,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    missing = [m for m in expect_markers if m not in out]
    if missing:
        ok = False
    detail = f"exit={proc.returncode}"
    if missing:
        detail += f", missing markers: {missing}"
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    if not ok:
        tail = "\n".join(out.strip().splitlines()[-15:])
        print(f"  --- last output ---\n{tail}")


def check_artifacts() -> None:
    print("\n--- artifact sanity ---")
    import json
    import pandas as pd

    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    ok = len(train) == 4256 and len(test) == 1064 and "target" in train.columns \
        and "quality" not in train.columns and not train.isna().any().any()
    results.append(("artifact sanity: 4256/1064 rows, no NaN, target only", ok, ""))
    print(f"  [{'PASS' if ok else 'FAIL'}] train/test shapes and schema")

    metrics = json.loads((PROJECT_ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    acc, f1 = metrics["metrics"]["accuracy"], metrics["metrics"]["f1"]
    ok = 0.70 <= acc <= 0.90 and 0.75 <= f1 <= 0.90
    results.append((f"metric sanity: accuracy={acc:.4f} f1={f1:.4f} within expected bands", ok, ""))
    print(f"  [{'PASS' if ok else 'FAIL'}] accuracy={acc:.4f}, f1={f1:.4f}")


def main() -> int:
    print("=" * 78)
    print("STAGE 8 | END-TO-END INTEGRATION TEST")
    print("=" * 78)
    for name, cmd, markers in STEPS:
        run_step(name, cmd, markers)
    check_artifacts()

    print("\n" + "=" * 78)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 78)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} integration checks passed")
    print("INTEGRATION TEST:", "PASSED" if failed == 0 else "FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
