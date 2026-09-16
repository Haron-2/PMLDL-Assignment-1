"""Stage 4 - API integration tests against a REAL uvicorn server.

Launches `uvicorn main:app` in a subprocess, waits for /health, then checks:
    1. /health reports the model loaded
    2. valid red-wine request -> 200 + consistent response schema
    3. valid white-wine request -> 200
    4. missing field            -> 422
    5. invalid wine_type        -> 422
    6. free SO2 > total SO2     -> 422
    7. non-numeric value        -> 422

Usage (from anywhere; requires models/model.joblib to exist):
    .venv/Scripts/python.exe code/deployment/api/test_api.py
Env:
    API_PORT (default 8123)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parents[2]
PORT = int(os.getenv("API_PORT", "8123"))
BASE = f"http://127.0.0.1:{PORT}"

VALID_RED = {
    "fixed acidity": 7.4, "volatile acidity": 0.66, "citric acid": 0.0,
    "residual sugar": 1.8, "chlorides": 0.075, "free sulfur dioxide": 13.0,
    "total sulfur dioxide": 40.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red",
}
VALID_WHITE = {
    "fixed acidity": 7.0, "volatile acidity": 0.27, "citric acid": 0.36,
    "residual sugar": 20.7, "chlorides": 0.045, "free sulfur dioxide": 45.0,
    "total sulfur dioxide": 170.0, "density": 1.001, "pH": 3.0,
    "sulphates": 0.45, "alcohol": 8.8, "wine_type": "white",
}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    results.append((name, ok, extra))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))


def wait_for_health(proc: subprocess.Popen, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(0.5)
    return False


def main() -> int:
    if not (PROJECT_ROOT / "models" / "model.joblib").exists():
        print("models/model.joblib not found - run code/models/train.py first")
        return 1

    print(f"launching uvicorn on port {PORT} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
         "--port", str(PORT), "--log-level", "warning"],
        cwd=str(API_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_health(proc):
            print("FAIL: uvicorn did not become healthy in time")
            return 1

        print("\nAPI INTEGRATION TESTS")
        r = requests.get(f"{BASE}/health", timeout=5)
        body = r.json()
        check("GET /health -> 200, model_loaded=True", r.status_code == 200 and body["model_loaded"] is True,
              f"type={body.get('model_type')}")

        r = requests.post(f"{BASE}/predict", json=VALID_RED, timeout=10)
        ok = (r.status_code == 200 and r.json()["label"] in ("Good", "Poor")
              and r.json()["prediction"] in (0, 1)
              and 0.0 <= r.json()["probability_good"] <= 1.0
              and (r.json()["prediction"] == (1 if r.json()["label"] == "Good" else 0)))
        check("POST /predict valid red -> 200 + consistent schema", ok, str(r.json()) if r.status_code == 200 else r.text)

        r = requests.post(f"{BASE}/predict", json=VALID_WHITE, timeout=10)
        check("POST /predict valid white -> 200", r.status_code == 200, str(r.json()) if r.status_code == 200 else r.text)

        missing = dict(VALID_RED); missing.pop("alcohol")
        r = requests.post(f"{BASE}/predict", json=missing, timeout=10)
        check("missing 'alcohol' -> 422", r.status_code == 422)

        bad_type = dict(VALID_RED); bad_type["wine_type"] = "rose"
        r = requests.post(f"{BASE}/predict", json=bad_type, timeout=10)
        check("wine_type='rose' -> 422", r.status_code == 422)

        bad_so2 = dict(VALID_RED); bad_so2["free sulfur dioxide"] = 200.0; bad_so2["total sulfur dioxide"] = 100.0
        r = requests.post(f"{BASE}/predict", json=bad_so2, timeout=10)
        check("free SO2 > total SO2 -> 422", r.status_code == 422)

        bad_val = dict(VALID_RED); bad_val["alcohol"] = "abc"
        r = requests.post(f"{BASE}/predict", json=bad_val, timeout=10)
        check("alcohol='abc' -> 422", r.status_code == 422)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} API tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
