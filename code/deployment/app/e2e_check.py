"""Stage 6/8 - container-to-container end-to-end check.

Runs INSIDE the app container and calls the API container over the Compose
network exactly the way the UI does. Used to verify Docker networking.

    python /tmp/e2e_check.py   (or from repo: code/deployment/app/e2e_check.py)
Env:
    API_URL (default http://api:8000)
"""

from __future__ import annotations

import os
import sys

import requests

API_URL = os.getenv("API_URL", "http://api:8000")
VALID_WHITE = {
    "fixed acidity": 7.0, "volatile acidity": 0.27, "citric acid": 0.36,
    "residual sugar": 20.7, "chlorides": 0.045, "free sulfur dioxide": 45.0,
    "total sulfur dioxide": 170.0, "density": 1.001, "pH": 3.0,
    "sulphates": 0.45, "alcohol": 8.8, "wine_type": "white",
}


def main() -> int:
    ok = True

    try:
        h = requests.get(f"{API_URL}/health", timeout=10)
        ok &= h.status_code == 200 and h.json().get("model_loaded") is True
        print(f"[{'PASS' if ok else 'FAIL'}] health: {h.status_code} {h.json()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] health: {exc}")
        return 1

    try:
        r = requests.post(f"{API_URL}/predict", json=VALID_WHITE, timeout=15)
        body = r.json()
        ok &= r.status_code == 200 and body["prediction"] in (0, 1) \
            and body["label"] in ("Good", "Poor") and 0.0 <= body["probability_good"] <= 1.0
        print(f"[{'PASS' if ok else 'FAIL'}] predict: {r.status_code} {body}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] predict: {exc}")
        return 1

    print("E2E CHECK:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
