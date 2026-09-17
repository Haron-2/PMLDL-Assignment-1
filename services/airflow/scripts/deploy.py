"""Stage 7 - Airflow deploy task: rebuild + restart the Docker Compose deployment.

Runs INSIDE the airflow container (docker CLI + compose plugin + the host
Docker daemon socket mounted at /var/run/docker.sock; on Docker Desktop the
container therefore runs as root to access the socket). Executed as the final
DAG task after evaluate, so every 5-minute pipeline run serves the FRESH
models/model.joblib: the artifact is baked into the API image at build time,
thus `docker compose build` MUST rerun after each retrain before `up -d`
recreates the containers.

This script does not just log commands - it validates the deployment:

    1. sha256(models/model.joblib) on the project mount      (before build)
    2. docker compose build  (fresh artifact -> API image)
    3. docker compose up -d  (recreate changed containers)
    4. wine-api /health polled via `docker exec` (model_loaded == true)
    5. POST /predict issued from INSIDE wine-app -> http://api:8000
       (proves the Streamlit -> API -> model path over the compose network)
    6. sha256sum inside wine-api == host sha256 -> FRESH MODEL VERIFIED
    7. Streamlit app /_stcore/health via `docker exec` in wine-app

Any failure exits non-zero so Airflow marks the task failed and retries.

Usage (Airflow task or manual):
    python services/airflow/scripts/deploy.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = PROJECT / "docker-compose.yml"
MODEL_ON_HOST = PROJECT / "models" / "model.joblib"
API_CONTAINER = "wine-api"
APP_CONTAINER = "wine-app"
MODEL_IN_API = "/app/models/model.joblib"
HEALTH_URL = "http://localhost:8000/health"

PREDICT_SCRIPT = """
import json, urllib.request
payload = json.dumps({
    "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red"}).encode()
req = urllib.request.Request("http://api:8000/predict", data=payload,
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
assert resp["prediction"] in (0, 1) and resp["label"] in ("Good", "Poor")
print("PREDICT_OK", json.dumps(resp))
"""


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sha256_of(path: Path) -> str:
    """SHA256 with small retry (transient empty reads right after writes)."""
    size = path.stat().st_size
    empty = hashlib.sha256(b"").hexdigest()
    for _ in range(5):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        if size == 0 or h.hexdigest() != empty:
            return h.hexdigest()
        time.sleep(0.2)
    return h.hexdigest()


def run(cmd: list[str], input_text: str | None = None) -> None:
    """Run a command, stream stdout/stderr live, raise on failure."""
    print(f"[deploy] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT), input=input_text,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit={proc.returncode}): {' '.join(cmd)}")


def compose(*args: str) -> None:
    run(["docker", "compose", "-f", str(COMPOSE_FILE), *args])


def exec_in(container: str, cmd: list[str], input_text: str | None = None) -> str:
    """docker exec returning captured stdout (for parseable checks)."""
    proc = subprocess.run(["docker", "exec", "-i", container, *cmd],
                          cwd=str(PROJECT), input=input_text, text=True,
                          encoding="utf-8", errors="replace",
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"docker exec {container} failed: {(proc.stderr or '').strip()}")
    return (proc.stdout or "").strip()


def main() -> None:
    header("STAGE 7 | DEPLOY - DOCKER COMPOSE REBUILD + RESTART")
    if not MODEL_ON_HOST.exists():
        raise FileNotFoundError(f"missing artifact: {MODEL_ON_HOST}")
    host_sha = sha256_of(MODEL_ON_HOST)
    print(f"[deploy] fresh model artifact : {MODEL_ON_HOST.name}")
    print(f"[deploy] sha256 (host)        : {host_sha}")

    header("1-3 | docker compose build + up -d")
    compose("build")
    compose("up", "-d")

    header("4 | API /health (inside wine-api, model_loaded must be true)")
    health_cmd = ["python", "-c",
                  "import json,sys,urllib.request;"
                  "d=json.loads(urllib.request.urlopen('" + HEALTH_URL + "',timeout=3).read().decode());"
                  "print('HEALTH_OK', json.dumps(d));"
                  "sys.exit(0 if d.get('model_loaded') else 1)"]
    last_err = ""
    for attempt in range(1, 31):
        try:
            out = exec_in(API_CONTAINER, health_cmd)
            print(f"[deploy] {out}")
            break
        except RuntimeError as exc:
            last_err = str(exc)
            print(f"[deploy] waiting for API health ({attempt}/30)...")
            time.sleep(2)
    else:
        raise RuntimeError(f"API never became healthy: {last_err}")

    header("5 | /predict from wine-app -> http://api:8000 (Streamlit -> API -> model)")
    out = exec_in(APP_CONTAINER, ["python", "-"], input_text=PREDICT_SCRIPT)
    line = [l for l in out.splitlines() if l.startswith("PREDICT_OK")][-1]
    print(f"[deploy] {line}")

    header("6 | FRESH MODEL VERIFICATION (sha256 inside wine-api vs host)")
    api_sha = exec_in(API_CONTAINER, ["sha256sum", MODEL_IN_API]).split()[0]
    print(f"[deploy] sha256 (wine-api)    : {api_sha}")
    if api_sha != host_sha:
        raise RuntimeError(f"STALE MODEL: api {api_sha} != host {host_sha}")
    print(f"[deploy] FRESH MODEL VERIFIED sha256={host_sha}")

    header("7 | Streamlit app health (inside wine-app)")
    last_err = ""
    for attempt in range(1, 16):
        try:
            out = exec_in(APP_CONTAINER, ["python", "-c",
                                          "import urllib.request;"
                                          "print(urllib.request.urlopen("
                                          "'http://localhost:8501/_stcore/health',"
                                          "timeout=5).read().decode())"])
            break
        except RuntimeError as exc:
            last_err = str(exc)
            print(f"[deploy] waiting for Streamlit health ({attempt}/15)...")
            time.sleep(2)
    else:
        raise RuntimeError(f"Streamlit never became healthy: {last_err}")
    if out != "ok":
        raise RuntimeError(f"streamlit health returned: {out!r}")
    print(f"[deploy] streamlit health: {out}")

    header("DEPLOY SUMMARY")
    print(f"  model sha256 (host & wine-api) : {host_sha}")
    print(f"  api health + predict           : OK")
    print(f"  streamlit -> api               : OK")
    print("\nDEPLOY: SUCCESS")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
