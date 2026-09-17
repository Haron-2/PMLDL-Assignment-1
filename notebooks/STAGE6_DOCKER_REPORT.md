# Stage 6 — Docker / Docker Compose Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 6 of 8 — Dockerfiles, `docker-compose.yml`, `.dockerignore`
**Date:** 2026-09-16
**Status:** ✅ Completed — build + run + host E2E + container-to-container E2E all verified

---

## 1. Objective

Package the FastAPI service (Stage 4) and the Streamlit UI (Stage 5) as two
separate containers, orchestrated by Docker Compose, with the API made a
hard dependency of the UI and everything reproducible from the repo.

## 2. Architecture

| Component | Image | Base | Notes |
|-----------|-------|------|-------|
| `api` (`wine-quality-api`) | `code/deployment/api/Dockerfile` | `python:3.13-slim` | Dependencies pinned (`api/requirements.txt`); model **baked in** at `/app/models/model.joblib`; `MODEL_PATH` env points there; uvicorn entrypoint |
| `app` (`wine-quality-app`) | `code/deployment/app/Dockerfile` | `python:3.13-slim` | `API_URL=http://api:8000`; Streamlit headless on 8501 |

- Compose: `api` exposes `8000:8000`, `app` exposes `8501:8501`; `api` has an
  HTTP healthcheck (`urllib` against `/health`, 10s interval, 5 retries,
  30s start period — no extra tools needed in the slim image); `app` uses
  `depends_on: condition: service_healthy`.
- Build context is the repo root with a curated `.dockerignore` (excludes
  `.venv/`, `.git/`, `data/`, `notebooks/`, `services/`, caches).
- No bind mounts in the deployment compose — images are self-contained.

## 3. Build & Run Results

```
docker compose build   → wine-quality-api:latest ✓   wine-quality-app:latest ✓
docker compose up -d   → wine-api Up (healthy)  ✓     wine-app Up ✓
```

The UI container waited for the API healthcheck before starting
(`Container wine-api Healthy → wine-app Starting`), confirming the
dependency contract.

## 4. End-to-End Verification

**From the host:**

| Check | Result |
|-------|--------|
| `GET /health` | 200, `model_loaded=true`, `model_type=Pipeline` |
| `POST /predict` valid red sample | 200 → `Poor`, p=0.0467, `rf-300-baseline` — **identical to the local Stage 4 run** (same artifact; current post-IQR model) |
| `POST /predict` invalid (missing fields) | **422** |
| `GET :8501/_stcore/health` | 200, body `ok` |
| `GET :8501/` | 200 (Streamlit SPA shell) |

**Container-to-container** (`code/deployment/app/e2e_check.py`, copied into
`wine-app` and executed there — the exact path the UI uses):

| Check | Result |
|-------|--------|
| `GET http://api:8000/health` | 200, `model_loaded=true`, path `/app/models/model.joblib` |
| `POST http://api:8000/predict` white sample | 200 → `Good`, p=0.6867 — **byte-identical probability to the local artifact** (current post-IQR model) |

`E2E CHECK: PASSED` (exit 0). Compose network DNS (`http://api:8000`) works.

## 5. Security Considerations

- Images contain only what they serve; no source tree, datasets or secrets.
- Dependencies pinned per service for reproducible builds.
- The model is baked in read-only; the API never writes to disk at runtime.
- Host exposure is limited to the two published ports; service-to-service
  traffic stays on the internal Compose network.

## 6. Limitations & Notes

- SQLite/standalone choices in the Airflow stack (Stage 7) mirror the same
  "lightweight, reproducible" philosophy.
- `e2e_check.py` is docker-cp'ed into the running UI container for the E2E
  test (the image itself predates the file); Stage 8 reuses it as the
  integration test.
- Containers are stopped after verification (`docker compose down`); the
  images remain cached for quick restarts.

## 7. Next Steps (Stage 7)

Airflow orchestration of the training pipeline (Stage 7) — see
`notebooks/STAGE7_AIRFLOW_REPORT.md`.
