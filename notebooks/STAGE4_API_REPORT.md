# Stage 4 — FastAPI Model-Serving API Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 4 of 8 — `code/deployment/api/` (`main.py`, `schemas.py`, `test_api.py`)
**Date:** 2026-09-16
**Status:** ✅ Completed — 7/7 integration tests pass against a real uvicorn server

---

## 1. Objective

Expose the Stage 3 single-artifact `Pipeline` (`models/model.joblib`) over
HTTP with strict request validation and a documented error contract, so the
Streamlit UI (Stage 5) and any other client can get predictions from **raw**
feature values.

## 2. Endpoints

| Method & path | Purpose | Success | Errors |
|---------------|---------|---------|--------|
| `GET /` | service info | 200 | — |
| `GET /health` | liveness + model load status | 200 (always) | — |
| `POST /predict` | predict from 11 numeric features + `wine_type` | 200 | 422 / 503 / 500 |

## 3. Request / Response

Request keys use the **dataset column names** (Pydantic aliases), e.g.:

```json
{"fixed acidity": 7.4, "volatile acidity": 0.66, "citric acid": 0.0,
 "residual sugar": 1.8, "chlorides": 0.075, "free sulfur dioxide": 13,
 "total sulfur dioxide": 40, "density": 0.9978, "pH": 3.51,
 "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red"}
```

Response:

```json
{"prediction": 0, "label": "Poor", "probability_good": 0.0333,
 "model_version": "rf-300-baseline"}
```

## 4. Validation Contract (schemas.py)

- all numeric fields are strictly typed floats (strings like `"abc"` → 422);
- physical bounds: non-negative concentrations; `density ∈ [0.9, 1.1]`;
  `pH ∈ [0, 14]`; `alcohol ≤ 100`;
- `wine_type` is a `Literal["red", "white"]` enum;
- cross-field rule: `free sulfur dioxide ≤ total sulfur dioxide`
  (`model_validator(mode="after")`).
- bounds are deliberately generous (physical, not data-min/max) so legitimate
  unseen wines are accepted while garbage is rejected.

## 5. Error Contract

- **422** — request validation failure (FastAPI/Pydantic, with per-field detail);
- **503** — model artifact not loaded (`/predict` only);
- **500** — unexpected prediction failure (wrapped, message preserved).

The model loads once at startup (FastAPI lifespan). `MODEL_PATH` env var
overrides the default `<project>/models/model.joblib` (used by Docker);
`MODEL_VERSION` labels responses.

## 6. Test Results (`test_api.py`, real uvicorn subprocess, port 8123)

| # | Case | Result |
|---|------|--------|
| 1 | `GET /health` → 200, `model_loaded=true` (type=Pipeline) | PASS |
| 2 | valid red sample → 200, consistent schema (`Poor`, p=0.0333) | PASS |
| 3 | valid white sample → 200 (`Poor`, p=0.4033) | PASS |
| 4 | missing `alcohol` → 422 | PASS |
| 5 | `wine_type="rose"` → 422 | PASS |
| 6 | `free SO2 (200) > total SO2 (100)` → 422 | PASS |
| 7 | `alcohol="abc"` → 422 | PASS |

**7/7 passed.** Interactive OpenAPI docs are available at `/docs`.

## 7. Security Considerations

- The model is loaded read-only; no refitting, no filesystem writes, no
  shell-outs at request time.
- Validation is fail-fast before the model sees any data.
- No secrets; the service binds to localhost/`0.0.0.0` inside the Docker
  network only (host port mapping controlled by Docker Compose).
- Container installs pinned dependencies (see `code/deployment/api/requirements.txt`).

## 8. Limitations & Notes

- Single-worker uvicorn is sufficient for this assignment; horizontal
  scaling would need a shared model store and multiple replicas.
- No persistent request logging/analytics (out of scope).
- CORS is intentionally not enabled: the Streamlit app calls the API
  server-side (container-to-container), not from the browser.
