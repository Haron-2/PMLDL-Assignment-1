# FINAL PROJECT REPORT — Wine Quality Prediction MLOps Pipeline

**Project:** PMLDL Assignment 1
**Author:** MLOps pipeline: Data → Model → API → UI → Docker → Airflow
**Date:** 2026-09-17
**Repo:** `PMLDL-Assignment-1` (all 8 stages committed)
**Status:** ✅ **All 8 stages completed and verified**

---

## 1. Executive Summary

A complete MLOps pipeline was built for binary wine-quality prediction
(Good: `quality > 5`) on the merged UCI red + white Wine Quality datasets.
The chain — deterministic preprocessing → `RandomForestClassifier` training
with MLflow tracking → FastAPI serving → Streamlit UI → Docker Compose
deployment → Airflow 5-minute orchestration — is fully reproducible from
`data/raw` and verified by automated checks at every stage (preprocessing
13/13, model consistency, independent re-evaluation 6/6, API 7/7,
container E2E, Airflow scheduled runs, integration test 8/8).

**Final model quality (test set, n=1064):** accuracy **0.7679**, precision
**0.7963**, recall **0.8453**, F1 **0.8201**, ROC-AUC **0.8389**
(+14.2 pp accuracy over the 0.6259 majority-class baseline).

## 2. Objectives (per assignment)

| Requirement | Status |
|-------------|--------|
| Download + merge UCI red/white with `wine_type` | ✅ Stage 1 (`download.py`) |
| EDA with cleaning/outlier/target decisions | ✅ Stage 1 (`explore.py`, report + 5 figures) |
| Data engineering: clean → target → split → preprocess | ✅ Stage 2 (`preprocess.py`) |
| RandomForest training + evaluation | ✅ Stage 3 (`train.py`, `evaluate.py`) |
| MLflow experiment tracking | ✅ Stage 3 (sqlite backend, `mlflow.db`) |
| FastAPI serving endpoint | ✅ Stage 4 (`code/deployment/api/`) |
| Streamlit frontend | ✅ Stage 5 (`code/deployment/app/`) |
| Docker + Docker Compose (2 containers) | ✅ Stage 6 (API + UI, healthcheck) |
| Airflow DAG, 5-min schedule | ✅ Stage 7 (`wine_quality_pipeline`) |
| Tests + final report | ✅ Stage 8 (`integration_test.py`, this document) |

## 3. Final Architecture

```text
data/raw (UCI) ──► preprocess.py ──► train.py ──► model.joblib ──► FastAPI :8000 ◄── Streamlit :8501
                     │ (Stage 2)       │ (Stage 3)                   ▲                ▲
                     ▼                 ▼                             │   Docker Compose│
              data/processed/     MLflow (mlflow.db) ────────────────┘   (Stage 6)     │
              models/preprocessor.joblib                                                │
                                                                                ────┘
Airflow (Stage 7): every 5 min: verify_raw ─► preprocess ─► train_model ─► evaluate_model
```

Design decisions that carry through all stages:

1. **Single source of truth** — `preprocess.py` exposes importable functions;
   training, evaluation, the Airflow DAG and the artifact all use the same
   logic. No transformation is ever duplicated.
2. **One-artifact serving** — `models/model.joblib` is a single sklearn
   `Pipeline` (preprocessor + classifier) mapping raw features to a
   prediction; serving cannot drift from training.
3. **Leakage prevention** — exact duplicates removed *before* the split;
   preprocessor fitted on train only (asserted bitwise-equal to Stage 2 CSVs).
4. **Determinism** — `random_state=42` everywhere; two runs produce identical
   SHA-256 artifacts (Stage 2) and identical metrics (Stages 3, 8).

## 4. Stage-by-Stage Results

| Stage | Deliverables | Key result |
|-------|--------------|------------|
| 1 — EDA | `explore.py`, `STAGE1_EDA_REPORT.md`, 5 figures | 6497 rows; 1177 exact duplicates; red loses 48% under drop-all-outliers → outliers kept; target `>5 → 1` (62.6% Good) |
| 2 — Data engineering | `preprocess.py`, `train.csv`/`test.csv`/`preprocessor.joblib` | 5320 clean rows → 4256/1064 stratified; 13 output features; 13/13 validation checks; byte-identical reruns |
| 3 — Model engineering | `train.py`, `evaluate.py`, `model.joblib`, `metrics.json` | accuracy 0.7679 / F1 0.8201 / ROC-AUC 0.8389; consistency check PASS; MLflow run `936592c4…`; re-evaluation 6/6 PASS |
| 4 — API | `main.py`, `schemas.py`, `test_api.py` | `/health`, `/predict`, `/docs`; dataset-named aliases; 422/503/500 contract; **7/7 tests** |
| 5 — UI | `app.py` | sliders (dataset ranges), live backend health, probability display; headless smoke test OK |
| 6 — Docker | 2 Dockerfiles, `docker-compose.yml`, `.dockerignore`, `e2e_check.py` | both images build; API healthy; host + container-to-container E2E **PASSED** (probabilities identical to local) |
| 7 — Airflow | `Dockerfile`, compose, `wine_quality_pipeline` DAG, `verify_raw.py` | manual **success**; scheduled runs (2×) **success**; 4/4 tasks green (0.4s/3.3s/19.2s/7.7s) |
| 8 — Tests & report | `integration_test.py`, `FINAL_PROJECT_REPORT.md` | **8/8 integration checks** re-running the whole chain from raw data |

## 5. Model Engineering Summary

- **Algorithm:** `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)`
  inside `Pipeline([ColumnTransformer, RF])`.
- **Preprocessing:** 11 numeric features passthrough; `wine_type` one-hot
  (`wine_type_red`, `wine_type_white`); `quality` excluded from features.
- **Split:** 80/20 stratified, `random_state=42` (4256 train / 1064 test).
- **Confusion matrix:** TN 254 · FP 144 · FN 103 · TP 563.
- **Per-class:** Poor P 0.7115 / R 0.6382 / F1 0.6728; Good P 0.7963 / R 0.8453 / F1 0.8201.
- **MLflow:** experiment `wine-quality-prediction` (sqlite `mlflow.db`);
  9 params, 5 metrics, tags, pinned-pip model artifact per run; the Airflow
  schedule appends a new run every 5 minutes by design.

## 6. Deployment Summary

- **Docker Compose (`docker-compose.yml`):** `wine-quality-api` (:8000) +
  `wine-quality-app` (:8501); API healthcheck gates the UI
  (`service_healthy`); model baked into the API image; pinned slim images.
- **Airflow (`services/airflow/`):** single-container `airflow standalone`
  (SequentialExecutor + sqlite) with a task-venv pinned to the exact training
  library versions; repo bind-mounted; UI on :8080; DAG chain
  `verify_raw_data → preprocess → train_model → evaluate_model`,
  `*/5 * * * *`, `catchup=False`, `retries=2`.

## 7. Testing & Verification Matrix

| Layer | Test | Result |
|-------|------|--------|
| Data | `validate_dataframe` fail-fast + 13 output checks | 13/13 PASS |
| Data | byte-identical reruns (SHA-256 of 3 artifacts) | PASS |
| Model | pipeline-vs-CSV bitwise preprocessing consistency | PASS |
| Model | independent re-evaluation vs `metrics.json` | 6/6 PASS |
| API | `test_api.py` (real uvicorn): schema, bounds, SO2 rule, types | 7/7 PASS |
| Docker | build, healthcheck, host E2E, container→container E2E | PASSED |
| Airflow | import errors, manual run, 2 scheduled runs, task states | success |
| Full chain | `code/tests/integration_test.py` | 8/8 PASS |

## 8. Reproducibility Statement

From a clean clone, the entire system regenerates from tracked code + raw
CSVs: `pip install -r requirements.txt` → `download.py` → `preprocess.py` →
`train.py` → `evaluate.py` → `test_api.py` → `docker compose up` → Airflow
compose. Verified end-to-end during Stage 8 (integration test re-runs
stages 1–5 equivalents and reports 8/8). Container images pin exact library
versions; the Airflow task venv pins the training versions, so models load
without version drift anywhere in the system.

## 9. Security Considerations

- No secrets in the repo; `.env` patterns Git-ignored.
- Fail-fast validation on every input boundary (dataset load, API request).
- Serving is read-only over the model; no refit, no disk writes at request time.
- Artifacts (`data/`, `models/`, `mlflow.db`, logs) Git-ignored and
  regenerable; container images contain no datasets or source-tree extras.
- Services bound to localhost/Compose-network only.

## 10. Limitations & Future Work

- The model is an untuned baseline; class-weighting/SMOTE could improve the
  minority (Poor) recall; hyperparameter search (Optuna) is a natural next step.
- Streamlit slider ranges cap at observed min/max; the API accepts wider
  physical ranges.
- Airflow uses SequentialExecutor + sqlite (assignment scale); production
  would use LocalExecutor/Celery + PostgreSQL.
- No CI pipeline yet — `integration_test.py` is CI-ready as a single command.

## 11. Quickstart

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

.venv\Scripts\python.exe code\datasets\download.py      # 1. raw data
.venv\Scripts\python.exe code\datasets\preprocess.py    # 2. preprocessing
.venv\Scripts\python.exe code\models\train.py           # 3. train + MLflow
.venv\Scripts\python.exe code\models\evaluate.py        #    independent eval

.venv\Scripts\python.exe -m uvicorn main:app --app-dir code\deployment\api --port 8000   # 4. API
.venv\Scripts\python.exe -m streamlit run code\deployment\app\app.py                     # 5. UI

docker compose up -d --build                                             # 6. deployment
docker compose -f services\airflow\docker-compose.airflow.yml up -d --build  # 7. Airflow (:8080)

.venv\Scripts\python.exe code\tests\integration_test.py  # 8. verify everything (8/8)
```

## 12. Conclusion

All eight stages of the assignment are implemented, documented (per-stage
reports in `notebooks/`) and verified by automated tests. The pipeline is
deterministic, leakage-free, containerized, orchestrated on a 5-minute
schedule, and reproducible end-to-end from raw data with a single
integration command.

**Final artifacts:** `models/model.joblib` (Pipeline), `models/metrics.json`,
`data/processed/{train,test}.csv`, `models/preprocessor.joblib`,
`mlflow.db` (experiment history), 6 report figures, 8 stage reports +
this final report.
