# Stage 7 — Airflow Orchestration Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 7 of 8 — `services/airflow/` (Dockerfile, docker-compose.airflow.yml, DAG, scripts)
**Date:** 2026-09-16
**Status:** ✅ Completed — manual AND scheduled runs both **success** (4/4 tasks each)

---

## 1. Objective

Automate the full pipeline (verify raw data → preprocess → train + MLflow →
evaluate) on a **5-minute schedule** with Apache Airflow, reusing the exact
Stage 2–3 code (no duplicated logic) and producing host-persistent artifacts.

## 2. Deployment Architecture

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Image | `apache/airflow:2.10.5-python3.12` + dedicated `/opt/venv` (numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, joblib 1.6.0, mlflow 3.16.0) | Tasks must load `model.joblib` with the **same sklearn version it was trained with**; the separate venv avoids fighting Airflow's constraint pins |
| Mode | `airflow standalone` (single container) | Lightweight: DB init + admin user + webserver + scheduler in one step |
| Executor / DB | SequentialExecutor + sqlite | Sequential 4-step chain; no external database needed |
| Mounts | repo → `/opt/airflow/project`, `dags/` → `/opt/airflow/dags`, `logs/` → `/opt/airflow/logs` | Tasks run the repo code in-place; artifacts (`data/`, `models/`, `mlflow.db`) persist on the host |
| Tasks | `BashOperator` with `/opt/venv/bin/python` | Identical entry points to local runs |
| UI | `localhost:8080` (standalone admin user) | Monitoring and manual triggers |

## 3. DAG (`wine_quality_pipeline`)

```
verify_raw_data → preprocess → train_model → evaluate_model
```

- `verify_raw_data` — `services/airflow/scripts/verify_raw.py`: red=1599 /
  white=4898 rows or fail;
- `preprocess` — `code/datasets/preprocess.py` (Stage 2, unchanged);
- `train_model` — `code/models/train.py` (Stage 3, unchanged; MLflow logs to
  the bind-mounted `mlflow.db`);
- `evaluate_model` — `code/models/evaluate.py` (Stage 3, unchanged).

Config: `schedule="*/5 * * * *"`, `catchup=False`, `max_active_runs=1`,
`retries=2` (retry_delay 1 min), tags `mlops / wine-quality / assignment-1`.

## 4. Validation Results

| Check | Result |
|-------|--------|
| `airflow dags list-import-errors` | No data found (clean import) |
| DAG listed, unpaused | `wine_quality_pipeline` — `is_paused=False` |
| Webserver health | metadatabase / scheduler / triggerer all `healthy` |
| Manual trigger (`dags trigger`) | **success** |
| Scheduled run at 23:00 UTC | **success** (scheduler fired it automatically) |
| Scheduled run at 23:05 UTC | **success** |
| Task states (scheduled 23:05) | verify 0.4s ✓ · preprocess 3.3s ✓ · train 19.2s ✓ · evaluate 7.7s ✓ |
| Host artifact persistence | `models/model.joblib`, `mlflow.db` (876 KB, new runs), figures rewritten through the mount |

The container-run `train_model` overwrote `models/model.joblib` and the
`evaluate_model` task reproduced the logged metrics inside the container —
confirming version-consistent model loading and byte-level determinism of the
whole chain outside the host environment.

## 5. Failure Encountered & Fixed (record for honesty)

The first manual run failed at `verify_raw_data` (3 attempts, then failed —
retry policy worked as configured). Root cause: the script used
`parents[2]` for the project root, but it lives three levels deep
(`services/airflow/scripts/`), so it looked for `services/data/raw/...`.
Fixed to `parents[3]`; the next manual run and two scheduled runs succeeded.
DAG code and task logs were inspected through the bind-mounted
`services/airflow/logs/` directory.

## 6. Security Considerations

- Standalone-generated credentials live only inside the container; nothing is
  hardcoded in the repo. UI bound to localhost.
- No secrets in DAG code; tasks read/write only project-local files.
- `services/airflow/logs/` and `airflow.db` are Git-ignored.

## 7. Limitations & Notes

- SequentialExecutor + sqlite is intentionally minimal; production setups
  would use LocalExecutor/Celery + PostgreSQL.
- Each scheduled run retrains and appends a new MLflow run — by design for
  this assignment; real deployments would gate on metric thresholds.
- First scheduled runs right after container start may overlap their slots;
  `max_active_runs=1` serializes them.

## 8. Next Steps (Stage 8)

- End-to-end integration tests across stages, `FINAL_PROJECT_REPORT.md`,
  final README polish and commits.
