# Stage 7 — Airflow Orchestration Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 7 of 8 — `services/airflow/` (Dockerfile, docker-compose.airflow.yml, DAG, scripts)
**Date:** 2026-09-17
**Status:** ✅ Completed — 8 consecutive runs **success** (scheduled + manual), 5/5 tasks each, including the automated `deploy` task

---

## 1. Objective

Automate the full pipeline (verify raw data → preprocess → train + MLflow →
evaluate → deploy) on a **5-minute schedule** with Apache Airflow, reusing
the exact Stage 2–3 code (no duplicated logic), producing host-persistent
artifacts and finishing every run with an automated Docker deployment of the
freshly trained model (verified by live health/predict/Streamlit checks).

## 2. Deployment Architecture

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Image | `apache/airflow:2.10.5-python3.12` + dedicated `/opt/venv` (numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, joblib 1.6.0, mlflow 3.16.0) | Tasks must load `model.joblib` with the **same sklearn version it was trained with**; the separate venv avoids fighting Airflow's constraint pins |
| Mode | `airflow standalone` (single container) | Lightweight: DB init + admin user + webserver + scheduler in one step |
| Executor / DB | SequentialExecutor + sqlite | Sequential 5-task chain; no external database needed |
| Mounts | repo → `/opt/airflow/project`, `dags/` → `/opt/airflow/dags`, `logs/` → `/opt/airflow/logs`, Docker socket → `/var/run/docker.sock` | Tasks run the repo code in-place; artifacts (`data/`, `models/`, `mlflow.db`) persist on the host; `deploy` can drive Docker |
| Docker tooling | `docker-ce-cli` + compose plugin baked into the image; container runs as root (0:0) | The `deploy` task rebuilds/restarts the Compose stack from inside Airflow |
| Tasks | `BashOperator` with `/opt/venv/bin/python`, prefixed by `env -u PYTHONPATH -u PYTHONUSERBASE` (ENV_GUARD) | Identical entry points to local runs; the guard strips the runtime-injected `PYTHONPATH` that would shadow the venv's numpy/sklearn stack |
| UI | `localhost:8080` (standalone admin user) | Monitoring and manual triggers |

## 3. DAG (`wine_quality_pipeline`)

```
verify_raw_data → preprocess → train_model → evaluate_model → deploy
```

- `verify_raw_data` — `services/airflow/scripts/verify_raw.py`: red=1599 /
  white=4898 rows or fail;
- `preprocess` — `code/datasets/preprocess.py` (dedup → pooled-fence IQR
  outlier removal → target → split → transform);
- `train_model` — `code/models/train.py` (MLflow logs to the bind-mounted
  `mlflow.db`);
- `evaluate_model` — `code/models/evaluate.py`;
- `deploy` (fifth and last task) — `services/airflow/scripts/deploy.py`:
  `docker compose build && docker compose up -d` through the mounted Docker
  socket, then verifies the served model SHA-256 equals the freshly trained
  `models/model.joblib`, `GET /health`, `POST /predict` and Streamlit
  `/_stcore/health` (retry-polling 15×2 s).

Config: `schedule="*/5 * * * *"`, `catchup=False`, `max_active_runs=1`,
`retries=2` (retry_delay 1 min), tags `mlops / wine-quality / assignment-1`.

## 4. Validation Results

| Check | Result |
|-------|--------|
| `airflow dags list-import-errors` | No data found (clean import) |
| DAG listed, unpaused | `wine_quality_pipeline` — `is_paused=False` |
| Webserver health | metadatabase / scheduler / triggerer all `healthy` |
| Consecutive runs | 8× **success** (scheduled 09:05 → 10:05 UTC + manual triggers) |
| Task states (manual run, 10:11:53 UTC) | verify 0.5s ✓ · preprocess 4.5s ✓ · train 16.8s ✓ · evaluate 5.2s ✓ · **deploy 6.8s ✓** |
| Task states (scheduled run, 09:45 UTC) | verify 0.4s ✓ · preprocess 3.1s ✓ · train 15.2s ✓ · evaluate 4.3s ✓ · **deploy 9.0s ✓** |
| Post-run live checks | `/health` → ok + `model_loaded=true`; `/predict` (red sample) → `Poor`, p=0.0533; `/_stcore/health` → `ok`; `wine-api` healthy, `wine-app` up |
| Fresh-model verification | served model SHA-256 == freshly trained `models/model.joblib` SHA-256 (checked by `deploy.py`) |
| Host artifact persistence | `models/model.joblib`, `mlflow.db` (new runs), figures rewritten through the mount |

Each run's `train_model` overwrites `models/model.joblib`, `evaluate_model`
reproduces the logged metrics inside the container, and the final `deploy`
task rebuilds/restarts the Docker Compose API + UI containers with that
artifact and verifies the live endpoints — the full loop (raw data →
verified fresh deployment) runs unattended every 5 minutes.

## 5. Failure Encountered & Fixed (record for honesty)

The first manual run failed at `verify_raw_data` (3 attempts, then failed —
retry policy worked as configured). Root cause: the script used
`parents[2]` for the project root, but it lives three levels deep
(`services/airflow/scripts/`), so it looked for `services/data/raw/...`.
Fixed to `parents[3]`; the next manual run and two scheduled runs succeeded.
DAG code and task logs were inspected through the bind-mounted
`services/airflow/logs/` directory.

Two further issues were hit and fixed during the automated-deployment work:

1. **Numpy shadowing (`AttributeError: module 'numpy' has no attribute 'long'`).**
   The image entrypoint injects
   `PYTHONPATH=/home/airflow/.local/lib/python3.12/site-packages:` at runtime,
   shadowing the task-venv numpy (2.5.3) with Airflow's pinned 1.26.4.
   Proven both ways with a probe task; fixed by prefixing every task command
   with `env -u PYTHONPATH -u PYTHONUSERBASE` (the DAG's `ENV_GUARD`) and
   adding `ENV PYTHONNOUSERSITE=1` to the image as defense-in-depth.
2. **Streamlit health check raced the container restart.** The first deploy
   attempt verified the API but reached `/_stcore/health` while `wine-app`
   was still restarting. `deploy.py` now retry-polls the Streamlit health
   endpoint (15×2 s); the retried run passed, and all subsequent runs pass.

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

## 8. Follow-up (Stage 8 — completed)

- End-to-end verification across stages, `FINAL_PROJECT_REPORT.md`, README
  finalization and commits — see `notebooks/FINAL_PROJECT_REPORT.md`.
- Known follow-up: `code/tests/integration_test.py` still expects the
  pre-IQR split (4256/1064); its artifact-count check needs a one-line sync
  to the current 4227/1057.
