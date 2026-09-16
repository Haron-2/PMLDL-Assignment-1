# Stage 3 — Model Engineering Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 3 of 8 — `code/models/train.py` + `code/models/evaluate.py`
**Date:** 2026-09-16
**Input:** Stage 2 artifacts (`data/processed/`, deterministic split), Stage 2 code (imported)
**Status:** ✅ Completed and verified

---

## 1. Executive Summary

Trained a `RandomForestClassifier` (300 trees, `random_state=42`) inside a
single sklearn `Pipeline` together with the Stage 2 `ColumnTransformer`. The
one artifact `models/model.joblib` maps **raw** features (11 numeric +
`wine_type`) to a prediction — training/serving consistency by construction.
Test metrics: **accuracy 0.7679, precision 0.7963, recall 0.8453, F1 0.8201,
ROC-AUC 0.8389** (+14.2 pp over the majority-class baseline 0.6259). All
params/metrics/model are tracked in MLflow (run
`936592c4554e4b32899faa2a76abb247`); a standalone re-evaluation reproduced
every metric exactly (6/6 checks, tol 1e-9).

## 2. Objective

- train the classifier required by the assignment on the Stage 2 split;
- produce ONE serializable artifact usable by the Stage 4 API without
  duplicating preprocessing code;
- log the experiment to MLflow (params, metrics, tags, model, dataset info);
- independently verify reproducibility of the saved artifact.

## 3. Methods / Implementation

| Aspect | Implementation |
|--------|----------------|
| Data flow | `preprocess.py` functions imported and reused verbatim (`load_raw` → validate → dedup → target → split) — no duplicated cleaning logic |
| Model | `RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)` |
| Artifact | `Pipeline([preprocessor (Stage 2 ColumnTransformer), classifier])` → `models/model.joblib` (32,535,554 B) |
| Consistency | The pipeline's fitted preprocessor must transform `X_train` **bitwise-identically** to Stage 2's `data/processed/train.csv` (asserted, PASS) |
| Metrics | accuracy, precision, recall, F1 (binary, positive class = Good), ROC-AUC, confusion matrix |
| Tracking | MLflow 3.16.0, local sqlite backend `mlflow.db`, experiment `wine-quality-prediction` |
| Verification | `evaluate.py`: reload artifact → rebuild test split → recompute → compare vs `models/metrics.json` (tol 1e-9) → render confusion-matrix figure |

### Why a single Pipeline artifact?

The Stage 4 FastAPI service receives **raw** feature values. Embedding the
preprocessor in the same object as the classifier means the API loads one
file and cannot drift from training-time preprocessing (no "second copy" of
the transformation logic at serving time). `preprocessor.joblib` from Stage 2
remains as the documented fitted-transform artifact.

## 4. Results

### Test metrics (n = 1064, single artifact on raw features)

| Metric | Value |
|--------|-------|
| Accuracy | **0.7679** |
| Precision (Good = 1) | 0.7963 |
| Recall (Good = 1) | 0.8453 |
| F1 (Good = 1) | 0.8201 |
| ROC-AUC | 0.8389 |
| Majority-class baseline | 0.6259 (666/1064) → RF is **+14.2 pp** |

Confusion matrix: TN = 254, FP = 144, FN = 103, TP = 563
(`notebooks/figures/fig6_confusion_matrix.png`).

Per-class (`classification_report`): Poor — P 0.7115 / R 0.6382 / F1 0.6728;
Good — P 0.7963 / R 0.8453 / F1 0.8201. The model is biased towards the
majority "Good" class, as expected on a 62.6/37.4 split without resampling
(out of Stage 1 scope; a candidate for future work, not a Stage 3 decision).

### MLflow run

| Field | Value |
|-------|-------|
| Experiment | `wine-quality-prediction` |
| Run | `936592c4554e4b32899faa2a76abb247` |
| Backend | sqlite: `mlflow.db` (Git-ignored) |
| Params | 9 (RF hyperparams + split config) |
| Metrics | 5 |
| Tags | model type, dataset description, row counts, feature counts, sklearn version |
| Model artifact | `model/` (MLmodel + skops serialization, pip requirements pinned) |
| Additional artifact | `metrics.json` |

## 5. Validation Results

- Preprocessing consistency (pipeline vs Stage 2 CSV): shape ✓, column order ✓,
  bitwise value equality ✓ → **PASS**.
- `evaluate.py` independent re-evaluation vs `metrics.json`: 5 metrics within
  1e-9 + confusion matrix equality → **6/6 PASS**.
- MLflow round-trip: run visible in `mlflow.db`, model artifact saved with
  pinned pip requirements.

## 6. Reproducibility

- Fixed `random_state=42` for the split (Stage 2) and the forest; `n_jobs`
  affects only wall-time, not the fitted ensemble (sklearn seeds each tree
  deterministically).
- Full chain `preprocess.py → train.py → evaluate.py` was executed
  back-to-back from a clean state; reruns give identical metrics (verified
  twice in this stage).

## 7. Limitations & Notes

- MLflow ≥ 3 requires opting out of the legacy file store; the project uses
  the recommended local **sqlite** backend instead of `mlruns/`.
- MLflow ≥ 3 serializes sklearn models via skops; `sklearn.tree._tree.Tree`
  is passed as a trusted type when logging (the model is our own, produced
  deterministically by this repo's code).
- Accuracy 0.7679 reflects an untuned baseline (default-ish depth, 300
  trees). Hyperparameter search is out of scope for this stage.
- `model.joblib` is ~31 MB (300 trees) — fine for a local API; not tracked
  by Git (regenerable via `train.py`).

## 8. Security Considerations

- No network access during training; MLflow tracking is a local sqlite file.
- No secrets in code or logs; artifacts are Git-ignored and reproducible.
- The API (Stage 4) will only expose `predict` on validated inputs — the
  model object itself is never re-fitted at serving time.

## 9. Next Steps (Stage 4)

- FastAPI service: `POST /predict` (11 numeric + `wine_type`), `GET /health`,
  Pydantic validation (types, physical ranges, `free ≤ total` SO2), 422/503
  error contract, `test_api.py` running against a real uvicorn server.

