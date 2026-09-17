# Stage 2 — Data Engineering / Preprocessing Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 2 of 8 — `code/datasets/preprocess.py`
**Date:** 2026-09-16
**Input:** Stage 1 EDA decisions (`notebooks/STAGE1_EDA_REPORT.md`), raw UCI data (`data/raw/`)
**Status:** ✅ Completed and verified

---

## 1. Executive Summary

Implemented the deterministic data engineering pipeline specified in Stage 1
(with the post-EDA refinement of a targeted outlier rule): raw red + white
UCI datasets are merged (6497 × 13), validated with fail-fast checks,
deduplicated (1177 exact duplicates removed → 5320 rows), filtered by
pooled-fence IQR outlier removal (36 rows, 0.68% → 5284 rows), given a
binary target (`quality > 5 → 1`), split 80/20 with stratification
(4227 / 1057), and preprocessed with a `ColumnTransformer` fitted **on the
train split only**. All artifacts are persisted and verified by 13 automated
checks; the pipeline is deterministic — consecutive runs produce
**byte-identical outputs** (matching SHA-256).

## 2. Objective

Turn the Stage 1 specification into executable, reproducible code:

- implement `preprocess.py` exactly per the Stage 1 report (§8);
- guarantee no data leakage (dedup before split, fit on train only);
- persist artifacts in the agreed formats;
- validate every invariant automatically;
- keep the module importable so later stages (training, evaluation, API,
  Airflow) reuse the same logic — a single source of truth.

## 3. Methods / Implementation

| Step | Implementation |
|------|----------------|
| Loading | `pandas.read_csv(sep=";")` for both files; `wine_type` column added (`red`/`white`); `pd.concat` → 6497 × 13 |
| Validation | Fail-fast `validate_dataframe`: exact schema, no NaN, numeric dtypes, integer quality ∈ [0, 10], `wine_type ⊆ {red, white}`, `free SO2 ≤ total SO2`, no negative feature values |
| Duplicates | `drop_duplicates` across **all 13 columns**, **before** the split (Stage 1 leakage rule) |
| IQR outliers | Pooled (red+white) fences, 3.0×IQR; rows flagged in **≥ 3 features** dropped — 36 rows (0.68%) removed → 5284 rows |
| Target | `target = (quality > 5).astype(int)`; `quality` is **not** a feature |
| Split | `train_test_split(test_size=0.2, random_state=42, stratify=target)` |
| Preprocessing | `ColumnTransformer`: 11 numeric → `passthrough` (RandomForest needs no scaling); `wine_type` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`; `verbose_feature_names_out=False` for clean names |
| Fitting | Preprocessor fitted **on `X_train` only** (n = 4227); train and test transformed with the same fitted object |
| Persistence | `joblib.dump` for the fitted preprocessor; `to_csv(index=False)` for datasets |
| Paths | All paths derived from `Path(__file__).resolve().parents[2]` → CWD-independent (verified by running from a different directory) |

### Design decision — reusable functions

`load_raw`, `validate_dataframe`, `drop_exact_duplicates`, `add_target`,
`make_split`, `build_preprocessor` are importable. Stage 3 rebuilds the
deterministic split from these functions so the served artifact
(`Pipeline(preprocessor + classifier)`) and the training data can never drift.

## 4. Results

| Metric | Value |
|--------|-------|
| Raw combined rows | 6497 (1599 red + 4898 white) |
| Exact duplicates removed | 1177 (18.12%) |
| Cleaned rows | 5320 |
| IQR outliers removed | 36 (0.68%) → 5284 rows |
| Train / test rows | 4227 / 1057 (80/20, stratified) |
| Output features | 13 (11 numeric passthrough + `wine_type_red` + `wine_type_white`) |
| Target distribution (full) | 0: 1969 (37.3%) · 1: 3315 (62.7%) |
| Target distribution (train) | 0: 1575 · 1: 2652 (62.7% good) |
| Target distribution (test) | 0: 394 · 1: 663 (62.7% good) |

### Artifacts written

| File | Size | SHA-256 |
|------|------|---------|
| `data/processed/train.csv` | 280,453 B | `574005e669b64d414460f3e335c2c937946820cbaf7e6ce64187571d85cf7a68` |
| `data/processed/test.csv` | 70,326 B | `12a8581084c2f914e811da5d5e6c48c71ec206b4f5c323e2295dd61476f21ff6` |
| `models/preprocessor.joblib` | 2,676 B | `00f8caca0d79b11ac26c99d39270814a892b1791b51317ef4d8e64db6592fd5b` |

### Output schema (both CSVs)

```
fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol,
wine_type_red, wine_type_white, target
```

No `quality` column, no NaN, `target ∈ {0, 1}` — as required.

## 5. Validation Results

Automated `OUTPUT VALIDATION` — **13 / 13 checks PASS**:

- train rows == 4227; test rows == 1057;
- no NaN in either file;
- `quality` absent, `target` present with values in {0, 1};
- `wine_type_red` / `wine_type_white` present;
- feature count == 13 and column order == features + `[target]`;
- all three artifacts exist on disk.

Runtime validations also passed for both the raw combined frame and the
post-dedup frame (schema, NaN, dtypes, wine_type domain, SO2 sanity, quality
range, non-negative features).

## 6. Reproducibility

- Fixed `random_state=42` for the split; no other stochastic step exists in
  Stage 2 (passthrough + one-hot are deterministic).
- Two consecutive executions (one from the project root, one from a different
  working directory) produced **identical SHA-256 for all three artifacts**.
- Every run re-derives everything from `data/raw/`; no cached intermediate
  state.

## 7. Limitations & Notes

- The 18.12% duplicate rate matches Stage 1 (white wines dominate the
  duplicate pool); duplicates are exact across all 13 columns, so no label
  contradiction is possible.
- IQR outliers are **not** removed wholesale: Stage 1 analysis showed a
  drop-all rule would discard ~20.6% of rows and skew the red/white mix, so
  the implemented rule is narrowly targeted — pooled-fence IQR (3.0×IQR) with
  a ≥3-flag threshold, removing only 36 rows (0.68%).
- Implementation note: immediately after writing files, a transient
  Windows AV/FS lock can make the first read of a just-written file return
  empty bytes; `sha256_of` therefore retries briefly instead of silently
  hashing an empty read.
- `handle_unknown="ignore"` on the one-hot encoder protects the serving path
  (Stage 4) against unseen categories at prediction time.

## 8. Security Considerations

- No network access, no user input, no secrets; reads only local CSVs and
  writes only project-local artifacts.
- All inputs are validated fail-fast (schema and value domain) before any
  transformation runs.
- Generated artifacts (`data/processed/`, `models/`) are Git-ignored;
  everything is regenerable from tracked code + raw data.

## 9. Next Steps (Stage 3)

- Train `RandomForestClassifier` inside a single
  `Pipeline(preprocessor, classifier)` artifact (`models/model.joblib`) —
  one sklearn object from raw features to prediction (training/serving
  consistency).
- Track hyperparameters, dataset statistics and metrics in MLflow.
- Standalone `evaluate.py` re-computing test metrics from the saved artifact
  and rendering the confusion matrix.
