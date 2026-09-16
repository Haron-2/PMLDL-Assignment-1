# Stage 2 — Data Engineering / Preprocessing Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 2 of 8 — `code/datasets/preprocess.py`
**Date:** 2026-09-16
**Input:** Stage 1 EDA decisions (`notebooks/STAGE1_EDA_REPORT.md`), raw UCI data (`data/raw/`)
**Status:** ✅ Completed and verified

---

## 1. Executive Summary

Implemented the deterministic data engineering pipeline specified in Stage 1:
raw red + white UCI datasets are merged (6497 × 13), validated with fail-fast
checks, deduplicated (1177 exact duplicates removed → 5320 rows), given a
binary target (`quality > 5 → 1`), split 80/20 with stratification
(4256 / 1064), and preprocessed with a `ColumnTransformer` fitted **on the
train split only**. All artifacts are persisted and verified by 13 automated
checks; two consecutive runs produce **byte-identical outputs** (matching
SHA-256).

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
| Target | `target = (quality > 5).astype(int)`; `quality` is **not** a feature |
| Split | `train_test_split(test_size=0.2, random_state=42, stratify=target)` |
| Preprocessing | `ColumnTransformer`: 11 numeric → `passthrough` (RandomForest needs no scaling); `wine_type` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`; `verbose_feature_names_out=False` for clean names |
| Fitting | Preprocessor fitted **on `X_train` only** (n = 4256); train and test transformed with the same fitted object |
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
| Train / test rows | 4256 / 1064 (80/20, stratified) |
| Output features | 13 (11 numeric passthrough + `wine_type_red` + `wine_type_white`) |
| Target distribution (full) | 0: 1988 (37.4%) · 1: 3332 (62.6%) |
| Target distribution (train) | 0: 1590 · 1: 2666 (62.6% good) |
| Target distribution (test) | 0: 398 · 1: 666 (62.6% good) |

### Artifacts written

| File | Size | SHA-256 |
|------|------|---------|
| `data/processed/train.csv` | 286,608 B | `913ce1242f1fdaa358b3e0486b2932abd0f6fd80428137de3c87dab2c69e9002` |
| `data/processed/test.csv` | 71,864 B | `e0edef5b0b074c28ed59a15975691f320d3780ed20efeff762a0ebb77e20b15e` |
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

- train rows == 4256; test rows == 1064;
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
- IQR outliers are deliberately **kept** (Stage 1 decision — wholesale
  outlier removal would discard ~20.6% of rows and skew the red/white mix).
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
