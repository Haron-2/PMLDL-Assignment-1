"""Stage 3 - Model Engineering: train RandomForest + MLflow tracking.

Builds ONE sklearn Pipeline = ColumnTransformer (Stage 2) +
RandomForestClassifier and fits it on the raw train split. This single
artifact (`models/model.joblib`) goes from raw features to prediction,
guaranteeing training/serving consistency for Stage 4 (FastAPI).

Steps:
    1. Rebuild the deterministic Stage 2 split by reusing `preprocess.py`
       functions (single source of truth - no duplicated cleaning logic).
    2. Fit Pipeline(preprocessor, RandomForestClassifier) on X_train (raw).
    3. Consistency check: the pipeline's fitted preprocessor must transform
       X_train into EXACTLY the features stored in data/processed/train.csv
       by Stage 2.
    4. Evaluate on X_test (accuracy, precision, recall, F1, ROC-AUC,
       confusion matrix).
    5. Save models/model.joblib + models/metrics.json.
    6. Log params, metrics, model and artifacts to MLflow (local sqlite
       tracking backend at <project>/mlflow.db).

Usage (from anywhere):
    .venv/Scripts/python.exe code/models/train.py
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

# reuse Stage 2 logic (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
import preprocess as pp  # noqa: E402

PROJECT_ROOT = pp.PROJECT_ROOT
MODELS_DIR = pp.MODELS_DIR
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
EXPERIMENT_NAME = "wine-quality-prediction"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


def log_model_compat(pipeline: Pipeline, name: str = "model") -> None:
    """mlflow>=3 renamed artifact_path -> name; support both. MLflow 3 saves
    sklearn models via skops, which requires trusted types for tree internals."""
    sig = inspect.signature(mlflow.sklearn.log_model).parameters
    pip_reqs = [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"joblib=={joblib.__version__}",
    ]
    kwargs: dict = {"pip_requirements": pip_reqs}
    if "skops_trusted_types" in sig:
        kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
    if "name" in sig:
        mlflow.sklearn.log_model(pipeline, name=name, **kwargs)
    else:
        mlflow.sklearn.log_model(pipeline, artifact_path=name, **kwargs)


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    """Compute test metrics + confusion matrix from the single artifact."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def consistency_check(pipeline: Pipeline, X_train: pd.DataFrame) -> None:
    """The pipeline's fitted preprocessor must reproduce Stage 2 train.csv."""
    pp.header("CONSISTENCY CHECK (train/serving)")
    train_csv = pd.read_csv(pp.PROCESSED_DIR / "train.csv")
    expected = train_csv.drop(columns=[pp.TARGET])
    pre = pipeline.named_steps["preprocessor"]
    actual = pd.DataFrame(pre.transform(X_train), columns=list(pre.get_feature_names_out()))
    same_shape = actual.shape == expected.shape
    same_cols = list(actual.columns) == list(expected.columns)
    exact = np.array_equal(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float))
    close = exact or np.allclose(actual.to_numpy(dtype=float),
                                 expected.to_numpy(dtype=float), atol=1e-12)
    print(f"  shape match           : {same_shape}  {actual.shape} vs {expected.shape}")
    print(f"  column order match    : {same_cols}")
    print(f"  values match (bitwise): {exact}  (allclose 1e-12: {close})")
    if not (same_shape and same_cols and close):
        raise AssertionError("preprocessing drift between Stage 2 artifacts and the training pipeline")
    print("  -> PASS: single-artifact preprocessing == Stage 2 preprocessing")


def main() -> dict:
    pp.header("STAGE 3 | MODEL ENGINEERING - TRAINING")
    print(f"project root : {PROJECT_ROOT}")
    print(f"mlflow       : {mlflow.__version__}  (sqlite: {MLFLOW_DB_PATH.name})")

    # deterministic Stage 2 data flow, reused verbatim
    combined = pp.load_raw()
    pp.validate_dataframe(combined, "raw-combined")
    cleaned = pp.drop_exact_duplicates(combined)
    pp.validate_dataframe(cleaned, "after-dedup")
    with_target = pp.add_target(cleaned)
    X_train, X_test, y_train, y_test = pp.make_split(with_target)

    pp.header("FIT PIPELINE (preprocessor + RandomForest)")
    pipeline = Pipeline([
        ("preprocessor", pp.build_preprocessor()),
        ("classifier", RandomForestClassifier(**RF_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)
    print(f"[fit] Pipeline fitted on raw X_train (n={len(X_train)}): "
          f"ColumnTransformer -> RandomForestClassifier(n_estimators={RF_PARAMS['n_estimators']}, "
          f"random_state={RF_PARAMS['random_state']})")

    consistency_check(pipeline, X_train)
    metrics, cm = evaluate(pipeline, X_test, y_test)

    pp.header("TEST METRICS (single artifact, raw X_test)")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"  confusion : TN={cm[0, 0]}  FP={cm[0, 1]}  FN={cm[1, 0]}  TP={cm[1, 1]}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[save] {MODEL_PATH.relative_to(PROJECT_ROOT)}  ({MODEL_PATH.stat().st_size:,} bytes)")

    pp.header("MLFLOW TRACKING")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="rf-baseline-300trees") as run:
        mlflow.log_params({
            **RF_PARAMS,
            "test_size": pp.TEST_SIZE,
            "split_random_state": pp.RANDOM_STATE,
            "quality_threshold": pp.QUALITY_THRESHOLD,
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "stage": "3-model-engineering",
            "model_type": "RandomForestClassifier",
            "dataset": "UCI wine quality red+white (deduplicated)",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "raw_feature_count": len(pp.FEATURES),
            "transformed_feature_count": 13,
            "sklearn_version": sklearn.__version__,
        })
        log_model_compat(pipeline, "model")
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}")
        print(f"[mlflow] params logged: {len(RF_PARAMS) + 3}, metrics logged: {len(metrics)}, model logged: yes")

    payload = {
        "model": "RandomForestClassifier",
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": RF_PARAMS,
        "split": {"test_size": pp.TEST_SIZE, "random_state": pp.RANDOM_STATE, "stratify": True},
        "dataset": {
            "combined_rows": len(combined),
            "duplicates_removed": len(combined) - len(cleaned),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "raw_feature_count": len(pp.FEATURES),
            "transformed_feature_count": 13,
        },
        "metrics": metrics,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "model_path": "models/model.joblib",
        "mlflow": {"experiment": EXPERIMENT_NAME, "run_id": run_id,
                   "tracking_uri": TRACKING_URI},
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from mlflow.tracking import MlflowClient
    MlflowClient().log_artifact(run_id, str(METRICS_PATH))
    print(f"[save] {METRICS_PATH.relative_to(PROJECT_ROOT)}")

    pp.header("TRAINING SUMMARY")
    print(f"  artifact      : {MODEL_PATH.relative_to(PROJECT_ROOT)} "
          f"(Pipeline: preprocessor + RandomForestClassifier)")
    print(f"  test accuracy : {metrics['accuracy']:.4f}  f1: {metrics['f1']:.4f}  "
          f"roc_auc: {metrics['roc_auc']:.4f}")
    print(f"  mlflow run    : {EXPERIMENT_NAME} / {run_id}")
    print("\nSTAGE 3 TRAINING: DONE")
    return payload


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

