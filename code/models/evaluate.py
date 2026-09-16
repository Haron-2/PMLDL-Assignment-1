"""Stage 3 - Standalone evaluation of the saved single model artifact.

Independent verification path for Stage 3: reloads `models/model.joblib`,
rebuilds the deterministic test split via `preprocess.py` (no reliance on
data/processed CSVs for predictions), recomputes all metrics and cross-checks
them against `models/metrics.json` written by train.py. Also renders the
confusion matrix figure for the report.

Usage (from anywhere):
    .venv/Scripts/python.exe code/models/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
import preprocess as pp  # noqa: E402

PROJECT_ROOT = pp.PROJECT_ROOT
MODEL_PATH = pp.MODELS_DIR / "model.joblib"
METRICS_PATH = pp.MODELS_DIR / "metrics.json"
FIG_PATH = PROJECT_ROOT / "notebooks" / "figures" / "fig6_confusion_matrix.png"
TOL = 1e-9


def main() -> None:
    pp.header("STAGE 3 | STANDALONE EVALUATION")
    for p in (MODEL_PATH, METRICS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"missing artifact: {p} (run train.py first)")
    print(f"[load] {MODEL_PATH.relative_to(PROJECT_ROOT)}  ({MODEL_PATH.stat().st_size:,} bytes)")
    pipeline = joblib.load(MODEL_PATH)
    saved = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    # deterministic test split, identical to training
    combined = pp.load_raw()
    cleaned = pp.drop_exact_duplicates(combined)
    with_target = pp.add_target(cleaned)
    _, X_test, _, y_test = pp.make_split(with_target)

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

    pp.header("CLASSIFICATION REPORT (test set, re-computed)")
    print(classification_report(y_test, y_pred, target_names=["Poor (0)", "Good (1)"], digits=4))
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"  confusion : TN={cm[0, 0]}  FP={cm[0, 1]}  FN={cm[1, 0]}  TP={cm[1, 1]}")

    pp.header("REPRODUCIBILITY CHECK vs metrics.json")
    all_ok = True
    for k, v in metrics.items():
        match = abs(v - saved["metrics"][k]) <= TOL
        all_ok &= match
        print(f"  [{'PASS' if match else 'FAIL'}] {k:10s}: {v:.6f} == {saved['metrics'][k]:.6f} (tol {TOL})")
    cm_saved = saved["confusion_matrix"]
    cm_match = [int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])] == \
               [cm_saved["tn"], cm_saved["fp"], cm_saved["fn"], cm_saved["tp"]]
    all_ok &= cm_match
    print(f"  [{'PASS' if cm_match else 'FAIL'}] confusion matrix matches metrics.json")
    if not all_ok:
        raise AssertionError("re-evaluated metrics differ from train.py metrics.json")
    print("  -> PASS: saved artifact reproduces the logged metrics exactly")

    pp.header("CONFUSION MATRIX FIGURE")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["Poor (0)", "Good (1)"])
    ax.set_yticks([0, 1], labels=["Poor (0)", "Good (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"RandomForest confusion matrix (test set)\n"
                 f"accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"[save] {FIG_PATH.relative_to(PROJECT_ROOT)}")
    print("\nSTAGE 3 EVALUATION: DONE")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()