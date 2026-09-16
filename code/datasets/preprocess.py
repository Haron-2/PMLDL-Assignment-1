"""Stage 2 - Data Engineering / Preprocessing for the Wine Quality project.

Implements the cleaning + preprocessing pipeline fixed by the Stage 1 EDA
(notebooks/STAGE1_EDA_REPORT.md):

    load raw data -> add wine_type -> concat -> validate -> drop exact
    duplicates -> create target -> train/test split -> fit ColumnTransformer
    on TRAIN only -> transform -> save artifacts -> validate outputs

Stage 1 decisions encoded here (do NOT change without re-running the EDA):
    - missing values: none exist (fail-fast validation)
    - outliers: kept (IQR diagnostics only, no removal)
    - duplicates: exact duplicates removed, BEFORE the split (leakage rule)
    - target: quality > 5 -> 1 (Good) else 0 (Poor)
    - wine_type: kept, one-hot encoded
    - numerical features: passthrough (RandomForest needs no scaling)

Artifacts written:
    data/processed/train.csv     transformed features + target (no `quality`)
    data/processed/test.csv      transformed features + target (no `quality`)
    models/preprocessor.joblib   ColumnTransformer fitted on X_train only

The module is CWD-independent (all paths derive from __file__) and exposes
importable functions so Stage 3 (train/evaluate) and the Airflow DAG reuse
the exact same cleaning/split/preprocessing logic (single source of truth).

Usage (from anywhere):
    .venv/Scripts/python.exe code/datasets/preprocess.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------------ paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ------------------------------------------------------------------ schema
NUMERIC_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
CATEGORICAL = "wine_type"
QUALITY = "quality"
FEATURES = NUMERIC_FEATURES + [CATEGORICAL]     # raw model input features
RAW_COLUMNS = FEATURES + [QUALITY]              # combined raw frame schema
TARGET = "target"

# ------------------------------------------------------------------ config
TEST_SIZE = 0.2
RANDOM_STATE = 42
QUALITY_THRESHOLD = 5
WINE_TYPES = {"red", "white"}


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================ pipeline steps
def load_raw() -> pd.DataFrame:
    """Load red + white raw CSVs, add wine_type, concat (6497, 13)."""
    red = pd.read_csv(RAW_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(RAW_DIR / "winequality-white.csv", sep=";")
    red[CATEGORICAL] = "red"
    white[CATEGORICAL] = "white"
    combined = pd.concat([red, white], ignore_index=True)[RAW_COLUMNS]
    return combined


def validate_dataframe(df: pd.DataFrame, stage: str) -> None:
    """Fail-fast deterministic checks. Raises ValueError on any problem."""
    problems: list[str] = []

    if list(df.columns) != RAW_COLUMNS:
        problems.append(f"unexpected schema: {list(df.columns)}")
    if df.empty:
        problems.append("empty dataframe")
    if df.isna().sum().sum() != 0:
        counts = df.isna().sum()
        problems.append(f"missing values found: {counts[counts > 0].to_dict()}")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in NUMERIC_FEATURES):
        bad = [c for c in NUMERIC_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
        problems.append(f"non-numeric feature columns: {bad}")
    if not pd.api.types.is_integer_dtype(df[QUALITY]):
        problems.append(f"quality is not integer dtype: {df[QUALITY].dtype}")
    if not set(map(str, df[CATEGORICAL].unique())) <= WINE_TYPES:
        problems.append(f"unexpected wine_type values: {sorted(map(str, df[CATEGORICAL].unique()))}")
    if not df[QUALITY].between(0, 10).all():
        problems.append("quality outside [0, 10]")
    bad_so2 = df[df["free sulfur dioxide"] > df["total sulfur dioxide"]]
    if not bad_so2.empty:
        problems.append(f"{len(bad_so2)} rows with free SO2 > total SO2")
    if not (df[NUMERIC_FEATURES] >= 0).all().all():
        neg = [c for c in NUMERIC_FEATURES if (df[c] < 0).any()]
        problems.append(f"negative values in feature columns: {neg}")

    if problems:
        raise ValueError(f"[{stage}] data validation failed:\n  - " + "\n  - ".join(problems))
    print(f"[validate:{stage}] OK  shape={df.shape}  no NaN  wine_type valid  "
          f"free SO2 <= total SO2  quality in [0,10]")


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates across ALL 13 columns (Stage 1 decision)."""
    before = len(df)
    cleaned = df.drop_duplicates(ignore_index=True)
    removed = before - len(cleaned)
    print(f"[clean] exact duplicates removed: {removed} ({removed / before:.2%})  "
          f"{before} -> {len(cleaned)} rows")
    return cleaned


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """target = 1 (Good) if quality > 5 else 0 (Poor). quality NOT a feature."""
    out = df.copy()
    out[TARGET] = (out[QUALITY] > QUALITY_THRESHOLD).astype(int)
    print(f"[target] distribution: {out[TARGET].value_counts().sort_index().to_dict()}")
    return out


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """X = raw features (11 numeric + wine_type); y = target. Deterministic."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[split] train={len(X_train)}  test={len(X_test)}  "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y)")
    print(f"[split] target train: {y_train.value_counts().sort_index().to_dict()}")
    print(f"[split] target test : {y_test.value_counts().sort_index().to_dict()}")
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Numerical -> passthrough; wine_type -> OneHot(handle_unknown='ignore')."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [CATEGORICAL]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_frame(preprocessor: ColumnTransformer,
                      X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Apply fitted preprocessor; return frame with clean feature names + target."""
    Xt = preprocessor.transform(X)
    feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
    out = pd.DataFrame(Xt, columns=feature_names, index=X.index)
    out[TARGET] = y.values
    return out


# ============================================================ artifacts
def sha256_of(path: Path) -> str:
    """SHA256 with a small retry: right after writing, a transient AV/FS lock
    can make the first read return empty bytes on Windows even for non-empty
    files, which would silently report sha256('')."""
    import time
    size = path.stat().st_size
    empty_hash = hashlib.sha256(b"").hexdigest()
    for _ in range(5):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        if size == 0 or h.hexdigest() != empty_hash:
            return h.hexdigest()
        time.sleep(0.2)
    return h.hexdigest()


def validate_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     feature_names: list[str]) -> None:
    """Post-save checks: sizes, schema, NaN, target, artifact presence."""
    header("OUTPUT VALIDATION")
    checks: list[tuple[str, bool, str]] = [
        ("train rows == 4256", len(train_df) == 4256, f"actual={len(train_df)}"),
        ("test rows == 1064", len(test_df) == 1064, f"actual={len(test_df)}"),
        ("no NaN in train", not train_df.isna().any().any(), ""),
        ("no NaN in test", not test_df.isna().any().any(), ""),
        ("'quality' NOT in outputs",
         QUALITY not in train_df.columns and QUALITY not in test_df.columns, ""),
        ("'target' in outputs (train/test)",
         TARGET in train_df.columns and TARGET in test_df.columns, ""),
        ("target values in {0,1}",
         set(train_df[TARGET].unique()) <= {0, 1} and set(test_df[TARGET].unique()) <= {0, 1}, ""),
        ("'wine_type_*' columns present",
         any(c.startswith("wine_type_") for c in train_df.columns),
         f"{[c for c in train_df.columns if c.startswith('wine_type_')]}"),
        ("feature count == 13", len(feature_names) == 13, f"actual={len(feature_names)}"),
        ("column order == features + [target]",
         list(train_df.columns) == feature_names + [TARGET], ""),
        ("preprocessor.joblib exists", (MODELS_DIR / "preprocessor.joblib").exists(), ""),
        ("train.csv exists", (PROCESSED_DIR / "train.csv").exists(), ""),
        ("test.csv exists", (PROCESSED_DIR / "test.csv").exists(), ""),
    ]
    failed = sum(not ok for _, ok, _ in checks)
    for name, ok, extra in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))
    if failed:
        raise AssertionError(f"output validation failed: {failed} check(s)")
    print(f"  -> all {len(checks)} checks passed")


# ============================================================ main
def main() -> dict:
    header("STAGE 2 | DATA ENGINEERING - PREPROCESSING")
    print(f"project root : {PROJECT_ROOT}")
    print(f"raw data dir : {RAW_DIR}")

    combined = load_raw()
    print(f"[load] combined shape: {combined.shape}")
    validate_dataframe(combined, "raw-combined")

    cleaned = drop_exact_duplicates(combined)
    validate_dataframe(cleaned, "after-dedup")

    with_target = add_target(cleaned)
    X_train, X_test, y_train, y_test = make_split(with_target)

    header("FIT PREPROCESSOR (on TRAIN only - leakage prevention)")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
    print(f"[fit] fitted on X_train only (n={len(X_train)})")
    print(f"[fit] output feature names ({len(feature_names)}): {feature_names}")

    train_df = transformed_frame(preprocessor, X_train, y_train)
    test_df = transformed_frame(preprocessor, X_test, y_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    prep_path = MODELS_DIR / "preprocessor.joblib"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(preprocessor, prep_path)

    header("SAVED ARTIFACTS (sha256 for reproducibility)")
    for p in (train_path, test_path, prep_path):
        print(f"  {p.relative_to(PROJECT_ROOT)}  size={p.stat().st_size:>8}  sha256={sha256_of(p)}")

    validate_outputs(pd.read_csv(train_path), pd.read_csv(test_path), feature_names)

    header("PROCESSING SUMMARY")
    print(f"  raw combined rows          : {len(combined)}")
    print(f"  exact duplicates removed   : {len(combined) - len(cleaned)}")
    print(f"  cleaned rows               : {len(cleaned)}")
    print(f"  train / test rows          : {len(train_df)} / {len(test_df)}")
    print(f"  output features            : {len(feature_names)}")
    print(f"  target dist (train)        : {train_df[TARGET].value_counts().sort_index().to_dict()}")
    print(f"  target dist (test)         : {test_df[TARGET].value_counts().sort_index().to_dict()}")
    print("\nSTAGE 2 PREPROCESSING: DONE")
    return {
        "combined_rows": len(combined),
        "duplicates_removed": len(combined) - len(cleaned),
        "cleaned_rows": len(cleaned),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_count": len(feature_names),
        "train_sha256": sha256_of(train_path),
        "test_sha256": sha256_of(test_path),
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
