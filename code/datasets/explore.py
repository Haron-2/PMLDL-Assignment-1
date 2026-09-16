"""Stage 1 - Data Engineering / EDA exploration for the Wine Quality project.

Reproducible script behind notebooks/data_exploration.ipynb (Stage 1 EDA).
It is read-only with respect to data: it never modifies data/raw and never
writes cleaned datasets - building the cleaned dataset is the job of Stage 2
(code/datasets/preprocess.py).

Phases covered:
  1  dataset integrity (shape, dtypes, missing, ranges, sanity checks)
  2  duplicate analysis (+ label-contradiction check)
  3-5 IQR outlier analysis before/after duplicate removal,
      row-level outlier_count distribution, feature-level table
  5b  supplement: pooled vs per-wine-type IQR bounds
  6   figures (saved to notebooks/figures/)
  7   target analysis (quality, binary target, wine_type interaction)
  8   correlations & skewness
  9   impact of candidate outlier strategies

Usage (from the project root):
    .venv/Scripts/python.exe code/datasets/explore.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved to disk, not shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

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
TARGET = "quality"
ALL_COLUMNS = NUMERIC_FEATURES + [CATEGORICAL, TARGET]

# matplotlib >= 3.9 renamed boxplot `labels` -> `tick_labels`
_MPL = tuple(int(p) for p in matplotlib.__version__.split(".")[:2])
_BOX_KW = {"tick_labels": ["red", "white"]} if _MPL >= (3, 9) else {"labels": ["red", "white"]}


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_combined() -> pd.DataFrame:
    """Load both raw csv files and combine them with a wine_type column."""
    red = pd.read_csv(RAW_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(RAW_DIR / "winequality-white.csv", sep=";")
    red[CATEGORICAL] = "red"
    white[CATEGORICAL] = "white"
    return pd.concat([red, white], ignore_index=True)[ALL_COLUMNS]


def phase1_integrity(combined: pd.DataFrame) -> None:
    """Phase 1: shape / dtypes / missing / uniques / ranges / sanity checks."""
    header("PHASE 1 | DATASET INTEGRITY (raw data, before any cleaning)")
    n_red = int((combined[CATEGORICAL] == "red").sum())
    n_white = int((combined[CATEGORICAL] == "white").sum())
    print(f"combined shape          : {combined.shape}")
    print(f"  red rows              : {n_red}")
    print(f"  white rows            : {n_white}")
    print(f"columns                 : {list(combined.columns)}")
    features_float = all(combined[c].dtype == np.float64 for c in NUMERIC_FEATURES)
    print(f"dtypes                  : features all float64 = {features_float}; "
          f"quality = {combined[TARGET].dtype}; wine_type = {combined[CATEGORICAL].dtype}")
    print(f"quality unique values   : {sorted(int(q) for q in combined[TARGET].unique())}")
    print(f"wine_type unique values : {sorted(combined[CATEGORICAL].unique())}")

    missing = combined.isna().sum()
    print("\nMissing values:")
    print("  none in any column" if int(missing.sum()) == 0 else missing[missing > 0].to_string())

    print("\nPer-feature min / max (combined):")
    ranges = combined[NUMERIC_FEATURES].agg(["min", "max"]).T
    print(ranges.to_string(float_format=lambda v: f"{v:10.4f}"))

    print("\nSanity checks (physically impossible / inconsistent values):")
    print(f"  negative feature values       : {int((combined[NUMERIC_FEATURES] < 0).sum().sum())}")
    print(f"  density outside [0.98, 1.05]  : {int((~combined['density'].between(0.98, 1.05)).sum())}")
    print(f"  pH outside [2.5, 4.5]         : {int((~combined['pH'].between(2.5, 4.5)).sum())}")
    print(f"  alcohol outside [8, 16] %vol  : {int((~combined['alcohol'].between(8, 16)).sum())}")
    free_gt_total = combined["free sulfur dioxide"] > combined["total sulfur dioxide"]
    print(f"  free SO2 > total SO2          : {int(free_gt_total.sum())}")
def phase2_duplicates(combined: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: exact duplicate analysis + label-contradiction check."""
    header("PHASE 2 | DUPLICATE ANALYSIS")
    n = len(combined)
    dup_mask = combined.duplicated()
    cleaned = combined.drop_duplicates().reset_index(drop=True)

    print(f"rows total (combined)                : {n}")
    print(f"exact duplicate rows                 : {int(dup_mask.sum())} ({dup_mask.mean():.2%})")
    print(f"rows after drop_duplicates()         : {len(cleaned)}")
    for label in ("red", "white"):
        sub = combined[combined[CATEGORICAL] == label]
        d = int(sub.duplicated().sum())
        print(f"  {label:5s}: {d:4d} duplicates ({d / len(sub):5.2%}) -> {len(sub) - d} unique rows")

    # full feature+quality vector present in both wine types?
    key = NUMERIC_FEATURES + [TARGET]
    both_types = combined.groupby(key, dropna=False)[CATEGORICAL].transform("nunique") > 1
    print(f"rows whose feature+quality vector exists in BOTH wine types : {int(both_types.sum())}")

    # same feature vector but different quality -> label contradictions
    feat_key = NUMERIC_FEATURES + [CATEGORICAL]
    same_feat = combined.duplicated(subset=feat_key, keep=False)
    n_groups = int(combined[same_feat].groupby(feat_key, dropna=False).ngroups)
    contra_rows = int((combined[same_feat]
                       .groupby(feat_key, dropna=False)[TARGET].transform("nunique") > 1).sum())
    contra_groups = int((combined[same_feat]
                         .groupby(feat_key, dropna=False)[TARGET].nunique() > 1).sum())
    print(f"\nRows sharing an identical feature vector (quality excluded): {int(same_feat.sum())}")
    print(f"  such duplicate groups                                : {n_groups}")
    print(f"  groups with DIFFERENT quality for identical features : {contra_groups}")
    print(f"  rows in contradictory groups                         : "
          f"{contra_rows} ({contra_rows / n:.2%} of all rows)")
    return cleaned
def _flags_and_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """IQR outlier flags + (Q1, Q3, IQR, lower, upper) for the 11 features."""
    flags, bounds = {}, {}
    for col in NUMERIC_FEATURES:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bounds[col] = (q1, q3, iqr, lower, upper)
        flags[col] = (df[col] < lower) | (df[col] > upper)
    return pd.DataFrame(flags, index=df.index), bounds


def iqr_analysis(df: pd.DataFrame, label: str) -> pd.Series:
    """Phases 3-5: feature-level IQR table + row-level outlier_count."""
    header(f"PHASES 3-5 | IQR OUTLIER ANALYSIS - {label} (n={len(df)})")
    flags, bounds = _flags_and_bounds(df)
    rows = []
    for col in NUMERIC_FEATURES:
        q1, q3, iqr, lower, upper = bounds[col]
        rows.append({
            "feature": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower": lower, "upper": upper,
            "min": df[col].min(), "max": df[col].max(),
            "n_out": int(flags[col].sum()), "out_%": 100 * flags[col].mean(),
        })
    table = pd.DataFrame(rows).set_index("feature").sort_values("out_%", ascending=False)
    print(table.to_string(float_format=lambda v: f"{v:9.3f}"))

    outlier_count = flags.sum(axis=1)
    print("\nRow-level outlier_count distribution:")
    for k, v in outlier_count.value_counts().sort_index().items():
        print(f"  count = {int(k)}: {int(v):5d} rows  ({v / len(df):6.2%})")
    print("\nCumulative view:")
    for name, cond in [
        ("count == 0", outlier_count == 0), ("count == 1", outlier_count == 1),
        ("count >= 2", outlier_count >= 2), ("count >= 3", outlier_count >= 3),
        ("count >= 4", outlier_count >= 4),
    ]:
        v = int(cond.sum())
        print(f"  {name:12s}: {v:5d} rows  ({v / len(df):6.2%})")

    single = flags[outlier_count == 1].sum().sort_values(ascending=False)
    print("\nSingle-feature outlier rows (count == 1) by feature:")
    print(single[single > 0].to_string())

    y = (df[TARGET] > 5).astype(int)
    out_mask = outlier_count > 0
    print("\nTarget mix (share Good = quality > 5):")
    print(f"  outlier rows : {y[out_mask].mean():.2%} Good  (n={int(out_mask.sum())})")
    print(f"  clean rows   : {y[~out_mask].mean():.2%} Good  (n={int((~out_mask).sum())})")
    print("\nwine_type mix of outlier rows (%):")
    print((df.loc[out_mask, CATEGORICAL].value_counts(normalize=True) * 100).round(2).to_string())
    return outlier_count
def phase5b_per_type(df: pd.DataFrame) -> None:
    """Supplement: pooled IQR bounds vs bounds computed per wine type."""
    header("PHASE 5b | SUPPLEMENT - pooled vs per-wine-type IQR bounds")
    pooled_flags, pooled_bounds = _flags_and_bounds(df)
    pooled_count = pooled_flags.sum(axis=1)
    for label in ("red", "white"):
        sub = df[df[CATEGORICAL] == label]
        own_flags, own_bounds = _flags_and_bounds(sub)
        pooled_out = int((pooled_count.loc[sub.index] > 0).sum())
        own_out = int((own_flags.sum(axis=1) > 0).sum())
        print(f"\n{label}: n={len(sub)}")
        print(f"  rows flagged by POOLED bounds   : {pooled_out} ({pooled_out / len(sub):.2%})")
        print(f"  rows flagged by OWN-TYPE bounds : {own_out} ({own_out / len(sub):.2%})")
        diff = pooled_flags.loc[sub.index].sum() - own_flags.sum()
        diff = diff[diff != 0].sort_values()
        if len(diff):
            print("  per-feature outlier count difference (pooled - own):")
            print("    " + diff.to_string().replace("\n", "\n    "))
            worst = diff.abs().sort_values(ascending=False).index[0]
            pb, ob = pooled_bounds[worst], own_bounds[worst]
            print(f"  most affected feature '{worst}': pooled bounds "
                  f"[{pb[3]:.3f}, {pb[4]:.3f}] vs own-type bounds [{ob[3]:.3f}, {ob[4]:.3f}]")
def phase7_target(df: pd.DataFrame) -> None:
    """Phase 7: quality + binary target + wine_type interaction."""
    header("PHASE 7 | TARGET ANALYSIS (cleaned data)")
    print("quality distribution:")
    for q, v in df[TARGET].value_counts().sort_index().items():
        print(f"  quality = {q}: {int(v):5d}  ({v / len(df):6.2%})")

    y = (df[TARGET] > 5).astype(int)
    print("\nBinary target  (quality <= 5 -> 0 'Poor' | quality > 5 -> 1 'Good'):")
    for cls, name in ((0, "Poor"), (1, "Good")):
        v = int((y == cls).sum())
        print(f"  {cls} {name:4s}: {v:5d}  ({v / len(y):6.2%})")
    print(f"  imbalance ratio Good : Poor = {(y == 1).sum() / (y == 0).sum():.2f} : 1")

    print("\nquality x wine_type (row % within wine type):")
    print((pd.crosstab(df[CATEGORICAL], df[TARGET], normalize="index") * 100).round(2).to_string())
    print("\nbinary target x wine_type (row % within wine type):")
    print((pd.crosstab(df[CATEGORICAL], y, normalize="index") * 100).round(2).to_string())
    print("\nbinary target x wine_type (raw counts):")
    print(pd.crosstab(df[CATEGORICAL], y).to_string())


def phase8_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 8: pairwise correlations + skewness (transformation hints)."""
    header("PHASE 8 | CORRELATIONS & SKEWNESS (cleaned data)")
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    tri = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
           .stack().rename("r").reset_index())
    tri.columns = ["feature_a", "feature_b", "r"]
    tri["abs_r"] = tri["r"].abs()
    top = tri[tri["abs_r"] >= 0.40].sort_values("abs_r", ascending=False)
    print("Feature pairs with |Pearson r| >= 0.40:")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n|Correlation| of features with quality:")
    print(corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).round(3).to_string())
    print("\nSkewness of features:")
    print(df[NUMERIC_FEATURES].skew().sort_values(ascending=False).round(2).to_string())
    return corr


def phase9_strategy_impact(df: pd.DataFrame) -> None:
    """Phase 9: dataset impact of candidate outlier strategies."""
    header("PHASE 9 | IMPACT OF CANDIDATE OUTLIER STRATEGIES (cleaned data)")
    flags, _ = _flags_and_bounds(df)
    oc = flags.sum(axis=1)
    y = (df[TARGET] > 5).astype(int)
    n = len(df)
    print(f"baseline: {n} rows | Good = {y.mean():.2%} | red = {(df[CATEGORICAL] == 'red').mean():.2%}")
    strategies = [
        ("keep all (no removal)", pd.Series(True, index=df.index)),
        ("drop rows with count >= 4", oc < 4),
        ("drop rows with count >= 3", oc < 3),
        ("drop rows with count >= 2", oc < 2),
        ("drop all IQR outliers", oc < 1),
    ]
    print(f"\n{'strategy':26s} {'rows':>6s} {'kept':>7s} {'dropped':>8s} {'Good%':>7s} {'red%':>6s}")
    for name, keep in strategies:
        kept = df[keep]
        print(f"{name:26s} {len(kept):6d} {len(kept) / n:7.2%} {n - len(kept):8d} "
              f"{y[keep].mean():7.2%} {(kept[CATEGORICAL] == 'red').mean():6.2%}")
def make_figures(cleaned: pd.DataFrame, oc_before: pd.Series, oc_after: pd.Series,
                 corr: pd.DataFrame) -> None:
    """Phase 6: a small set of plots, each answering one concrete question."""
    header("PHASE 6 | FIGURES (saved to notebooks/figures/)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # fig 1 - target distributions (cleaned data)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    qd = cleaned[TARGET].value_counts().sort_index()
    axes[0].bar(qd.index.astype(str), qd.values, color="#4c72b0")
    axes[0].set_title("quality distribution (after dedup)")
    axes[0].set_xlabel("quality")
    axes[0].set_ylabel("rows")
    x = np.arange(len(qd.index))
    red_q = cleaned[cleaned[CATEGORICAL] == "red"][TARGET].value_counts().sort_index()
    white_q = cleaned[cleaned[CATEGORICAL] == "white"][TARGET].value_counts().sort_index()
    axes[1].bar(x - 0.2, red_q.reindex(qd.index, fill_value=0).values, 0.4,
                label="red", color="#c44e52")
    axes[1].bar(x + 0.2, white_q.reindex(qd.index, fill_value=0).values, 0.4,
                label="white", color="#55a868")
    axes[1].set_xticks(x, [str(q) for q in qd.index])
    axes[1].set_title("quality by wine_type (after dedup)")
    axes[1].legend()
    y = (cleaned[TARGET] > 5).astype(int).value_counts().sort_index()
    axes[2].bar(["0 Poor (<=5)", "1 Good (>5)"], y.values, color=["#c44e52", "#55a868"])
    for i, v in enumerate(y.values):
        axes[2].text(i, v, f"{v}\n({v / len(cleaned):.1%})", ha="center", va="bottom")
    axes[2].set_ylim(0, y.max() * 1.18)
    axes[2].set_title("binary target (after dedup)")
    axes[2].set_ylabel("rows")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_target_distribution.png", dpi=150)
    plt.close(fig)

    # fig 2 - feature boxplots, red vs white (cleaned data)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        data = [cleaned.loc[cleaned[CATEGORICAL] == "red", col],
                cleaned.loc[cleaned[CATEGORICAL] == "white", col]]
        ax.boxplot(data, **_BOX_KW)
        ax.set_title(col, fontsize=10)
    axes.flat[-1].axis("off")
    fig.suptitle("Numerical features: red vs white (after dedup)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_boxplots_red_vs_white.png", dpi=150)
    plt.close(fig)

    # fig 3 - outlier_count per row, before vs after dedup
    levels = sorted(set(oc_before.unique()) | set(oc_after.unique()))
    x = np.arange(len(levels))
    width = 0.38
    before = [(oc_before == k).mean() * 100 for k in levels]
    after = [(oc_after == k).mean() * 100 for k in levels]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - width / 2, before, width, label=f"before dedup (n={len(oc_before)})")
    ax.bar(x + width / 2, after, width, label=f"after dedup (n={len(oc_after)})")
    for xi, v in zip(x - width / 2, before):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, after):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(k) for k in levels])
    ax.set_xlabel("outlier_count per row (11 IQR features)")
    ax.set_ylabel("% of rows")
    ax.set_title("IQR outlier_count per row: before vs after duplicate removal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_outlier_count_before_after.png", dpi=150)
    plt.close(fig)

    # fig 4 - top-4 features by outlier rate with IQR bounds (cleaned data)
    flags, bounds = _flags_and_bounds(cleaned)
    top4 = flags.sum().sort_values(ascending=False).index[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, col in zip(axes.flat, top4):
        lo, up = bounds[col][3], bounds[col][4]
        ax.hist(cleaned[col], bins=60, color="#4c72b0")
        ax.axvline(lo, color="#c44e52", ls="--", lw=1.2, label=f"lower = {lo:.2f}")
        ax.axvline(up, color="#c44e52", ls=":", lw=1.2, label=f"upper = {up:.2f}")
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Top-4 features by IQR outlier rate (after dedup)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_top_outlier_features.png", dpi=150)
    plt.close(fig)

    # fig 5 - correlation heatmap (cleaned data)
    cols = NUMERIC_FEATURES + [TARGET]
    c = corr.loc[cols, cols].values
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c[i, j]) > 0.55 else "black")
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Pearson correlation (after dedup)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    for name in sorted(p.name for p in FIGURES_DIR.glob("fig*.png")):
        print("  saved", name)
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    combined = load_combined()
    phase1_integrity(combined)
    cleaned = phase2_duplicates(combined)
    oc_before = iqr_analysis(combined, "BEFORE duplicate removal")
    oc_after = iqr_analysis(cleaned, "AFTER duplicate removal")
    phase5b_per_type(cleaned)
    phase7_target(cleaned)
    corr = phase8_correlations(cleaned)
    phase9_strategy_impact(cleaned)
    make_figures(cleaned, oc_before, oc_after, corr)
    print("\nDone. Figures are saved in:", FIGURES_DIR)


if __name__ == "__main__":
    main()
