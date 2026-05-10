"""
Step 1: Fit a KMeans style-clustering model on per-fighter career averages.

Usage:
    python src/cluster.py data/raw/<ufc_csv>.csv

Outputs:
    models/style_kmeans.pkl   — {"model": KMeans, "labels": dict, "features": list}
    models/style_scaler.pkl   — StandardScaler
    data/fighters_with_style.csv
    data/cluster_centroids.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column suffixes present in the mdabbert dataset (prefixed with R_ or B_ per corner).
# avg_ columns are cumulative career averages up to each fight date.
# Note: avg_KD, SIG_STR_def, TD_def are absent from this dataset — ko_finish_rate
# is derived in extract_career_stats() as a KO-power proxy instead.
STYLE_FEATURES = [
    "avg_SIG_STR_pct",    # striking accuracy (scale-stable across all eras)
    "avg_TD_pct",         # takedown accuracy
    "avg_TD_landed",      # takedown volume (per-fight avg, consistent pre/post 2020)
    "avg_SUB_ATT",        # submission attempts per fight
    # avg_SIG_STR_landed intentionally excluded: pre-2020 values are raw per-fight totals
    # (~30 mean) while post-2020 values are per-minute rates (~4.6 mean) — 7x scale
    # gap causes the Volume_Striker centroid to be unreachable for all modern fighters.
]

# Extra columns needed only to derive ko_finish_rate; not used directly as features.
_AUX_COLS = ["wins", "win_by_KO/TKO"]

MIN_FIGHTS = 3   # fighters with fewer fights in their weight class are excluded
K_RANGE    = (4, 6)

_ROOT      = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.normpath(os.path.join(_ROOT, "models"))
DATA_DIR   = os.path.normpath(os.path.join(_ROOT, "data"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Normalize Winner to "R" / "B" regardless of how the dataset encodes it
    if "Winner" in df.columns:
        wmap = {"red": "R", "blue": "B", "r": "R", "b": "B"}
        normalized = df["Winner"].str.strip().str.lower().map(wmap)
        df["Winner"] = normalized.where(normalized.notna(), df["Winner"])

    print(f"Loaded {len(df):,} fights  ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    return " ".join(name.strip().split()).title()


# ---------------------------------------------------------------------------
# Reshape to long format
# ---------------------------------------------------------------------------

def reshape_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one-row-per-fight format (R_/B_ columns) into one-row-per-fighter-per-fight.
    Each fight produces two rows — one per corner.
    """
    records = []
    for corner in ("R", "B"):
        row_data: dict = {
            "fighter":      df[f"{corner}_fighter"].apply(normalize_name).values,
            "date":         df["date"].values,
            "weight_class": df["weight_class"].values if "weight_class" in df.columns else np.nan,
            "won":          (df["Winner"] == corner).astype(int).values,
        }
        for feat in STYLE_FEATURES + _AUX_COLS:
            src_col = f"{corner}_{feat}"
            row_data[feat] = (
                pd.to_numeric(df[src_col], errors="coerce").values
                if src_col in df.columns
                else np.nan
            )
        records.append(pd.DataFrame(row_data))

    long = pd.concat(records, ignore_index=True)
    long = long[long["fighter"] != ""]
    return long


# ---------------------------------------------------------------------------
# Career stats extraction
# ---------------------------------------------------------------------------

def extract_career_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fighter, take their most recent fight row — which holds their
    cumulative career-average stats up to that date — filtered to their most
    recent weight class.

    These averages are used only to fit the cluster shape. Per-fight style
    assignments during feature engineering use rolling pre-fight stats (features.py).
    """
    latest_wc = (
        long_df.dropna(subset=["weight_class"])
               .sort_values("date")
               .groupby("fighter")["weight_class"]
               .last()
               .rename("latest_wc")
    )

    df = long_df.merge(latest_wc, on="fighter", how="left")
    df = df[df["weight_class"] == df["latest_wc"]]

    fight_counts = df.groupby("fighter").size().rename("n_fights")

    career = (
        df.sort_values("date")
          .groupby("fighter", as_index=False)
          .last()
          .merge(fight_counts.reset_index(), on="fighter")
    )
    career = career[career["n_fights"] >= MIN_FIGHTS].copy()

    # Derive KO finish rate as a proxy for knockout power (replaces missing avg_KD).
    # Clamped to [0, 1]: KO/TKO wins divided by total wins (floor at 1 to avoid div/0).
    if "win_by_KO/TKO" in career.columns and "wins" in career.columns:
        ko = pd.to_numeric(career["win_by_KO/TKO"], errors="coerce").fillna(0)
        w  = pd.to_numeric(career["wins"],          errors="coerce").fillna(0).clip(lower=1)
        career["ko_finish_rate"] = (ko / w).clip(0, 1)

    all_features = STYLE_FEATURES + (["ko_finish_rate"] if "ko_finish_rate" in career.columns else [])
    available = [f for f in all_features if f in career.columns]
    for feat in available:
        career[feat] = pd.to_numeric(career[feat], errors="coerce")
        career[feat] = career[feat].fillna(career[feat].median())

    print(f"Fighters eligible for clustering: {len(career):,} "
          f"(≥{MIN_FIGHTS} fights in most recent weight class)")
    return career.set_index("fighter")[["weight_class", "n_fights"] + available]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def select_best_k(X_scaled: np.ndarray, k_range: tuple) -> int:
    scores: dict[int, float] = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        scores[k] = silhouette_score(X_scaled, km.fit_predict(X_scaled))
        print(f"  k={k}  silhouette={scores[k]:.4f}")
    best = max(scores, key=scores.get)
    print(f"  → Best k={best}  (silhouette={scores[best]:.4f})")
    return best


def fit_clustering(career: pd.DataFrame) -> tuple[KMeans, StandardScaler, list[str]]:
    """Scale features, pick best k, fit KMeans. Returns (kmeans, scaler, feature_names)."""
    # Include any derived features (e.g. ko_finish_rate) that were added to the DataFrame.
    base = STYLE_FEATURES + ["ko_finish_rate"]
    feature_names = [f for f in base if f in career.columns]
    X = career[feature_names].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nSelecting best k in range {K_RANGE}...")
    best_k = select_best_k(X_scaled, K_RANGE)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans, scaler, feature_names


def label_clusters(
    kmeans: KMeans,
    scaler: StandardScaler,
    feature_names: list[str],
) -> dict[int, str]:
    """
    Assign human-readable style labels by scoring each cluster's centroid
    against archetype definitions using z-score-normalized centroid values.
    Each archetype is assigned to at most one cluster; remaining clusters get
    "Well-Rounded" as a fallback.
    """
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_names,
    )
    # Z-score so features on different scales are comparable across clusters
    cz = (centroids - centroids.mean()) / (centroids.std() + 1e-9)

    # Positive weight = feature should be elevated; negative = should be low.
    # avg_SIG_STR_landed removed (7x scale gap pre/post 2020), so Volume_Striker
    # is redefined as high-accuracy, low-KO-rate (decision-hunter) using pct.
    archetypes: dict[str, dict[str, float]] = {
        "Pressure Striker":    {"ko_finish_rate": 2.5},
        "Volume Striker":      {"avg_SIG_STR_pct": 1.5, "ko_finish_rate": -1.5,
                                "avg_TD_pct": -0.5},
        "Wrestler/Grappler":   {"avg_TD_pct": 2.0, "avg_TD_landed": 1.5},
        "Submission Grappler": {"avg_SUB_ATT": 2.5, "ko_finish_rate": -1.0},
        "Well-Rounded":        {},
    }

    labels: dict[int, str] = {}
    assigned: set[str] = set()

    for idx in range(len(centroids)):
        best_label, best_score = "Well-Rounded", -np.inf
        for archetype, weights in archetypes.items():
            if archetype in assigned:
                continue
            score = sum(
                cz.loc[idx, feat] * w
                for feat, w in weights.items()
                if feat in cz.columns
            )
            if score > best_score:
                best_score, best_label = score, archetype
        labels[idx] = best_label
        if best_label != "Well-Rounded":
            assigned.add(best_label)

    return labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_cluster_summary(
    career: pd.DataFrame,
    assignments: np.ndarray,
    labels: dict[int, str],
    feature_names: list[str],
) -> None:
    df = career.copy()
    df["cluster_id"] = assignments
    df["style"] = df["cluster_id"].map(labels)

    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY — verify these make intuitive sense")
    print("=" * 60)
    for cid, style in sorted(labels.items()):
        grp = df[df["cluster_id"] == cid]
        print(f"\n[{cid}] {style}  (n={len(grp)})")
        print(grp[feature_names].mean().round(3).to_string())
        sample = grp.index[:8].tolist()
        print(f"  Sample fighters: {', '.join(sample)}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(
    kmeans: KMeans,
    scaler: StandardScaler,
    labels: dict[int, str],
    feature_names: list[str],
) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {"model": kmeans, "labels": labels, "features": feature_names}
    with open(os.path.join(MODELS_DIR, "style_kmeans.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(MODELS_DIR, "style_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModels saved → {MODELS_DIR}/")


def save_fighters_csv(
    career: pd.DataFrame,
    assignments: np.ndarray,
    labels: dict[int, str],
) -> None:
    out = career.copy()
    out["cluster_id"] = assignments
    out["style"] = out["cluster_id"].map(labels)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "fighters_with_style.csv")
    out.to_csv(path)
    print(f"Fighter style assignments saved → {path}")


def save_centroid_chart(
    kmeans: KMeans,
    scaler: StandardScaler,
    feature_names: list[str],
    labels: dict[int, str],
) -> None:
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_names,
    )
    centroids.index = [f"[{i}] {labels[i]}" for i in range(len(centroids))]

    fig, ax = plt.subplots(figsize=(11, 5))
    centroids.T.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Fighter Style Clusters — Centroid Feature Values")
    ax.set_ylabel("Average value (unscaled)")
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = os.path.join(DATA_DIR, "cluster_centroids.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Centroid chart saved → {path}")


# ---------------------------------------------------------------------------
# Public API (imported by features.py)
# ---------------------------------------------------------------------------

def load_models() -> tuple[KMeans, StandardScaler, dict[int, str], list[str]]:
    """Load saved clustering models from disk."""
    with open(os.path.join(MODELS_DIR, "style_kmeans.pkl"), "rb") as f:
        bundle = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "style_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return bundle["model"], scaler, bundle["labels"], bundle["features"]


def predict_style(
    stats: dict,
    kmeans: KMeans,
    scaler: StandardScaler,
    labels: dict[int, str],
    feature_names: list[str],
) -> str:
    """
    Return the style archetype for a fighter given their stat dict.
    Missing features default to 0.0 (below-average treatment).
    Called by features.py for per-fight rolling style assignment.
    """
    x = np.array([[stats.get(f, 0.0) for f in feature_names]])
    cluster_id = int(kmeans.predict(scaler.transform(x))[0])
    return labels[cluster_id]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(csv_path: str) -> None:
    df_raw  = load_raw_data(csv_path)
    df_long = reshape_to_long(df_raw)
    career  = extract_career_stats(df_long)

    kmeans, scaler, feature_names = fit_clustering(career)
    assignments = kmeans.predict(scaler.transform(career[feature_names].values))
    labels      = label_clusters(kmeans, scaler, feature_names)

    print_cluster_summary(career, assignments, labels, feature_names)
    save_models(kmeans, scaler, labels, feature_names)
    save_fighters_csv(career, assignments, labels)
    save_centroid_chart(kmeans, scaler, feature_names, labels)

    print("\nStep 1 complete.")
    print("Review the cluster summary above — if the style labels look off,")
    print("inspect data/cluster_centroids.png and tune the weights in label_clusters().")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/cluster.py data/raw/<ufc_csv>.csv")
        sys.exit(1)
    main(sys.argv[1])
