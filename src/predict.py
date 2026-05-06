"""
Step 4/5: Predict fight outcome with SHAP explanations and optional LLM summary.

Usage:
    python src/predict.py "Jon Jones" "Stipe Miocic"

Set ANTHROPIC_API_KEY to enable the LLM plain-English summary.
The prediction + SHAP explanation always runs regardless of whether the key is set.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from cluster import load_raw_data, normalize_name, load_models as load_cluster_models
from features import WEIGHT_CLASS_ORD
from calibration import _CalibratedModel  # noqa: F401 — required for pickle to resolve the class

ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

# Human-readable labels for each feature column
FEATURE_LABELS: dict[str, str] = {
    "avg_SIG_STR_pct_delta":     "Striking accuracy",
    "avg_SIG_STR_landed_delta":  "Striking volume",
    "avg_TD_pct_delta":          "Takedown accuracy",
    "avg_TD_landed_delta":       "Grappling output",
    "avg_SUB_ATT_delta":         "Submission attempts",
    "ko_finish_rate_delta":      "KO/TKO finish rate",
    "reach_delta":               "Reach (cm)",
    "age_delta":                 "Age",
    "experience_delta":          "Fight experience",
    "days_since_delta":          "Days since last fight",
    "style_matchup_a_winrate":   "Historical style matchup",
    "a_style_Pressure_Striker":  "A: Pressure Striker",
    "a_style_Volume_Striker":    "A: Volume Striker",
    "a_style_Well-Rounded":      "A: Well-Rounded",
    "a_style_Wrestler_Grappler": "A: Wrestler/Grappler",
    "b_style_Pressure_Striker":  "Opp: Pressure Striker",
    "b_style_Volume_Striker":    "Opp: Volume Striker",
    "b_style_Well-Rounded":      "Opp: Well-Rounded",
    "b_style_Wrestler_Grappler": "Opp: Wrestler/Grappler",
    "a_low_data":                "Limited data (A)",
    "b_low_data":                "Limited data (B)",
    "weight_class_ord":          "Weight class",
    "recent_win_rate_3_delta":   "Win rate (last 3 fights)",
    "recent_win_rate_5_delta":   "Win rate (last 5 fights)",
    "recent_finish_rate_3_delta":"Recent finish rate",
    "curr_win_streak_delta":      "Current win streak",
    "curr_lose_streak_delta":     "Current loss streak",
    "avg_absorbed_str_delta":     "Strikes absorbed (per 15 min)",
    "avg_absorbed_td_delta":      "Takedowns absorbed (per 15 min)",
    "str_net_rate_delta":         "Net striking rate",
    "td_net_rate_delta":          "Net grappling rate",
    "glicko_rating_delta":        "Glicko-2 skill rating",
    "glicko_rd_delta":            "Rating uncertainty (RD)",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_bundle() -> tuple:
    """Load trained XGBoost model bundle from disk."""
    with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


# ---------------------------------------------------------------------------
# Fighter lookup
# ---------------------------------------------------------------------------

def _fuzzy_match(name: str, candidates: set[str]) -> str:
    """Return closest matching name from candidates using rapidfuzz or difflib."""
    try:
        from rapidfuzz.process import extractOne
        match, score, _ = extractOne(name, candidates)
        if score < 65:
            raise ValueError(f"No confident match for '{name}' (best: '{match}', score={score:.0f})")
        return match
    except ImportError:
        from difflib import get_close_matches
        matches = get_close_matches(name, candidates, n=1, cutoff=0.65)
        if not matches:
            raise ValueError(f"Fighter '{name}' not found in dataset")
        return matches[0]


def _fval(v, default: float = 0.0) -> float:
    """Safe float conversion: returns default for None, NaN, or non-numeric values."""
    try:
        r = float(v)
        return default if r != r else r   # r != r is True only for NaN
    except (TypeError, ValueError):
        return default


def _absorbed_stats(all_rows: pd.DataFrame) -> dict:
    """Compute career-average absorbed strike/TD rates from fight history."""
    opp_str_sum = opp_td_sum = time_sum = 0.0
    for _, row in all_rows.iterrows():
        corner     = str(row["_corner"])
        opp_corner = "B" if corner == "R" else "R"
        opp_str = _fval(row.get(f"{opp_corner}_avg_SIG_STR_landed"))
        opp_td  = _fval(row.get(f"{opp_corner}_avg_TD_landed"))
        t_secs  = row.get("total_fight_time_secs")
        try:
            t_min = float(t_secs) / 60.0 if t_secs and float(t_secs) > 0 else 15.0
        except (TypeError, ValueError):
            t_min = 15.0
        opp_str_sum += opp_str * t_min
        opp_td_sum  += opp_td  * t_min
        time_sum    += t_min

    if time_sum <= 0:
        return {"avg_absorbed_str": 0.0, "avg_absorbed_td": 0.0,
                "str_net_rate": 0.0, "td_net_rate": 0.0}

    avg_abs_str = opp_str_sum / time_sum * 15
    avg_abs_td  = opp_td_sum  / time_sum * 15
    return {"avg_absorbed_str": avg_abs_str, "avg_absorbed_td": avg_abs_td,
            "str_net_rate": 0.0, "td_net_rate": 0.0}   # net computed below


def _recent_form(all_rows: pd.DataFrame) -> dict:
    """Compute recent-form stats from a fighter's fight history (sorted oldest-first)."""
    if all_rows.empty:
        return {"recent_win_rate_3": 0.5, "recent_win_rate_5": 0.5,
                "recent_finish_rate_3": 0.0, "curr_win_streak": 0, "curr_lose_streak": 0}

    def _won(row) -> bool:
        w = str(row["Winner"]).strip()
        c = str(row["_corner"]).strip()
        return (w in ("R", "Red") and c == "R") or (w in ("B", "Blue") and c == "B")

    def _finish(row) -> bool:
        f = str(row.get("finish", "")).upper()
        return "KO" in f or "SUB" in f

    outcomes = [(_won(r), _finish(r)) for _, r in all_rows.iterrows()]
    l3, l5 = outcomes[-3:], outcomes[-5:]

    win3 = sum(w for w, _ in l3) / len(l3) if l3 else 0.5
    win5 = sum(w for w, _ in l5) / len(l5) if l5 else 0.5
    fin3 = sum(f for _, f in l3) / len(l3) if l3 else 0.0

    # Walk backwards to find current streak
    win_streak = lose_streak = 0
    for won, _ in reversed(outcomes):
        if won:
            if lose_streak: break
            win_streak += 1
        else:
            if win_streak: break
            lose_streak += 1

    return {"recent_win_rate_3": win3, "recent_win_rate_5": win5,
            "recent_finish_rate_3": fin3, "curr_win_streak": win_streak,
            "curr_lose_streak": lose_streak}


def lookup_fighter(name: str, df: pd.DataFrame, ratings_df: pd.DataFrame | None = None) -> dict:
    """
    Find a fighter in the dataset by name (fuzzy matched) and return their
    most recent pre-fight career stats.
    """
    name_norm = normalize_name(name)
    all_names = set(
        df["R_fighter"].apply(normalize_name).tolist() +
        df["B_fighter"].apply(normalize_name).tolist()
    )
    matched = _fuzzy_match(name_norm, all_names)
    if matched != name_norm:
        print(f"  Matched '{name}' → '{matched}'")

    # Collect all appearances for this fighter
    r_rows = df[df["R_fighter"].apply(normalize_name) == matched].copy()
    b_rows = df[df["B_fighter"].apply(normalize_name) == matched].copy()
    r_rows["_corner"] = "R"
    b_rows["_corner"] = "B"

    all_rows = pd.concat([r_rows, b_rows]).sort_values("date")
    if all_rows.empty:
        raise ValueError(f"No fights found for '{matched}'")

    latest = all_rows.iloc[-1]
    c = latest["_corner"]

    ko   = _fval(latest.get(f"{c}_win_by_KO/TKO"))
    wins = _fval(latest.get(f"{c}_wins"))
    w    = _fval(latest.get(f"{c}_losses"))
    d    = _fval(latest.get(f"{c}_draw"))

    form       = _recent_form(all_rows)
    absorbed   = _absorbed_stats(all_rows)
    str_landed = _fval(latest.get(f"{c}_avg_SIG_STR_landed"))
    td_landed  = _fval(latest.get(f"{c}_avg_TD_landed"))
    absorbed["str_net_rate"] = str_landed - absorbed["avg_absorbed_str"]
    absorbed["td_net_rate"]  = td_landed  - absorbed["avg_absorbed_td"]

    # Glicko-2 rating lookup from pre-computed ratings file
    glicko_rating = 1500.0
    glicko_rd     = 350.0
    if ratings_df is not None and not ratings_df.empty and "fighter" in ratings_df.columns:
        mask = ratings_df["fighter"].apply(normalize_name) == matched
        if mask.any():
            row_r = ratings_df[mask].iloc[0]
            glicko_rating = _fval(row_r.get("glicko_rating"), 1500.0)
            glicko_rd     = _fval(row_r.get("glicko_rd"),     350.0)

    return {
        "name":               matched,
        "last_fight_date":    latest["date"],
        "weight_class":       str(latest.get("weight_class", "") or ""),
        "avg_SIG_STR_pct":    _fval(latest.get(f"{c}_avg_SIG_STR_pct")),
        "avg_SIG_STR_landed": str_landed,
        "avg_TD_pct":         _fval(latest.get(f"{c}_avg_TD_pct")),
        "avg_TD_landed":      td_landed,
        "avg_SUB_ATT":        _fval(latest.get(f"{c}_avg_SUB_ATT")),
        "ko_finish_rate":     ko / max(wins, 1),
        "reach_cms":          _fval(latest.get(f"{c}_Reach_cms")),
        "age":                _fval(latest.get(f"{c}_age")),
        "total_fights":       wins + w + d,
        "glicko_rating":      glicko_rating,
        "glicko_rd":          glicko_rd,
        **form,
        **absorbed,
    }


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def _style_to_col_suffix(style: str) -> str:
    return style.replace("/", "_").replace(" ", "_")


def get_style_winrate(style_a: str, style_b: str) -> float:
    """
    Return the most recent historical win rate of style_a vs style_b
    from matchup_features.csv. Defaults to 0.5 if no history exists.
    """
    path = os.path.join(DATA_DIR, "matchup_features.csv")
    if not os.path.exists(path):
        return 0.5
    df = pd.read_csv(path, usecols=["a_style", "b_style", "style_matchup_a_winrate", "date"])
    mask = (df["a_style"] == style_a) & (df["b_style"] == style_b)
    rows = df[mask]
    if rows.empty:
        return 0.5
    return float(rows.sort_values("date").iloc[-1]["style_matchup_a_winrate"])


def build_feature_vector(
    stats_a: dict,
    stats_b: dict,
    style_a: str,
    style_b: str,
    winrate: float,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build a single-row feature DataFrame in the exact format the model expects."""
    today = pd.Timestamp.now()
    days_a = (today - pd.Timestamp(stats_a["last_fight_date"])).days
    days_b = (today - pd.Timestamp(stats_b["last_fight_date"])).days

    # Resolve weight class: prefer fighter A's most recent class, fall back to B's
    wc = stats_a.get("weight_class") or stats_b.get("weight_class") or ""
    wc_ord = WEIGHT_CLASS_ORD.get(wc, 6)

    row: dict = {
        "avg_SIG_STR_pct_delta":    stats_a["avg_SIG_STR_pct"]     - stats_b["avg_SIG_STR_pct"],
        "avg_SIG_STR_landed_delta": stats_a["avg_SIG_STR_landed"]   - stats_b["avg_SIG_STR_landed"],
        "avg_TD_pct_delta":         stats_a["avg_TD_pct"]           - stats_b["avg_TD_pct"],
        "avg_TD_landed_delta":      stats_a["avg_TD_landed"]        - stats_b["avg_TD_landed"],
        "avg_SUB_ATT_delta":        stats_a["avg_SUB_ATT"]          - stats_b["avg_SUB_ATT"],
        "ko_finish_rate_delta":     stats_a["ko_finish_rate"]       - stats_b["ko_finish_rate"],
        "reach_delta":              stats_a["reach_cms"]            - stats_b["reach_cms"],
        "age_delta":                stats_a["age"]                  - stats_b["age"],
        "experience_delta":         stats_a["total_fights"]         - stats_b["total_fights"],
        "days_since_delta":         float(days_a - days_b),
        "style_matchup_a_winrate":    winrate,
        "weight_class_ord":           wc_ord,
        "a_low_data":                 int(stats_a["total_fights"] < 3),
        "b_low_data":                 int(stats_b["total_fights"] < 3),
        "recent_win_rate_3_delta":    stats_a["recent_win_rate_3"]    - stats_b["recent_win_rate_3"],
        "recent_win_rate_5_delta":    stats_a["recent_win_rate_5"]    - stats_b["recent_win_rate_5"],
        "recent_finish_rate_3_delta": stats_a["recent_finish_rate_3"] - stats_b["recent_finish_rate_3"],
        "curr_win_streak_delta":      stats_a["curr_win_streak"]      - stats_b["curr_win_streak"],
        "curr_lose_streak_delta":     stats_a["curr_lose_streak"]     - stats_b["curr_lose_streak"],
        "avg_absorbed_str_delta":     stats_a["avg_absorbed_str"]     - stats_b["avg_absorbed_str"],
        "avg_absorbed_td_delta":      stats_a["avg_absorbed_td"]      - stats_b["avg_absorbed_td"],
        "str_net_rate_delta":         stats_a["str_net_rate"]         - stats_b["str_net_rate"],
        "td_net_rate_delta":          stats_a["td_net_rate"]          - stats_b["td_net_rate"],
        "glicko_rating_delta":        stats_a["glicko_rating"]        - stats_b["glicko_rating"],
        "glicko_rd_delta":            stats_a["glicko_rd"]            - stats_b["glicko_rd"],
    }

    # Style one-hot columns
    for feat in feature_cols:
        if feat.startswith("a_style_"):
            suffix = feat[len("a_style_"):]
            style_a_suffix = _style_to_col_suffix(style_a)
            row[feat] = int(style_a_suffix == suffix)
        elif feat.startswith("b_style_"):
            suffix = feat[len("b_style_"):]
            style_b_suffix = _style_to_col_suffix(style_b)
            row[feat] = int(style_b_suffix == suffix)

    return pd.DataFrame([row])[feature_cols]


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def compute_shap(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return per-feature SHAP values for a single prediction.
    Positive values push toward fighter A winning; negative toward fighter B.
    Unwraps CalibratedClassifierCV to reach the underlying XGBoost tree model.
    """
    base = getattr(model, "estimator", model)
    explainer = shap.TreeExplainer(base)
    sv = explainer.shap_values(X)

    # Handle both old API (list per class) and new API (single array)
    if isinstance(sv, list):
        sv = sv[1]
    arr = np.array(sv)
    # Collapse any extra class dimension
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return arr[0]  # shape: (n_features,)


def extract_top_factors(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    feature_cols: list[str],
    n: int = 3,
) -> tuple[list[tuple], list[tuple]]:
    """
    Return top-n factors for fighter A (positive SHAP) and fighter B (negative SHAP).
    Each entry: (human_label, display_value, shap_value)

    For B's factors, delta feature values are negated so they read as B's
    positive advantage (e.g. B has +1.1 more strikes/fight, not -1.1).
    style_matchup_a_winrate is converted to B's perspective (1 - value).
    """
    pairs = sorted(
        zip(feature_cols, X.iloc[0].values, shap_vals),
        key=lambda x: x[2],
        reverse=True,
    )
    label = lambda feat: FEATURE_LABELS.get(feat, feat)
    top_a = [(label(f), v, s) for f, v, s in pairs if s > 0][:n]

    top_b = []
    for f, v, s in reversed(pairs):
        if s >= 0:
            continue
        if f.endswith("_delta"):
            display_v = abs(v)      # always show magnitude; "Factors favoring B" gives direction
        elif f == "style_matchup_a_winrate":
            display_v = 1.0 - v     # show B's historical win rate
        else:
            display_v = v
        top_b.append((label(f), display_v, s))
        if len(top_b) >= n:
            break

    return top_a, top_b


# ---------------------------------------------------------------------------
# LLM summary
# ---------------------------------------------------------------------------

def llm_summary(
    name_a: str,
    name_b: str,
    winner: str,
    prob: float,
    style_a: str,
    style_b: str,
    top_for_a: list[tuple],
    top_for_b: list[tuple],
) -> str | None:
    """Call Claude Haiku to generate a 2-3 sentence plain-English fight preview."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        def fmt_factors(factors):
            lines = []
            for label, val, shap in factors:
                if "delta" in label.lower() or label in ("Reach (cm)", "Age", "Fight experience"):
                    lines.append(f"  - {label}: {val:+.2f} advantage")
                elif "win rate" in label.lower():
                    lines.append(f"  - {label}: {val:.0%} historically")
                else:
                    lines.append(f"  - {label} (SHAP: {shap:+.3f})")
            return "\n".join(lines) if lines else "  - No dominant factors"

        prompt = (
            f"You are a concise UFC analyst. Write a 2-3 sentence fight preview.\n\n"
            f"Matchup: {name_a} ({style_a}) vs {name_b} ({style_b})\n"
            f"Model prediction: {winner} wins ({prob:.0%} confidence)\n\n"
            f"Key factors favoring {name_a}:\n{fmt_factors(top_for_a)}\n\n"
            f"Key factors favoring {name_b}:\n{fmt_factors(top_for_b)}\n\n"
            f"Be direct and specific. Reference the stats above. No filler phrases."
        )

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=220,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    except Exception as e:
        return f"[LLM summary unavailable: {e}]"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _factor_line(label: str, val: float, shap: float, winner_side: str) -> str:
    arrow = "→" if winner_side == "a" else "←"
    if "matchup" in label.lower() or "win rate" in label.lower():
        val_str = f"{val:.0%} historical win rate"
    elif "rate" in label.lower():
        val_str = f"{val:+.0%}"
    elif abs(val) >= 1:
        val_str = f"{val:+.1f}"
    else:
        val_str = f"{val:+.3f}"
    return f"  {arrow} {label}: {val_str}"


def print_prediction(
    name_a: str,
    name_b: str,
    prob: float,
    style_a: str,
    style_b: str,
    top_for_a: list[tuple],
    top_for_b: list[tuple],
    summary: str | None,
) -> None:
    winner = name_a if prob >= 0.5 else name_b
    conf   = prob if prob >= 0.5 else 1 - prob
    loser  = name_b if prob >= 0.5 else name_a

    print()
    print("=" * 58)
    print(f"  {name_a} ({style_a})")
    print(f"  vs")
    print(f"  {name_b} ({style_b})")
    print("=" * 58)
    print(f"  Predicted winner: {winner}  ({conf:.0%} confidence)")
    print("=" * 58)

    print(f"\nFactors favoring {name_a}:")
    if top_for_a:
        for label, val, shap in top_for_a:
            print(_factor_line(label, val, shap, "a"))
    else:
        print("  (no dominant factors)")

    print(f"\nFactors favoring {name_b}:")
    if top_for_b:
        for label, val, shap in top_for_b:
            print(_factor_line(label, val, shap, "b"))
    else:
        print("  (no dominant factors)")

    if summary:
        print(f"\nSummary:\n  {summary}")
    else:
        print("\n[Set ANTHROPIC_API_KEY for a plain-English summary]")
    print()


# ---------------------------------------------------------------------------
# Module-level cache — avoids reloading models/data on every bot request
# ---------------------------------------------------------------------------

_cache: dict = {}

def _load_all(csv_path: str) -> tuple:
    if "bundle" not in _cache:
        model, feature_cols             = load_model_bundle()
        kmeans, scaler, labels, cluster_features = load_cluster_models()
        df = load_raw_data(csv_path)
        for corner in ("R", "B"):
            ko   = pd.to_numeric(df[f"{corner}_win_by_KO/TKO"], errors="coerce").fillna(0)
            wins = pd.to_numeric(df[f"{corner}_wins"],          errors="coerce").fillna(0).clip(lower=1)
            df[f"{corner}_ko_finish_rate"] = (ko / wins).clip(0, 1)
        ratings_path = os.path.join(DATA_DIR, "fighter_ratings.csv")
        ratings_df = pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame()
        _cache["bundle"] = (model, feature_cols, kmeans, scaler, labels, cluster_features, df, ratings_df)
    return _cache["bundle"]


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_matchup(
    name_a: str,
    name_b: str,
    csv_path: str | None = None,
    verbose: bool = True,
) -> dict:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "raw", "ufc-master.csv")

    model, feature_cols, kmeans, scaler, labels, cluster_features, df, ratings_df = _load_all(csv_path)

    if verbose:
        print(f"\nLooking up fighters...")
    stats_a = lookup_fighter(name_a, df, ratings_df)
    stats_b = lookup_fighter(name_b, df, ratings_df)

    # Assign styles using the cluster model
    from cluster import predict_style
    style_a = predict_style(stats_a, kmeans, scaler, labels, cluster_features)
    style_b = predict_style(stats_b, kmeans, scaler, labels, cluster_features)

    # Historical style matchup win rate
    winrate = get_style_winrate(style_a, style_b)

    # Build feature vector
    X = build_feature_vector(stats_a, stats_b, style_a, style_b, winrate, feature_cols)

    # Predict
    prob_a = float(model.predict_proba(X)[0, 1])

    # SHAP explanation
    shap_vals = compute_shap(model, X)
    top_for_a, top_for_b = extract_top_factors(shap_vals, X, feature_cols)

    # LLM summary (optional)
    winner = stats_a["name"] if prob_a >= 0.5 else stats_b["name"]
    conf   = prob_a if prob_a >= 0.5 else 1 - prob_a
    summary = llm_summary(
        stats_a["name"], stats_b["name"],
        winner, conf, style_a, style_b,
        top_for_a, top_for_b,
    )

    if verbose:
        print_prediction(
            stats_a["name"], stats_b["name"],
            prob_a, style_a, style_b,
            top_for_a, top_for_b,
            summary,
        )

    return {
        "winner":     winner,
        "confidence": conf,
        "prob_a":     prob_a,
        "name_a":     stats_a["name"],
        "name_b":     stats_b["name"],
        "style_a":    style_a,
        "style_b":    style_b,
        "glicko_a":   stats_a["glicko_rating"],
        "glicko_b":   stats_b["glicko_rating"],
        "glicko_rd_a": stats_a["glicko_rd"],
        "glicko_rd_b": stats_b["glicko_rd"],
        "top_for_a":  top_for_a,
        "top_for_b":  top_for_b,
        "summary":    summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python src/predict.py \"Fighter A\" \"Fighter B\"")
        print('Example: python src/predict.py "Jon Jones" "Stipe Miocic"')
        sys.exit(1)
    predict_matchup(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
