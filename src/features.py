"""
Step 2: Engineer matchup features — one row per fight (doubled, no leakage).

The mdabbert dataset pre-computes cumulative career averages up to each fight
date in the avg_ columns, so no manual rolling is needed for the core stats.
This script derives additional features and assembles the full training matrix.

Leakage protection:
  - avg_ columns already represent pre-fight career averages (dataset guarantee)
  - Historical style matchup win rate is computed in strict date order on the
    original (undoubled) dataset — each fight only counts once in the tally
  - days_since_last_fight is computed from fight dates in the dataset

Run:
    python src/features.py data/raw/ufc-master.csv

Output:
    data/matchup_features.csv
"""

import os
import sys
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cluster import load_raw_data, normalize_name, load_models

ROOT     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")

# Stat columns used for both style assignment and delta computation.
# Must match STYLE_FEATURES + derived features in cluster.py.
STAT_COLS = [
    "avg_SIG_STR_pct",
    "avg_SIG_STR_landed",
    "avg_TD_pct",
    "avg_TD_landed",
    "avg_SUB_ATT",
    "ko_finish_rate",   # derived below
]

# Stats that get exponentially-weighted counterparts (decay_rate = 0.13/year).
# Must NOT include cluster-style features (those live in STAT_COLS only).
EW_STAT_COLS = list(STAT_COLS) + ["knockdown_rate", "ctrl_rate"]

# Ordinal encoding ordered by weight limit. Each class gets a unique integer so
# XGBoost can learn division-specific patterns (KO rates, grappling dominance, etc.)
WEIGHT_CLASS_ORD: dict[str, int] = {
    "Women's Strawweight":   1,   # 115 lbs
    "Women's Flyweight":     2,   # 125 lbs
    "Flyweight":             3,   # 125 lbs (men's)
    "Women's Bantamweight":  4,   # 135 lbs
    "Bantamweight":          5,   # 135 lbs (men's)
    "Women's Featherweight": 6,   # 145 lbs
    "Featherweight":         7,   # 145 lbs (men's)
    "Lightweight":           8,   # 155 lbs
    "Welterweight":          9,   # 170 lbs
    "Middleweight":         10,   # 185 lbs
    "Light Heavyweight":    11,   # 205 lbs
    "Heavyweight":          12,   # 265 lbs
    "Catch Weight":          6,   # map to mid-range
}

# Exponential time-decay rate for career stat averages (per year)
# Weight for a fight t years ago = exp(-0.13 * t); ~8-year-old fight weighs ~3x less than a recent one
_EW_DECAY = 0.13

# Glicko-2 hyperparameters
_GLICKO_TAU         = 0.5       # system constant — limits how fast volatility changes
_GLICKO_INIT_R      = 1500.0    # starting rating for a new fighter
_GLICKO_INIT_RD     = 350.0     # starting RD (high uncertainty at debut)
_GLICKO_INIT_SIG    = 0.06      # starting volatility
_GLICKO_PERIOD_DAYS = 180       # one "rating period" = 6 months for RD decay
_MU_SCALE           = 173.7178  # converts between Glicko-2 internal μ and Elo-like scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_fights(df: pd.DataFrame, corner: str) -> pd.Series:
    w = pd.to_numeric(df[f"{corner}_wins"],   errors="coerce").fillna(0)
    l = pd.to_numeric(df[f"{corner}_losses"], errors="coerce").fillna(0)
    d = pd.to_numeric(df[f"{corner}_draw"],   errors="coerce").fillna(0)
    return w + l + d


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def add_ko_finish_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add {R,B}_ko_finish_rate: KO/TKO wins ÷ total wins, clamped to [0, 1]."""
    df = df.copy()
    for corner in ("R", "B"):
        ko   = pd.to_numeric(df[f"{corner}_win_by_KO/TKO"], errors="coerce").fillna(0)
        wins = pd.to_numeric(df[f"{corner}_wins"],          errors="coerce").fillna(0).clip(lower=1)
        df[f"{corner}_ko_finish_rate"] = (ko / wins).clip(0, 1)
    return df


def add_kd_ctrl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive knockdown and grappling-control proxy features from cumulative win-method counts.

    knockdown_rate: KO/TKO wins ÷ total fights  (per-fight KO danger, not just per-win)
    ctrl_rate:      submission wins ÷ wins       (fraction of wins by submission — grappling quality)

    Both use data that's available for every fighter in the base dataset.
    When the scraper adds real avg_KD / avg_ctrl_secs columns, those can be used alongside.
    """
    df = df.copy()
    for corner in ("R", "B"):
        ko  = pd.to_numeric(df[f"{corner}_win_by_KO/TKO"],    errors="coerce").fillna(0)
        sub = pd.to_numeric(df[f"{corner}_win_by_Submission"], errors="coerce").fillna(0)
        wins  = pd.to_numeric(df[f"{corner}_wins"],   errors="coerce").fillna(0).clip(lower=1)
        total = _total_fights(df, corner).clip(lower=1)
        df[f"{corner}_knockdown_rate"] = (ko  / total).clip(0, 1)
        df[f"{corner}_ctrl_rate"]      = (sub / wins ).clip(0, 1)
    return df


def add_method_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add win/loss method distribution features in strict date order.

    Derived from cumulative win-method counts (no rolling needed):
        {R,B}_decision_rate  — decision wins / wins (style indicator)

    Computed in date order (rolling, no leakage):
        {R,B}_finish_loss_rate  — losses stopped by KO or submission (vulnerability)
        {R,B}_punch_finish_rate — KO wins via punches/elbows (head-strike targeting proxy)
        {R,B}_kick_finish_rate  — KO wins via kicks/knees   (leg/body-strike targeting proxy)
    """
    PUNCH_DETAILS = {
        "Punch", "Punches", "Elbow", "Elbows",
        "Spinning Back Elbow", "Spinning Back Fist",
        "Punch to Head At Distance", "Punch to Body At Distance",
        "Punches to Head On Ground",
        "Elbows to Head From Half Guard", "Elbows to Body From Half Guard",
    }
    KICK_DETAILS = {
        "Kick", "Kicks", "Knee", "Knees", "Flying Knee",
        "Kick to Head At Distance",
        "Knee to Body In Clinch", "Knee to Head At Distance",
        "Spinning Back Kick",
    }

    df = df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)

    # State per fighter: [finish_loss_count, loss_count, punch_ko_count, kick_ko_count, ko_win_count]
    state: dict = {}

    arrays: dict[str, np.ndarray] = {
        "R_finish_loss_rate":  np.zeros(n),
        "B_finish_loss_rate":  np.zeros(n),
        "R_punch_finish_rate": np.zeros(n),
        "B_punch_finish_rate": np.zeros(n),
        "R_kick_finish_rate":  np.zeros(n),
        "B_kick_finish_rate":  np.zeros(n),
    }

    finish_s  = df["finish"].fillna("").astype(str).str.upper()
    details_s = df["finish_details"].fillna("").astype(str)

    for i in range(n):
        row     = df.iloc[i]
        r_name  = normalize_name(str(row["R_fighter"]))
        b_name  = normalize_name(str(row["B_fighter"]))
        winner  = str(row["Winner"]).strip()
        finish  = finish_s.iloc[i]
        details = details_s.iloc[i]

        is_ko_finish  = finish == "KO/TKO"
        is_sub_finish = finish == "SUB"
        is_punch      = details in PUNCH_DETAILS
        is_kick       = details in KICK_DETAILS

        for corner, name in [("R", r_name), ("B", b_name)]:
            if name not in state:
                state[name] = [0, 0, 0, 0, 0]
            fl, lc, pk, kk, ko = state[name]

            # Record pre-fight stats (before update)
            arrays[f"{corner}_finish_loss_rate"][i]  = fl / lc if lc > 0 else 0.0
            arrays[f"{corner}_punch_finish_rate"][i] = pk / ko if ko > 0 else 0.0
            arrays[f"{corner}_kick_finish_rate"][i]  = kk / ko if ko > 0 else 0.0

            # Determine this fight's outcome for this corner
            won  = (winner in ("R", "Red")  and corner == "R") or \
                   (winner in ("B", "Blue") and corner == "B")
            lost = (winner in ("R", "Red")  and corner == "B") or \
                   (winner in ("B", "Blue") and corner == "R")

            if won and is_ko_finish:
                ko += 1
                if is_punch: pk += 1
                if is_kick:  kk += 1
            if lost:
                lc += 1
                if is_ko_finish or is_sub_finish:
                    fl += 1

            state[name] = [fl, lc, pk, kk, ko]

    for col, arr in arrays.items():
        df[col] = arr

    # Simple derived: decision rate — ratio of cumulative columns, no rolling needed
    for corner in ("R", "B"):
        dec_u = pd.to_numeric(df[f"{corner}_win_by_Decision_Unanimous"], errors="coerce").fillna(0)
        dec_s = pd.to_numeric(df[f"{corner}_win_by_Decision_Split"],     errors="coerce").fillna(0)
        dec_m = pd.to_numeric(df[f"{corner}_win_by_Decision_Majority"],  errors="coerce").fillna(0)
        wins  = pd.to_numeric(df[f"{corner}_wins"], errors="coerce").fillna(0).clip(lower=1)
        df[f"{corner}_decision_rate"] = ((dec_u + dec_s + dec_m) / wins).clip(0, 1)

    return df


def add_days_since_last_fight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add {R,B}_days_since: days between this fight and the fighter's previous
    appearance in the dataset. NaN for a fighter's first recorded fight.
    """
    df = df.copy()

    # Build one row per fighter per fight, then compute gap to previous fight
    long = pd.concat([
        df[["date", "R_fighter"]].rename(columns={"R_fighter": "fighter"}),
        df[["date", "B_fighter"]].rename(columns={"B_fighter": "fighter"}),
    ]).copy()
    long["fighter"] = long["fighter"].apply(normalize_name)
    long = long.sort_values("date").drop_duplicates(subset=["fighter", "date"])
    long["prev_date"] = long.groupby("fighter")["date"].shift(1)
    long["days_since"] = (long["date"] - long["prev_date"]).dt.days

    lookup = long.set_index(["fighter", "date"])["days_since"].to_dict()

    for corner, col in [("R", "R_fighter"), ("B", "B_fighter")]:
        names = df[col].apply(normalize_name)
        df[f"{corner}_days_since"] = [
            lookup.get((n, d), np.nan) for n, d in zip(names, df["date"])
        ]
    return df


# ---------------------------------------------------------------------------
# Style assignment
# ---------------------------------------------------------------------------

def add_style_labels(
    df: pd.DataFrame,
    kmeans,
    scaler,
    labels: dict,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Assign a style archetype per fighter per fight using their pre-fight
    cumulative career stats (avg_ columns + ko_finish_rate).
    """
    df = df.copy()
    for corner in ("R", "B"):
        X = np.column_stack([
            pd.to_numeric(
                df[f"{corner}_{f}"] if f != "ko_finish_rate"
                else df[f"{corner}_ko_finish_rate"],
                errors="coerce",
            ).fillna(0).values
            for f in feature_names
        ])
        cluster_ids = kmeans.predict(scaler.transform(X))
        df[f"{corner}_style"] = [labels[int(c)] for c in cluster_ids]
    return df


# ---------------------------------------------------------------------------
# Rolling style matchup win rate
# ---------------------------------------------------------------------------

def add_style_winrates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fight (in date order), compute the historical win rate of
    R_style vs B_style using only fights strictly before this date.
    Stored as R_style_winrate on the original df; B perspective = 1 - this.
    Defaults to 0.5 when no prior history exists for a matchup pair.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    wins:   dict[tuple, int] = {}
    totals: dict[tuple, int] = {}
    rates = []

    for _, row in df.iterrows():
        key  = (row["R_style"], row["B_style"])
        rate = wins.get(key, 0) / totals[key] if key in totals else 0.5
        rates.append(rate)

        # Update tally AFTER recording the feature (strict no-leakage)
        if row["Winner"] == "R":
            wins[key] = wins.get(key, 0) + 1
        totals[key] = totals.get(key, 0) + 1

    df["R_style_winrate"] = rates
    return df


# ---------------------------------------------------------------------------
# Glicko-2 ratings
# ---------------------------------------------------------------------------

def _glicko2_update(
    r: float, rd: float, sigma: float,
    opp_r: float, opp_rd: float, score: float,
) -> tuple[float, float, float]:
    """
    Single-fight Glicko-2 update. Returns (new_r, new_rd, new_sigma).
    score: 1.0 = win, 0.5 = draw, 0.0 = loss.
    Implements Glickman (2012) with the Illinois root-finding algorithm.
    """
    tau = _GLICKO_TAU

    mu    = (r     - 1500.0) / _MU_SCALE
    phi   = rd               / _MU_SCALE
    mu_j  = (opp_r - 1500.0) / _MU_SCALE
    phi_j = opp_rd            / _MU_SCALE

    g_j   = 1.0 / np.sqrt(1.0 + 3.0 * phi_j**2 / np.pi**2)
    E_j   = 1.0 / (1.0 + np.exp(-g_j * (mu - mu_j)))
    v     = 1.0 / (g_j**2 * E_j * (1.0 - E_j))
    delta = v * g_j * (score - E_j)

    a = np.log(sigma**2)

    def f(x: float) -> float:
        ex = np.exp(x)
        d2 = phi**2 + v + ex
        return (ex * (delta**2 - phi**2 - v - ex) / (2.0 * d2**2)
                - (x - a) / tau**2)

    A = a
    if delta**2 > phi**2 + v:
        B = np.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
        B = a - k * tau

    fA, fB = f(A), f(B)
    for _ in range(500):
        if abs(B - A) < 1e-6:
            break
        C  = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0.0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC

    sigma_new = np.exp(A / 2.0)
    phi_star  = np.sqrt(phi**2 + sigma_new**2)
    phi_new   = 1.0 / np.sqrt(1.0 / phi_star**2 + 1.0 / v)
    mu_new    = mu + phi_new**2 * g_j * (score - E_j)

    return _MU_SCALE * mu_new + 1500.0, _MU_SCALE * phi_new, sigma_new


def add_glicko_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-fighter Glicko-2 ratings in strict date order and attach them
    as pre-fight features (no leakage: ratings recorded BEFORE each fight,
    updated AFTER). RD widens automatically during inactivity.

    Adds columns:
        {R,B}_glicko_rating  — estimated skill (~1500 baseline)
        {R,B}_glicko_rd      — rating deviation / uncertainty (~350 debut → ~80 active)

    Also saves data/fighter_ratings.csv so predict.py can look up current ratings.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)

    PHI_INIT = _GLICKO_INIT_RD / _MU_SCALE  # RD decay cap

    # State: fighter → [r, rd, sigma, last_date]
    state: dict[str, list] = {}

    def get_pre_fight(name: str, fight_date) -> tuple[float, float, float]:
        if name not in state:
            return _GLICKO_INIT_R, _GLICKO_INIT_RD, _GLICKO_INIT_SIG
        r, rd, sigma, last_date = state[name]
        days = (fight_date - last_date).days
        if days > 0:
            phi     = rd / _MU_SCALE
            periods = days / _GLICKO_PERIOD_DAYS
            rd      = min(np.sqrt(phi**2 + sigma**2 * periods), PHI_INIT) * _MU_SCALE
        return r, rd, sigma

    r_ratings = np.full(n, _GLICKO_INIT_R)
    r_rds     = np.full(n, _GLICKO_INIT_RD)
    b_ratings = np.full(n, _GLICKO_INIT_R)
    b_rds     = np.full(n, _GLICKO_INIT_RD)

    for i in range(n):
        row    = df.iloc[i]
        date   = row["date"]
        r_name = normalize_name(str(row["R_fighter"]))
        b_name = normalize_name(str(row["B_fighter"]))

        r_r, r_rd, r_sig = get_pre_fight(r_name, date)
        b_r, b_rd, b_sig = get_pre_fight(b_name, date)

        r_ratings[i] = r_r;  r_rds[i] = r_rd
        b_ratings[i] = b_r;  b_rds[i] = b_rd

        winner  = str(row.get("Winner", "")).strip()
        r_score = 1.0 if winner in ("R", "Red") else (0.0 if winner in ("B", "Blue") else 0.5)
        b_score = 1.0 - r_score

        new_r_r, new_r_rd, new_r_sig = _glicko2_update(r_r, r_rd, r_sig, b_r, b_rd, r_score)
        new_b_r, new_b_rd, new_b_sig = _glicko2_update(b_r, b_rd, b_sig, r_r, r_rd, b_score)

        state[r_name] = [new_r_r, new_r_rd, new_r_sig, date]
        state[b_name] = [new_b_r, new_b_rd, new_b_sig, date]

    df["R_glicko_rating"] = r_ratings
    df["R_glicko_rd"]     = r_rds
    df["B_glicko_rating"] = b_ratings
    df["B_glicko_rd"]     = b_rds

    # Persist final ratings for predict.py lookups
    records = [
        {"fighter": name, "glicko_rating": s[0], "glicko_rd": s[1], "glicko_sigma": s[2]}
        for name, s in state.items()
    ]
    ratings_df = pd.DataFrame(records).sort_values("glicko_rating", ascending=False)
    os.makedirs(DATA_DIR, exist_ok=True)
    ratings_path = os.path.join(DATA_DIR, "fighter_ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)

    print(f"  Glicko-2 ratings computed for {len(state):,} fighters → {ratings_path}")
    return df


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the doubled feature matrix: for every fight, create two rows —
    one with A=Red/B=Blue and one with A=Blue/B=Red. Delta features for the
    second perspective are the negation of the first.

    Style matchup win rate uses the pre-computed R_style_winrate column:
      A=R perspective → R_style_winrate
      A=B perspective → 1 - R_style_winrate  (complement, leakage-free)
    """
    perspectives = []

    for a, b in [("R", "B"), ("B", "R")]:
        is_r_perspective = (a == "R")

        rows: dict = {
            "fight_id":     df.index.values,
            "a_fighter":    df[f"{a}_fighter"].apply(normalize_name).values,
            "b_fighter":    df[f"{b}_fighter"].apply(normalize_name).values,
            "date":         df["date"].values,
            "weight_class": df["weight_class"].values if "weight_class" in df.columns else np.nan,
            "a_won":        (df["Winner"] == a).astype(int).values,
            "a_style":      df[f"{a}_style"].values,
            "b_style":      df[f"{b}_style"].values,
            "style_matchup": (df[f"{a}_style"] + "_vs_" + df[f"{b}_style"]).values,
            "style_matchup_a_winrate": (
                df["R_style_winrate"].values if is_r_perspective
                else (1 - df["R_style_winrate"]).values
            ),
            "a_low_data":   (_total_fights(df, a) < 3).astype(int).values,
            "b_low_data":   (_total_fights(df, b) < 3).astype(int).values,
        }

        # Sign multiplier: +1 for A=R perspective, -1 for A=B (negates all deltas)
        sign = 1 if is_r_perspective else -1

        # Stat deltas
        for stat in STAT_COLS:
            r_col = f"R_{stat}" if stat != "ko_finish_rate" else "R_ko_finish_rate"
            b_col = f"B_{stat}" if stat != "ko_finish_rate" else "B_ko_finish_rate"
            a_vals = pd.to_numeric(df[r_col], errors="coerce").fillna(0)
            b_vals = pd.to_numeric(df[b_col], errors="coerce").fillna(0)
            rows[f"{stat}_delta"] = (sign * (a_vals - b_vals)).values

        # Knockdown & control-time proxy deltas (columns added by add_kd_ctrl_features)
        for stat, fill in [("knockdown_rate", 0.0), ("ctrl_rate", 0.0)]:
            r_v = pd.to_numeric(df.get(f"R_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            b_v = pd.to_numeric(df.get(f"B_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            rows[f"{stat}_delta"] = (sign * (r_v - b_v)).values

        # Physical / meta deltas
        rows["reach_delta"] = (sign * (
            pd.to_numeric(df["R_Reach_cms"], errors="coerce") -
            pd.to_numeric(df["B_Reach_cms"], errors="coerce")
        )).fillna(0).values

        r_age = pd.to_numeric(df["R_age"], errors="coerce").fillna(0)
        b_age = pd.to_numeric(df["B_age"], errors="coerce").fillna(0)
        rows["age_delta"] = (sign * (r_age - b_age)).values
        # Absolute age: captures non-linear prime/decline effects the delta alone cannot
        rows["a_age"] = (r_age if is_r_perspective else b_age).values
        rows["b_age"] = (b_age if is_r_perspective else r_age).values

        rows["experience_delta"] = (
            sign * (_total_fights(df, "R") - _total_fights(df, "B"))
        ).values

        r_days = pd.to_numeric(df["R_days_since"], errors="coerce")
        b_days = pd.to_numeric(df["B_days_since"], errors="coerce")
        rows["days_since_delta"] = (sign * (r_days - b_days)).fillna(0).values

        # Absorbed / defensive deltas (columns added by add_absorbed_stats)
        for stat, fill in [
            ("avg_absorbed_str", 0.0),
            ("avg_absorbed_td",  0.0),
            ("str_net_rate",     0.0),
            ("td_net_rate",      0.0),
        ]:
            r_v = pd.to_numeric(df.get(f"R_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            b_v = pd.to_numeric(df.get(f"B_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            rows[f"{stat}_delta"] = (sign * (r_v - b_v)).values

        # Recent form deltas (columns added by add_recent_form)
        for stat, fill in [
            ("recent_win_rate_3",    0.5),
            ("recent_win_rate_5",    0.5),
            ("recent_finish_rate_3", 0.0),
            ("curr_win_streak",      0.0),
            ("curr_lose_streak",     0.0),
        ]:
            r_v = pd.to_numeric(df.get(f"R_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            b_v = pd.to_numeric(df.get(f"B_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            rows[f"{stat}_delta"] = (sign * (r_v - b_v)).values

        # Glicko-2 rating deltas (columns added by add_glicko_ratings)
        for stat, fill in [
            ("glicko_rating", _GLICKO_INIT_R),
            ("glicko_rd",     _GLICKO_INIT_RD),
        ]:
            r_v = pd.to_numeric(df.get(f"R_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            b_v = pd.to_numeric(df.get(f"B_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            rows[f"{stat}_delta"] = (sign * (r_v - b_v)).values

        # Exponentially-weighted stat deltas (columns added by add_ew_stats)
        for stat in EW_STAT_COLS:
            r_v = pd.to_numeric(df.get(f"R_ew_{stat}", pd.Series(0.0, index=df.index)),
                                errors="coerce").fillna(0.0)
            b_v = pd.to_numeric(df.get(f"B_ew_{stat}", pd.Series(0.0, index=df.index)),
                                errors="coerce").fillna(0.0)
            rows[f"ew_{stat}_delta"] = (sign * (r_v - b_v)).values

        # Win/loss method distribution deltas (columns added by add_method_distribution)
        for stat, fill in [
            ("decision_rate",    0.0),
            ("finish_loss_rate", 0.0),
            ("punch_finish_rate",0.0),
            ("kick_finish_rate", 0.0),
        ]:
            r_v = pd.to_numeric(df.get(f"R_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            b_v = pd.to_numeric(df.get(f"B_{stat}", pd.Series(fill, index=df.index)),
                                errors="coerce").fillna(fill)
            rows[f"{stat}_delta"] = (sign * (r_v - b_v)).values

        perspectives.append(pd.DataFrame(rows))

    return (
        pd.concat(perspectives, ignore_index=True)
          .sort_values(["date", "fight_id"])
          .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Recent form features
# ---------------------------------------------------------------------------

def add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-fighter rolling form features in strict date order (no leakage).
    Each fight records the fighter's stats from their PREVIOUS fights, then
    updates the tally — so the current fight's outcome never contaminates its
    own feature values.

    Adds columns:
        {R,B}_recent_win_rate_3    — win rate in last 3 fights (0.5 if no history)
        {R,B}_recent_win_rate_5    — win rate in last 5 fights (0.5 if no history)
        {R,B}_recent_finish_rate_3 — KO/sub finish rate in last 3 fights
        {R,B}_curr_win_streak      — current win streak entering this fight
        {R,B}_curr_lose_streak     — current loss streak entering this fight
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)

    # Per-fighter running state
    history:     dict[str, deque] = {}   # deque of (won: bool, is_finish: bool), maxlen=5
    win_streak:  dict[str, int]   = {}
    lose_streak: dict[str, int]   = {}

    arrays: dict[str, np.ndarray] = {
        "R_recent_win_rate_3":    np.full(n, 0.5),
        "R_recent_win_rate_5":    np.full(n, 0.5),
        "R_recent_finish_rate_3": np.zeros(n),
        "R_curr_win_streak":      np.zeros(n),
        "R_curr_lose_streak":     np.zeros(n),
        "B_recent_win_rate_3":    np.full(n, 0.5),
        "B_recent_win_rate_5":    np.full(n, 0.5),
        "B_recent_finish_rate_3": np.zeros(n),
        "B_curr_win_streak":      np.zeros(n),
        "B_curr_lose_streak":     np.zeros(n),
    }

    finish_series = (df["finish"].fillna("").astype(str).str.upper()
                     if "finish" in df.columns
                     else pd.Series([""] * n, index=df.index))

    for i in range(n):
        row      = df.iloc[i]
        r_name   = normalize_name(row["R_fighter"])
        b_name   = normalize_name(row["B_fighter"])
        winner   = str(row["Winner"]).strip()
        r_won    = winner in ("R", "Red")
        b_won    = winner in ("B", "Blue")
        finish   = finish_series.iloc[i]
        is_fin   = "KO" in finish or "SUB" in finish

        for corner, name, won in [("R", r_name, r_won), ("B", b_name, b_won)]:
            h    = history.get(name, deque(maxlen=5))
            hist = list(h)
            l3   = hist[-3:];  l5 = hist[-5:]

            arrays[f"{corner}_recent_win_rate_3"][i]    = (sum(w for w,_ in l3) / len(l3)) if l3 else 0.5
            arrays[f"{corner}_recent_win_rate_5"][i]    = (sum(w for w,_ in l5) / len(l5)) if l5 else 0.5
            arrays[f"{corner}_recent_finish_rate_3"][i] = (sum(f for _,f in l3) / len(l3)) if l3 else 0.0
            arrays[f"{corner}_curr_win_streak"][i]      = win_streak.get(name, 0)
            arrays[f"{corner}_curr_lose_streak"][i]     = lose_streak.get(name, 0)

            # Update AFTER recording (no leakage)
            if name not in history:
                history[name] = deque(maxlen=5)
            history[name].append((won, is_fin))

            if won:
                win_streak[name]  = win_streak.get(name, 0) + 1
                lose_streak[name] = 0
            elif r_won or b_won:     # actual loss (not a draw/NC)
                lose_streak[name] = lose_streak.get(name, 0) + 1
                win_streak[name]  = 0
            # draw / NC: leave streaks unchanged

    for col, arr in arrays.items():
        df[col] = arr

    return df


# ---------------------------------------------------------------------------
# Absorbed / defensive stats
# ---------------------------------------------------------------------------

def _fight_time_min(row) -> float:
    """Return fight duration in minutes. Falls back to round info when column is null."""
    t = row.get("total_fight_time_secs")
    try:
        if t and float(t) > 0:
            return float(t) / 60.0
    except (TypeError, ValueError):
        pass
    # Derive from finish_round + finish_round_time
    try:
        fr = int(float(row.get("finish_round", 3) or 3))
        ft = str(row.get("finish_round_time", "5:00") or "5:00")
        m, s = ft.split(":")
        return (fr - 1) * 5.0 + float(m) + float(s) / 60.0
    except Exception:
        return 15.0   # three-round decision fallback


def add_absorbed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-fighter rolling absorbed-strike stats in strict date order.

    For each fight, the opponent's pre-fight career strike/TD rate is used as
    a proxy for what the fighter absorbs — updated AFTER recording so no
    fight's result contaminates its own feature.

    Adds columns:
        {R,B}_avg_absorbed_str  — avg sig strikes absorbed per 15 min
        {R,B}_avg_absorbed_td   — avg takedowns absorbed per 15 min
        {R,B}_str_net_rate      — avg_SIG_STR_landed minus avg_absorbed_str
        {R,B}_td_net_rate       — avg_TD_landed minus avg_absorbed_td
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)

    # Cumulative weighted sums per fighter
    opp_str_sum:  dict[str, float] = {}   # sum(opp_str_rate * fight_min)
    opp_td_sum:   dict[str, float] = {}
    time_sum:     dict[str, float] = {}   # sum(fight_min)

    arrays = {
        "R_avg_absorbed_str": np.zeros(n),
        "R_avg_absorbed_td":  np.zeros(n),
        "B_avg_absorbed_str": np.zeros(n),
        "B_avg_absorbed_td":  np.zeros(n),
    }

    for i in range(n):
        row    = df.iloc[i]
        r_name = normalize_name(row["R_fighter"])
        b_name = normalize_name(row["B_fighter"])
        t_min  = _fight_time_min(row)

        # Opponent's pre-fight striking rates (proxy for absorption this fight)
        r_opp_str = float(row.get("B_avg_SIG_STR_landed", 0) or 0)   # B throws → R absorbs
        r_opp_td  = float(row.get("B_avg_TD_landed",      0) or 0)
        b_opp_str = float(row.get("R_avg_SIG_STR_landed", 0) or 0)   # R throws → B absorbs
        b_opp_td  = float(row.get("R_avg_TD_landed",      0) or 0)

        for corner, name, opp_str, opp_td in [
            ("R", r_name, r_opp_str, r_opp_td),
            ("B", b_name, b_opp_str, b_opp_td),
        ]:
            total_t = time_sum.get(name, 0.0)
            if total_t > 0:
                arrays[f"{corner}_avg_absorbed_str"][i] = opp_str_sum[name] / total_t * 15
                arrays[f"{corner}_avg_absorbed_td"][i]  = opp_td_sum[name]  / total_t * 15
            # else: 0 (no history before debut)

            # Update AFTER recording
            opp_str_sum[name] = opp_str_sum.get(name, 0.0) + opp_str * t_min
            opp_td_sum[name]  = opp_td_sum.get(name,  0.0) + opp_td  * t_min
            time_sum[name]    = total_t + t_min

    for col, arr in arrays.items():
        df[col] = arr

    # Derived net rates (landed minus absorbed)
    for corner in ("R", "B"):
        landed_str = pd.to_numeric(df[f"{corner}_avg_SIG_STR_landed"], errors="coerce").fillna(0)
        landed_td  = pd.to_numeric(df[f"{corner}_avg_TD_landed"],      errors="coerce").fillna(0)
        df[f"{corner}_str_net_rate"] = landed_str - df[f"{corner}_avg_absorbed_str"]
        df[f"{corner}_td_net_rate"]  = landed_td  - df[f"{corner}_avg_absorbed_td"]

    return df


# ---------------------------------------------------------------------------
# Exponentially-weighted career stat averages
# ---------------------------------------------------------------------------

def add_ew_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace simple career averages with exponentially time-decayed equivalents.
    Each historical fight contributes weight exp(-0.13 * years_ago), so fights
    from ~8 years ago weigh roughly 1/3 as much as fights from last month.

    Adds columns:
        {R,B}_ew_{stat}  for each stat in EW_STAT_COLS

    Also saves data/fighter_ew_stats.csv so predict.py can look up current values.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)

    # State per fighter per stat: [ew_num, ew_den, last_avg, last_n, last_date]
    state: dict = {}

    out: dict[str, np.ndarray] = {}
    for stat in EW_STAT_COLS:
        out[f"R_ew_{stat}"] = np.zeros(n)
        out[f"B_ew_{stat}"] = np.zeros(n)

    def _n_fights(row, corner: str) -> float:
        w = max(float(row.get(f"{corner}_wins",   0) or 0), 0)
        l = max(float(row.get(f"{corner}_losses", 0) or 0), 0)
        d = max(float(row.get(f"{corner}_draw",   0) or 0), 0)
        return w + l + d

    for i in range(n):
        row        = df.iloc[i]
        fight_date = row["date"]
        r_name     = normalize_name(str(row["R_fighter"]))
        b_name     = normalize_name(str(row["B_fighter"]))

        for corner, name in [("R", r_name), ("B", b_name)]:
            if name not in state:
                state[name] = {s: [0.0, 0.0, 0.0, 0.0, None] for s in EW_STAT_COLS}

            n_cur = _n_fights(row, corner)

            for stat in EW_STAT_COLS:
                ew_num, ew_den, last_avg, last_n, last_date = state[name][stat]

                # Decay existing EW sums since last appearance
                if last_date is not None:
                    years = (fight_date - last_date).days / 365.25
                    factor = np.exp(-_EW_DECAY * years)
                    ew_num *= factor
                    ew_den *= factor

                # Record pre-fight EW average (no leakage — before update)
                out[f"{corner}_ew_{stat}"][i] = ew_num / ew_den if ew_den > 0 else 0.0

                # Read pre-fight cumulative avg from dataset
                col = f"{corner}_ko_finish_rate" if stat == "ko_finish_rate" else f"{corner}_{stat}"
                cur_avg = 0.0
                if col in df.columns:
                    v = df.at[i, col]
                    try:
                        fv = float(v)
                        cur_avg = 0.0 if fv != fv else fv  # guard NaN
                    except (TypeError, ValueError):
                        pass

                # Back-calculate per-fight contribution and update EW state
                # cur_avg is the average over n_cur previous fights;
                # last_avg was the average over last_n previous fights.
                # Δ = cur_avg * n_cur - last_avg * last_n  (stat total added since last row)
                if n_cur > last_n:
                    n_new = n_cur - last_n
                    delta = cur_avg * n_cur - last_avg * last_n
                    ew_num += delta
                    ew_den += n_new

                state[name][stat] = [ew_num, ew_den, cur_avg, n_cur, fight_date]

    for col, arr in out.items():
        df[col] = arr

    # Persist final EW values for predict.py lookups
    records = []
    for name, stats in state.items():
        rec = {"fighter": name}
        for stat, (ew_num, ew_den, *_) in stats.items():
            rec[f"ew_{stat}"] = ew_num / ew_den if ew_den > 0 else 0.0
        records.append(rec)
    ew_df = pd.DataFrame(records)
    os.makedirs(DATA_DIR, exist_ok=True)
    ew_path = os.path.join(DATA_DIR, "fighter_ew_stats.csv")
    ew_df.to_csv(ew_path, index=False)
    print(f"  EW stats computed for {len(records):,} fighters → {ew_path}")
    return df


# ---------------------------------------------------------------------------
# Weight class ordinal encoding
# ---------------------------------------------------------------------------

def add_weight_class_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Map weight_class string → integer ordinal column `weight_class_ord`."""
    df = df.copy()
    df["weight_class_ord"] = (
        df["weight_class"]
        .map(WEIGHT_CLASS_ORD)
        .fillna(6)      # unknown → mid-range default
        .astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Style one-hot encoding
# ---------------------------------------------------------------------------

def add_style_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode a_style and b_style columns."""
    df = df.copy()
    all_styles = sorted(set(df["a_style"].unique()) | set(df["b_style"].unique()))
    for style in all_styles:
        safe = style.replace("/", "_").replace(" ", "_")
        df[f"a_style_{safe}"] = (df["a_style"] == style).astype(int)
        df[f"b_style_{safe}"] = (df["b_style"] == style).astype(int)
    return df


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def print_sanity_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SANITY CHECK — verify a few known matchups look right")
    print("=" * 60)

    known = [
        ("Khabib Nurmagomedov", "Conor Mcgregor"),
        ("Jon Jones",           "Stipe Miocic"),
        ("Israel Adesanya",     "Alex Pereira"),
        ("Amanda Nunes",        "Ronda Rousey"),
    ]
    for a, b in known:
        row = df[(df["a_fighter"] == a) & (df["b_fighter"] == b)]
        if row.empty:
            row = df[(df["a_fighter"] == b) & (df["b_fighter"] == a)]
        if not row.empty:
            r = row.iloc[0]
            print(
                f"\n  {r['a_fighter']} ({r['a_style']}) vs "
                f"{r['b_fighter']} ({r['b_style']})  |  "
                f"a_won={r['a_won']}  "
                f"reach_delta={r['reach_delta']:.1f}  "
                f"ko_rate_delta={r['ko_finish_rate_delta']:.2f}"
            )
        else:
            print(f"\n  {a} vs {b} — not found in dataset")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str) -> None:
    print("Loading data and cluster models...")
    df              = load_raw_data(csv_path)
    kmeans, scaler, labels, feature_names = load_models()

    print("Deriving ko_finish_rate...")
    df = add_ko_finish_rate(df)

    print("Deriving knockdown rate and grappling control rate...")
    df = add_kd_ctrl_features(df)

    print("Computing win/loss method distribution...")
    df = add_method_distribution(df)

    print("Computing days since last fight...")
    df = add_days_since_last_fight(df)

    print("Assigning style labels per fight...")
    df = add_style_labels(df, kmeans, scaler, labels, feature_names)

    print("Computing rolling style matchup win rates...")
    df = add_style_winrates(df)

    print("Computing recent form features...")
    df = add_recent_form(df)

    print("Computing absorbed / defensive stats...")
    df = add_absorbed_stats(df)

    print("Computing Glicko-2 ratings...")
    df = add_glicko_ratings(df)

    print("Computing exponentially-weighted career stats...")
    df = add_ew_stats(df)

    print("Building doubled feature matrix...")
    features = build_feature_matrix(df)

    print("One-hot encoding styles...")
    features = add_style_onehot(features)

    print("Encoding weight class...")
    features = add_weight_class_ordinal(features)

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "matchup_features.csv")
    features.to_csv(out_path, index=False)

    n_rows = len(features)
    n_cols = len(features.columns)
    win_rate = features["a_won"].mean()
    print(f"\nSaved {n_rows:,} rows × {n_cols} columns → {out_path}")
    print(f"Win rate (a_won=1): {win_rate:.1%}  (expect ~50% after doubling)")
    print(f"Date range: {features['date'].min()} → {features['date'].max()}")
    print(f"Unique fighters: {features['a_fighter'].nunique():,}")
    print(f"\nFeature columns: {[c for c in features.columns if c not in ['fight_id','a_fighter','b_fighter','date','weight_class','a_won','a_style','b_style','style_matchup']]}")

    print_sanity_check(features)
    print("\nStep 2 complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/features.py data/raw/ufc-master.csv")
        sys.exit(1)
    main(sys.argv[1])
