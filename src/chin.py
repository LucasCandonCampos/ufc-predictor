"""
Fighter chin / KO-durability profiler and KO power ranker.

Chin metrics:
    kd_per_100          : knockdowns absorbed per 100 sig strikes absorbed (raw)
    kd_surplus_per_fight: actual KDs absorbed minus expected given each opponent's
                          offensive KD rate (Bayesian-shrunk).  Negative = absorbed
                          fewer KDs than the schedule warranted → better chin.
    ko_loss_rate        : fraction of losses by KO/TKO (Bayesian-shrunk)
    ko_loss_rate_adj    : ko_loss_rate / weight-class expected KO rate
    chin_score          : composite percentile [0, 1]; higher = tougher chin

KO Power metrics (build_ko_power_profiles):
    kd_per_100_off      : KDs scored per 100 sig strikes landed (raw / shrunk)
    ko_win_rate_adj     : KO wins / total wins, normalised by weight-class KO rate
    power_score         : composite percentile [0, 1]; higher = more dangerous finisher

Weight-class KO rates (UFC historical finish distribution):
    Heavyweight       51.8%
    Light Heavyweight 44.0%
    Middleweight      37.4%
    Welterweight      33.3%
    Lightweight       29.5%
    Featherweight     28.9%
    Bantamweight      25.5%
    Flyweight         23.6%

Usage:
    from chin import build_chin_profiles, build_ko_power_profiles
    chin_df  = build_chin_profiles()
    power_df = build_ko_power_profiles()
"""

import os
import logging
import pandas as pd

log = logging.getLogger(__name__)

ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(ROOT, "data")
ROUND_CSV = os.path.join(DATA_DIR, "round_stats.csv")

WEIGHT_CLASS_KO_RATE: dict[str, float] = {
    "Heavyweight":       0.518,
    "Light Heavyweight": 0.440,
    "Middleweight":      0.374,
    "Welterweight":      0.333,
    "Lightweight":       0.295,
    "Featherweight":     0.289,
    "Bantamweight":      0.255,
    "Flyweight":         0.236,
}
# Column groups used by features.py and predict.py for delta computation
CHIN_COLS  = ["chin_score", "kd_surplus_adj", "ko_loss_rate_adj"]
POWER_COLS = ["power_score", "kd_per_100_off", "ko_win_rate_adj"]

_POP_KO_RATE = 0.310   # fallback for catch-weight / unknown divisions

# Bayesian shrinkage constants
_KD_SHRINK_K  = 5    # fight-count prior for chin surplus & kd_per_100
_KO_SHRINK_K  = 3    # loss/win-count prior for ko rates
_OPP_SHRINK_K = 3    # fight-count prior for opponent offensive rates

_MIN_STRIKES_ABSORBED = 30
_MIN_STRIKES_LANDED   = 30


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _title(name: str) -> str:
    return " ".join(str(name).strip().split()).title()


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def _kd_stats_from_rounds(round_csv: str) -> pd.DataFrame:
    """
    Aggregate per-round KD / strike data to fighter level.

    Returns per-fighter:
        kd_absorbed, ssl_absorbed               — defensive (chin) columns
        kd_scored, ssl_landed, kd_per_100_off   — offensive (power) columns
        kd_surplus_per_fight                    — opponent-quality-adjusted chin metric
        kd_per_100_raw                          — raw KD absorption rate
        n_fights, last_fight_date
    """
    if not os.path.exists(round_csv):
        return pd.DataFrame(columns=[
            "fighter", "kd_absorbed", "ssl_absorbed",
            "kd_scored", "ssl_landed", "kd_per_100_raw", "kd_per_100_off",
            "kd_surplus_per_fight", "n_fights", "last_fight_date",
        ])

    rdf = pd.read_csv(round_csv)
    rdf["date"] = pd.to_datetime(rdf["date"], errors="coerce")
    for col in ("kd_0", "kd_1", "ssl_0", "ssl_1"):
        rdf[col] = pd.to_numeric(rdf[col], errors="coerce").fillna(0)

    # ── Per-fight stats for each corner ─────────────────────────────────────
    f0 = (
        rdf.groupby(["fight_url", "fighter_0"])
        .agg(
            kd_absorbed=("kd_1", "sum"), ssl_absorbed=("ssl_1", "sum"),
            kd_scored=("kd_0", "sum"),   ssl_landed=("ssl_0", "sum"),
            opp=("fighter_1", "first"),
        )
        .reset_index()
        .rename(columns={"fighter_0": "fighter"})
    )
    f1 = (
        rdf.groupby(["fight_url", "fighter_1"])
        .agg(
            kd_absorbed=("kd_0", "sum"), ssl_absorbed=("ssl_0", "sum"),
            kd_scored=("kd_1", "sum"),   ssl_landed=("ssl_1", "sum"),
            opp=("fighter_0", "first"),
        )
        .reset_index()
        .rename(columns={"fighter_1": "fighter"})
    )
    fight_level = pd.concat([f0, f1], ignore_index=True)

    # ── Offensive KD rate per fighter ────────────────────────────────────────
    off = (
        fight_level.groupby("fighter")
        .agg(
            kd_scored_total=("kd_scored", "sum"),
            ssl_landed_total=("ssl_landed", "sum"),
            n_fights_off=("fight_url", "nunique"),
        )
        .reset_index()
    )
    off["kd_per_100_off"] = (
        off["kd_scored_total"] / off["ssl_landed_total"].clip(lower=_MIN_STRIKES_LANDED) * 100
    ).round(4)

    # Bayesian-shrunk opponent rates used for expected-KD calculation.
    # Shrinking toward pool mean prevents small-sample outliers (a fighter who
    # KD'd someone once in 1 fight) from inflating another fighter's expected_kd.
    pool_mean = float(off["kd_per_100_off"].mean())
    n_off     = off["n_fights_off"]
    off["kd_per_100_off_shrunk"] = (
        (off["kd_per_100_off"] * n_off + pool_mean * _OPP_SHRINK_K) / (n_off + _OPP_SHRINK_K)
    )
    opp_kd_lookup = dict(zip(off["fighter"], off["kd_per_100_off_shrunk"]))

    # ── Per-fight KD surplus ─────────────────────────────────────────────────
    # surplus < 0 → fewer KDs absorbed than opponent power predicted → good chin
    # Opponents with no rate in the lookup fall back to pool mean so that
    # weight-class power differences propagate even for sparse opponents.
    fight_level["opp_kd_rate"] = (
        fight_level["opp"].map(opp_kd_lookup).fillna(pool_mean)
    )
    fight_level["expected_kd"] = fight_level["ssl_absorbed"] * fight_level["opp_kd_rate"] / 100
    fight_level["kd_surplus"]  = fight_level["kd_absorbed"] - fight_level["expected_kd"]

    # ── Fighter-level aggregates ─────────────────────────────────────────────
    fighter_kd = (
        fight_level.groupby("fighter")
        .agg(
            kd_absorbed=("kd_absorbed", "sum"),
            ssl_absorbed=("ssl_absorbed", "sum"),
            kd_scored=("kd_scored", "sum"),
            ssl_landed=("ssl_landed", "sum"),
            n_fights=("fight_url", "nunique"),
            kd_surplus_per_fight=("kd_surplus", "mean"),
        )
        .reset_index()
    )

    denom = fighter_kd["ssl_absorbed"].clip(lower=_MIN_STRIKES_ABSORBED)
    fighter_kd["kd_per_100_raw"] = (fighter_kd["kd_absorbed"] / denom * 100).round(4)

    # Attach raw offensive KD rate (will be shrunk further in build_*)
    fighter_kd = fighter_kd.merge(
        off[["fighter", "kd_per_100_off"]], on="fighter", how="left"
    )
    fighter_kd["kd_per_100_off"] = fighter_kd["kd_per_100_off"].fillna(0.0)

    # ── Last fight date ──────────────────────────────────────────────────────
    last0 = rdf.groupby("fighter_0")["date"].max().reset_index()
    last0.columns = ["fighter", "last_fight_date"]
    last1 = rdf.groupby("fighter_1")["date"].max().reset_index()
    last1.columns = ["fighter", "last_fight_date"]
    last_dates = (
        pd.concat([last0, last1], ignore_index=True)
        .groupby("fighter")["last_fight_date"].max()
        .reset_index()
    )
    fighter_kd = fighter_kd.merge(last_dates, on="fighter", how="left")
    return fighter_kd


def _ko_loss_stats_from_csv(csv_path: str) -> pd.DataFrame:
    """Per-fighter KO/TKO loss counts and most recent weight class."""
    df = pd.read_csv(
        csv_path,
        usecols=["R_fighter", "B_fighter", "Winner", "finish", "date", "weight_class"],
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    records = []
    for _, row in df.iterrows():
        finish = str(row.get("finish", "")).strip().upper()
        winner = str(row.get("Winner", "")).strip()
        is_ko  = finish == "KO/TKO"
        wc     = str(row.get("weight_class", "")).strip()
        date   = row.get("date")

        for corner in ("R", "B"):
            name = _title(str(row.get(f"{corner}_fighter", "")))
            if not name:
                continue
            lost = (winner in ("R", "Red")  and corner == "B") or \
                   (winner in ("B", "Blue") and corner == "R")
            records.append({
                "fighter":      name,
                "ko_loss":      int(lost and is_ko),
                "total_loss":   int(lost),
                "weight_class": wc,
                "date":         date,
            })

    if not records:
        return pd.DataFrame(columns=["fighter", "ko_losses", "total_losses", "weight_class"])

    rec_df = pd.DataFrame(records)
    agg = (
        rec_df.sort_values("date")
        .groupby("fighter")
        .agg(
            ko_losses=("ko_loss", "sum"),
            total_losses=("total_loss", "sum"),
            weight_class=("weight_class", "last"),
        )
        .reset_index()
    )
    agg["weight_class"] = agg["weight_class"].str.replace(r"^Women's\s+", "", regex=True)
    return agg


def _ko_win_stats_from_csv(csv_path: str) -> pd.DataFrame:
    """Per-fighter KO/TKO WIN counts and most recent weight class."""
    df = pd.read_csv(
        csv_path,
        usecols=["R_fighter", "B_fighter", "Winner", "finish", "date", "weight_class"],
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    records = []
    for _, row in df.iterrows():
        finish = str(row.get("finish", "")).strip().upper()
        winner = str(row.get("Winner", "")).strip()
        is_ko  = finish == "KO/TKO"
        wc     = str(row.get("weight_class", "")).strip()
        date   = row.get("date")

        for corner in ("R", "B"):
            name = _title(str(row.get(f"{corner}_fighter", "")))
            if not name:
                continue
            won = (winner in ("R", "Red")  and corner == "R") or \
                  (winner in ("B", "Blue") and corner == "B")
            records.append({
                "fighter":      name,
                "ko_win":       int(won and is_ko),
                "total_win":    int(won),
                "weight_class": wc,
                "date":         date,
            })

    if not records:
        return pd.DataFrame(columns=["fighter", "ko_wins", "total_wins", "weight_class"])

    rec_df = pd.DataFrame(records)
    agg = (
        rec_df.sort_values("date")
        .groupby("fighter")
        .agg(
            ko_wins=("ko_win", "sum"),
            total_wins=("total_win", "sum"),
            weight_class=("weight_class", "last"),
        )
        .reset_index()
    )
    agg["weight_class"] = agg["weight_class"].str.replace(r"^Women's\s+", "", regex=True)
    return agg


# ---------------------------------------------------------------------------
# Main builders
# ---------------------------------------------------------------------------

def build_chin_profiles(
    csv_path: str | None = None,
    min_fights: int = 5,
) -> pd.DataFrame:
    """
    Build chin / durability profiles for all fighters with enough data.

    Returns a DataFrame sorted by chin_score descending with columns:
        fighter, n_fights, last_fight_date,
        kd_absorbed, ssl_absorbed, kd_per_100_raw, kd_per_100,
        kd_surplus_per_fight, kd_surplus_adj,
        ko_losses, total_losses, weight_class, wc_ko_rate,
        ko_loss_rate_raw, ko_loss_rate, ko_loss_rate_adj,
        chin_score
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "raw", "ufc-master.csv")

    kd_df = _kd_stats_from_rounds(ROUND_CSV)
    ko_df = _ko_loss_stats_from_csv(csv_path)

    kd_df["_key"] = kd_df["fighter"].apply(_norm)
    ko_df["_key"] = ko_df["fighter"].apply(_norm)

    df = kd_df.merge(ko_df, on="_key", how="outer", suffixes=("", "_ko"))
    df["fighter"] = df["fighter"].combine_first(df["fighter_ko"])
    df.drop(columns=["fighter_ko", "_key"], inplace=True, errors="ignore")

    df["n_fights"]            = df["n_fights"].fillna(0).astype(int)
    df["kd_absorbed"]         = df["kd_absorbed"].fillna(0)
    df["ssl_absorbed"]        = df["ssl_absorbed"].fillna(0)
    df["kd_surplus_per_fight"]= df["kd_surplus_per_fight"].fillna(0.0)
    df["ko_losses"]           = df["ko_losses"].fillna(0)
    df["total_losses"]        = df["total_losses"].fillna(0)
    df["kd_per_100_raw"]      = df["kd_per_100_raw"].fillna(0.0)
    df["weight_class"]        = df["weight_class"].fillna("")

    df = df[df["n_fights"] >= min_fights].copy()

    n = df["n_fights"]

    # ── KD rate (informational) ──────────────────────────────────────────────
    pop_kd_median = df["kd_per_100_raw"].median()
    df["kd_per_100"] = (
        (df["kd_per_100_raw"] * n + pop_kd_median * _KD_SHRINK_K) / (n + _KD_SHRINK_K)
    ).round(4)

    # ── Opponent-quality-adjusted KD surplus ─────────────────────────────────
    # Bayesian shrinkage toward 0 (pool expectation: absorb exactly as many KDs
    # as the average opponent would inflict).  Fighters with few fights are pulled
    # toward 0 so small samples don't dominate the ranking.
    df["kd_surplus_adj"] = (
        df["kd_surplus_per_fight"] * n / (n + _KD_SHRINK_K)
    ).round(4)

    # ── KO loss rate ─────────────────────────────────────────────────────────
    df["ko_loss_rate_raw"] = (
        df["ko_losses"] / df["total_losses"].clip(lower=1)
    ).round(4)

    nl = df["total_losses"]
    df["ko_loss_rate"] = (
        (df["ko_losses"] + _POP_KO_RATE * _KO_SHRINK_K) / (nl + _KO_SHRINK_K)
    ).round(4)

    df["wc_ko_rate"]       = df["weight_class"].map(WEIGHT_CLASS_KO_RATE).fillna(_POP_KO_RATE)
    df["ko_loss_rate_adj"] = (df["ko_loss_rate"] / df["wc_ko_rate"]).round(4)

    # ── Composite chin score ─────────────────────────────────────────────────
    # Percentile ranks — inverted so lower surplus / lower adj KO loss = higher score.
    n_total = len(df)
    # kd_surplus_adj: more negative = better chin → lower rank value = better.
    kd_pct = df["kd_surplus_adj"].rank(method="average")     / n_total
    ko_pct = df["ko_loss_rate_adj"].rank(method="average")   / n_total

    df["chin_score"] = ((1 - kd_pct) * 0.55 + (1 - ko_pct) * 0.45).round(4)

    return df.sort_values("chin_score", ascending=False).reset_index(drop=True)


def build_ko_power_profiles(
    csv_path: str | None = None,
    min_fights: int = 5,
) -> pd.DataFrame:
    """
    Rank fighters by offensive KO / knockdown power.

    Returns a DataFrame sorted by power_score descending with columns:
        fighter, n_fights, last_fight_date, weight_class, wc_ko_rate,
        kd_per_100_off_raw, kd_per_100_off,
        ko_wins, total_wins, ko_win_rate, ko_win_rate_adj,
        power_score
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "raw", "ufc-master.csv")

    kd_df   = _kd_stats_from_rounds(ROUND_CSV)
    win_df  = _ko_win_stats_from_csv(csv_path)

    # Use offensive columns from round_stats
    off_df = kd_df[["fighter", "kd_per_100_off", "n_fights", "last_fight_date"]].copy()
    off_df.rename(columns={"kd_per_100_off": "kd_per_100_off_raw"}, inplace=True)

    off_df["_key"] = off_df["fighter"].apply(_norm)
    win_df["_key"] = win_df["fighter"].apply(_norm)

    df = off_df.merge(win_df, on="_key", how="outer", suffixes=("", "_ko"))
    df["fighter"] = df["fighter"].combine_first(df["fighter_ko"])
    df.drop(columns=[c for c in ["fighter_ko", "_key"] if c in df.columns], inplace=True)

    df["n_fights"]          = df["n_fights"].fillna(0).astype(int)
    df["kd_per_100_off_raw"]= df["kd_per_100_off_raw"].fillna(0.0)
    df["ko_wins"]           = df["ko_wins"].fillna(0)
    df["total_wins"]        = df["total_wins"].fillna(0)
    df["weight_class"]      = df["weight_class"].fillna("")

    df = df[df["n_fights"] >= min_fights].copy()

    n = df["n_fights"]

    # ── Offensive KD rate (Bayesian-shrunk) ──────────────────────────────────
    pool_off_mean = float(kd_df["kd_per_100_off"].mean())
    df["kd_per_100_off"] = (
        (df["kd_per_100_off_raw"] * n + pool_off_mean * _KD_SHRINK_K) / (n + _KD_SHRINK_K)
    ).round(4)

    # ── KO win rate ──────────────────────────────────────────────────────────
    nw = df["total_wins"]
    df["ko_win_rate"] = (
        (df["ko_wins"] + _POP_KO_RATE * _KO_SHRINK_K) / (nw + _KO_SHRINK_K)
    ).round(4)

    df["wc_ko_rate"]      = df["weight_class"].map(WEIGHT_CLASS_KO_RATE).fillna(_POP_KO_RATE)
    df["ko_win_rate_adj"] = (df["ko_win_rate"] / df["wc_ko_rate"]).round(4)

    # ── Composite power score ────────────────────────────────────────────────
    n_total = len(df)
    kd_pct = df["kd_per_100_off"].rank(method="average") / n_total  # higher = more dangerous
    ko_pct = df["ko_win_rate_adj"].rank(method="average") / n_total  # higher = more dangerous

    df["power_score"] = (kd_pct * 0.5 + ko_pct * 0.5).round(4)

    return df.sort_values("power_score", ascending=False).reset_index(drop=True)


def load_chin_power_profiles(
    min_fights: int = 3,
) -> tuple[dict[str, dict], dict, dict[str, dict], dict]:
    """
    Load chin and KO-power profiles and return lookup dicts for use in predict.py.

    Returns:
        chin_lookup    : {norm_name: {chin_score, kd_surplus_adj, ko_loss_rate_adj}}
        chin_defaults  : {col: population_median}  — used when fighter not found
        power_lookup   : {norm_name: {power_score, kd_per_100_off, ko_win_rate_adj}}
        power_defaults : {col: population_median}
    """
    chin_df  = build_chin_profiles(min_fights=min_fights)
    power_df = build_ko_power_profiles(min_fights=min_fights)

    chin_defaults: dict = {
        c: float(chin_df[c].median()) if c in chin_df.columns else 0.0
        for c in CHIN_COLS
    }
    power_defaults: dict = {
        c: float(power_df[c].median()) if c in power_df.columns else 0.0
        for c in POWER_COLS
    }

    chin_lookup: dict[str, dict] = {}
    for _, row in chin_df.iterrows():
        key = _norm(str(row.get("fighter", "")))
        if key:
            chin_lookup[key] = {c: float(row.get(c, chin_defaults[c])) for c in CHIN_COLS}

    power_lookup: dict[str, dict] = {}
    for _, row in power_df.iterrows():
        key = _norm(str(row.get("fighter", "")))
        if key:
            power_lookup[key] = {c: float(row.get(c, power_defaults[c])) for c in POWER_COLS}

    return chin_lookup, chin_defaults, power_lookup, power_defaults
