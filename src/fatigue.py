"""
Fighter fatigue profiler.

Tier 1 — proxy features (all historical fights, no new scraping):
    cardio_score              : avg(finish_round / no_of_rounds) across career
    late_finish_rate_for / against, early_finish_rate_for / against

Tier 2 — real round curves (from round_stats.csv built by round_scraper.py):
    output_drop_per_rd        : slope of sig_str_landed vs round (neg = fades)
    accuracy_drop_per_rd      : slope of sig_str_pct vs round
    td_def_drop_per_rd        : slope of (1 - opp_td_pct) vs round (neg = td def degrades)
    degradation_score         : composite [-1, 1]; positive = gets better late,
                                negative = fades badly. NaN when round data absent.

Usage:
    from fatigue import build_proxy_fatigue, load_fatigue_profiles
    profiles = build_proxy_fatigue(df)   # returns DataFrame, also writes fighter_fatigue.csv
    lookup   = load_fatigue_profiles()   # dict {normalised_name: {feature: value}}
    profiles = build_round_fatigue()     # round-data-only profiles (for /highcardio etc.)
"""

import os
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT         = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(ROOT, "data")
FATIGUE_CSV  = os.path.join(DATA_DIR, "fighter_fatigue.csv")
ROUND_CSV    = os.path.join(DATA_DIR, "round_stats.csv")

FATIGUE_COLS = [
    "cardio_score",
    "late_finish_rate_for",
    "late_finish_rate_against",
    "early_finish_rate_for",
    "early_finish_rate_against",
    "output_drop_per_rd",
    "accuracy_drop_per_rd",
    "td_def_drop_per_rd",
]

_DEFAULTS = {c: 0.0 for c in FATIGUE_COLS}
_DEFAULTS["cardio_score"] = 0.7   # sensible prior: most fights go 2+ rounds

# Columns present in round_stats.csv that drive Tier-2 degradation
_ROUND_DEGRADATION_COLS = ["output_drop_per_rd", "accuracy_drop_per_rd", "td_def_drop_per_rd"]

# New grappling-inclusive round-curve columns added to the model
ROUND_CURVE_COLS = [
    "activity_drop_per_rd",
    "champ_round_activity",
    "fade_score",
    "late_activity_abs",
]


def _norm(name: str) -> str:
    return str(name).strip().lower()


# ---------------------------------------------------------------------------
# Tier 1 — proxy features from existing CSV
# ---------------------------------------------------------------------------

def build_proxy_fatigue(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Compute per-fighter proxy fatigue features from the raw fight DataFrame.

    Returns a DataFrame with columns: fighter + FATIGUE_COLS.
    Also writes fighter_fatigue.csv if save=True.
    """
    finish_round = pd.to_numeric(df["finish_round"], errors="coerce").fillna(1.0)
    no_of_rounds = pd.to_numeric(df["no_of_rounds"], errors="coerce").fillna(3.0).clip(lower=1)
    finish       = df["finish"].fillna("U-DEC").str.strip()
    winner_col   = df.get("Winner", pd.Series([""] * len(df), index=df.index)).fillna("")

    is_finish = finish.str.contains(r"KO|TKO|SUB", case=False, na=False)
    round_ratio = (finish_round / no_of_rounds).clip(0.0, 1.0)

    records: list[dict] = []

    for corner in ("R", "B"):
        name_col = f"{corner}_fighter"
        if name_col not in df.columns:
            continue
        opp_corner = "B" if corner == "R" else "R"

        names  = df[name_col].fillna("")
        won    = winner_col.str.strip().str.upper() == corner

        rec = pd.DataFrame({
            "fighter":    names,
            "won":        won,
            "fin_round":  finish_round,
            "sched":      no_of_rounds,
            "is_finish":  is_finish,
            "rr":         round_ratio,
        })
        records.append(rec)

    all_fights = pd.concat(records, ignore_index=True)
    all_fights = all_fights[all_fights["fighter"].str.strip() != ""]

    def _profile(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        cardio = g["rr"].mean() if n > 0 else 0.7

        fin_won  = g[g["won"]  & g["is_finish"]]
        fin_lost = g[~g["won"] & g["is_finish"]]

        def _rate(subset: pd.DataFrame, lo: float, hi: float) -> float:
            if len(subset) == 0:
                return 0.0
            return ((subset["fin_round"] >= lo) & (subset["fin_round"] <= hi)).sum() / len(subset)

        return pd.Series({
            "cardio_score":              cardio,
            "late_finish_rate_for":      _rate(fin_won,  3.0, 99.0),
            "late_finish_rate_against":  _rate(fin_lost, 3.0, 99.0),
            "early_finish_rate_for":     _rate(fin_won,  1.0,  2.0),
            "early_finish_rate_against": _rate(fin_lost, 1.0,  2.0),
            # Tier-2 placeholders filled later from round_stats.csv
            "output_drop_per_rd":   0.0,
            "accuracy_drop_per_rd": 0.0,
        })

    profiles = (
        all_fights.groupby("fighter", sort=False)
        .apply(_profile)
        .reset_index()
    )

    # ---------------------------------------------------------------------------
    # Tier 2 — real round curves (if round_stats.csv exists and has data)
    # ---------------------------------------------------------------------------
    if os.path.exists(ROUND_CSV):
        try:
            rdf = pd.read_csv(ROUND_CSV)
            round_curves = _build_round_curves(rdf)
            profiles = profiles.merge(
                round_curves[["fighter", "output_drop_per_rd", "accuracy_drop_per_rd"]],
                on="fighter", how="left", suffixes=("", "_real"),
            )
            for col in ("output_drop_per_rd", "accuracy_drop_per_rd"):
                real = f"{col}_real"
                if real in profiles.columns:
                    profiles[col] = profiles[real].where(profiles[real].notna(), profiles[col])
                    profiles.drop(columns=[real], inplace=True)
        except Exception as exc:
            log.warning("Could not merge round curves: %s", exc)

    if save:
        profiles.to_csv(FATIGUE_CSV, index=False)
        log.info("Saved fatigue profiles for %d fighters → %s", len(profiles), FATIGUE_CSV)

    return profiles


# ---------------------------------------------------------------------------
# Tier 2 — real fatigue curves from per-round data
# ---------------------------------------------------------------------------

def _build_round_curves(rdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit per-round degradation slopes + output retention for each fighter.

    Metrics computed per fight, then averaged across fights per fighter:
        output_drop_per_rd   : slope of sig_str_landed vs round (neg = output fades)
        accuracy_drop_per_rd : slope of sig_str_pct vs round
        td_def_drop_per_rd   : slope of (1 - opp_td_pct) vs round
        output_retention     : avg(ssl rounds 3+) / avg(ssl rounds 1-2) — ratio of
                               late output to early output.  1.0 = consistent,
                               >1 = gets stronger, <1 = fades
        late_ssl_abs         : avg sig strikes landed in rounds 3+ (absolute volume)
        degradation_score    : composite of z(slope) + z(retention) + z(late volume),
                               scaled to [-1, 1].  Positive = elite cardio.

    The composite uses output_retention (50%) and late_ssl_abs (30%) so that
    fighters who maintain consistently high output — not just those whose output
    technically increases from a low baseline — rank at the top.
    """
    required = {"fight_url", "round", "fighter_0", "fighter_1",
                "ssl_0", "ssa_0", "ssl_1", "ssa_1"}
    if not required.issubset(rdf.columns):
        return pd.DataFrame(
            columns=["fighter"] + _ROUND_DEGRADATION_COLS
                    + ["output_retention", "late_ssl_abs",
                       "late_activity_abs", "activity_drop_per_rd",
                       "degradation_score"]
        )

    # Weights for the grappling-inclusive activity metric.
    # Each takedown ≈ 10 activity pts; 10 s of control ≈ 1 pt.
    # Baseline: ~17 ssl/rd + 0.5 td×10 + 67s/10 = ~29 activity/rd
    _TD_WEIGHT   = 10   # takedowns landed: meaningful grappling output
    _CTRL_DIVISOR = 10  # seconds of control → activity points

    # Pivot to one row per (fight, round, fighter)
    halves = []
    for idx in (0, 1):
        opp = 1 - idx
        has_ctrl = f"ctrl_{idx}" in rdf.columns
        has_tdl  = f"tdl_{idx}"  in rdf.columns
        sub = rdf[[
            "fight_url", "round",
            f"fighter_{idx}",
            f"ssl_{idx}", f"ssa_{idx}",
            *(([f"tdl_{idx}"])  if has_tdl  else []),
            *(([f"ctrl_{idx}"]) if has_ctrl else []),
            f"tdl_{opp}", f"tda_{opp}",
        ]].copy()
        cols = ["fight_url", "round", "fighter", "ssl", "ssa"]
        if has_tdl:  cols.append("tdl")
        if has_ctrl: cols.append("ctrl")
        cols += ["opp_tdl", "opp_tda"]
        sub.columns = cols
        if "tdl"  not in sub.columns: sub["tdl"]  = 0.0
        if "ctrl" not in sub.columns: sub["ctrl"] = 0.0
        halves.append(sub)

    long = pd.concat(halves, ignore_index=True)
    long = long[long["fighter"].notna() & (long["fighter"].str.strip() != "")]

    for col in ("round", "ssl", "ssa", "tdl", "ctrl", "opp_tdl", "opp_tda"):
        long[col] = pd.to_numeric(long[col], errors="coerce")

    long["ssl"]     = long["ssl"].fillna(0)
    long["ssa"]     = long["ssa"].fillna(0)
    long["tdl"]     = long["tdl"].fillna(0)
    long["ctrl"]    = long["ctrl"].fillna(0)
    long["opp_tdl"] = long["opp_tdl"].fillna(0)
    long["opp_tda"] = long["opp_tda"].fillna(0)
    long["acc"]     = np.where(long["ssa"] > 0, long["ssl"] / long["ssa"], np.nan)
    long["td_def"]  = np.where(
        long["opp_tda"] > 0,
        1.0 - long["opp_tdl"] / long["opp_tda"],
        np.nan,
    )
    # Combined output: strikes + takedown contribution + control-time contribution.
    # Captures full grappling-inclusive effort per round.
    long["activity"] = long["ssl"] + long["tdl"] * _TD_WEIGHT + long["ctrl"] / _CTRL_DIVISOR

    def _linfit(x: np.ndarray, y: np.ndarray) -> float:
        mask = ~np.isnan(y)
        xm, ym = x[mask], y[mask]
        if len(xm) < 2 or np.std(ym) == 0:
            return np.nan
        return float(np.polyfit(xm, ym, 1)[0])

    _all_metric_cols = _ROUND_DEGRADATION_COLS + [
        "output_retention", "late_ssl_abs",
        "late_activity_abs", "activity_drop_per_rd",
        "activity_fade", "champ_round_activity", "fight_max_round",
    ]

    def _fight_metrics(grp: pd.DataFrame) -> pd.Series:
        g = grp.dropna(subset=["round"]).sort_values("round")
        if len(g) < 2:
            return pd.Series({c: np.nan for c in _all_metric_cols})

        # Detect partial final rounds (fight finished mid-round via KO/sub).
        # If the last round's ssl is < 40% of the average ssl in all prior rounds
        # AND prior rounds have at least 5 ssl on average (guards low-volume grapplers),
        # that final round is a finishing sequence — exclude it so it doesn't
        # drag down late-round averages (e.g. Oliveira subs someone in rd 3 quickly).
        max_rd      = g["round"].max()
        prior       = g[g["round"] < max_rd]["ssl"]
        final_ssl   = g[g["round"] == max_rd]["ssl"].values
        if (len(prior) > 0 and len(final_ssl) > 0
                and prior.mean() > 5
                and final_ssl[0] < 0.40 * prior.mean()):
            g = g[g["round"] < max_rd]  # drop partial finish round
        if len(g) < 2:
            return pd.Series({c: np.nan for c in _all_metric_cols})

        x = g["round"].values.astype(float)

        early_ssl      = g[g["round"] <= 2]["ssl"].mean()
        late_ssl       = g[g["round"] >= 3]["ssl"].mean()
        early_activity = g[g["round"] <= 2]["activity"].mean()
        late_activity  = g[g["round"] >= 3]["activity"].mean()

        if pd.notna(early_ssl) and early_ssl > 0 and pd.notna(late_ssl):
            retention = late_ssl / early_ssl
        else:
            retention = np.nan

        # fade: how much does combined output DROP relative to the fighter's own early output?
        # 0 = no fade, 1 = went to zero. Only defined for fights that reach round 3+.
        # Clipped to [0, 1]: positive output growth (anti-fade) is treated as 0 fade.
        if (pd.notna(early_activity) and early_activity > 0
                and pd.notna(late_activity)):
            fade = float(np.clip(
                (early_activity - late_activity) / early_activity, 0.0, 1.0
            ))
        else:
            fade = np.nan

        # Championship rounds: activity in rounds 4+ (only 5-round fights reach these).
        # A fighter who maintains high output in rd 4-5 gets credit here that
        # fighters who only fight 3-round prelims can never accumulate.
        champ_activity = g[g["round"] >= 4]["activity"].mean()  # NaN if fight < 4 rounds

        return pd.Series({
            "output_drop_per_rd":   _linfit(x, g["ssl"].values.astype(float)),
            "accuracy_drop_per_rd": _linfit(x, g["acc"].values),
            "td_def_drop_per_rd":   _linfit(x, g["td_def"].values),
            "output_retention":     retention,
            "late_ssl_abs":         late_ssl if pd.notna(late_ssl) else np.nan,
            "late_activity_abs":    late_activity if pd.notna(late_activity) else np.nan,
            "activity_drop_per_rd": _linfit(x, g["activity"].values.astype(float)),
            "activity_fade":        fade,
            "champ_round_activity": champ_activity if pd.notna(champ_activity) else np.nan,
            "fight_max_round":      float(g["round"].max()),
        })

    fight_metrics = (
        long.groupby(["fighter", "fight_url"], sort=False)
        .apply(_fight_metrics)
        .reset_index()
    )

    fighter_curves = (
        fight_metrics.groupby("fighter")[_all_metric_cols]
        .agg(lambda s: s.dropna().mean() if s.dropna().count() >= 2 else np.nan)
        .reset_index()
    )

    # Composite cardio score (best cardio / degradation_score).
    # Uses grappling-inclusive activity and percentile ranks (outlier-robust).
    #   late_activity_abs    (50%): combined output in rounds 3+
    #   activity_drop_per_rd (30%): slope — getting stronger or weaker per round
    #   champ_round_activity (20%): avg activity in rounds 4-5 only; NaN → median fill.
    #                               fighters who never reach rd 4 score at the median,
    #                               not penalised — but those who DO go deep and perform
    #                               well get a meaningful bonus.
    n = len(fighter_curves)
    _COMPOSITE_PCT = {
        "late_activity_abs":    0.50,
        "activity_drop_per_rd": 0.30,
        "champ_round_activity": 0.20,
    }

    for col, _w in _COMPOSITE_PCT.items():
        filled = fighter_curves[col].fillna(fighter_curves[col].median())
        ranks  = filled.rank(method="average", na_option="bottom")
        fighter_curves[f"_pct_{col}"] = ranks / n

    fighter_curves["degradation_score"] = sum(
        fighter_curves[f"_pct_{col}"] * w for col, w in _COMPOSITE_PCT.items()
    )
    fighter_curves["degradation_score"] = (
        (fighter_curves["degradation_score"] * 2 - 1).round(4)
    )

    # fade_score: length-adjusted proportional drop in combined output.
    # Raw fade is normalised by fight length so that fading 30% over 5 rounds
    # is treated as equivalent to fading 18% over 3 rounds (30% × 3/5).
    # avg_fight_rounds defaults to 3 for fighters without that data.
    avg_rounds = fighter_curves["fight_max_round"].fillna(3.0).clip(lower=3.0)
    raw_fade   = fighter_curves["activity_fade"].fillna(0.0)
    fighter_curves["fade_score"] = (raw_fade * (3.0 / avg_rounds)).round(4)

    fighter_curves.drop(
        columns=["activity_fade", "fight_max_round"]
                + [f"_pct_{c}" for c in _COMPOSITE_PCT],
        inplace=True,
    )
    return fighter_curves


# ---------------------------------------------------------------------------
# Public: build round-only profiles (for bot cardio commands)
# ---------------------------------------------------------------------------

def build_round_fatigue(min_fights: int = 3) -> pd.DataFrame:
    """
    Load round_stats.csv and return per-fighter degradation profiles.

    Returns a DataFrame with columns:
        fighter, n_fights,
        output_drop_per_rd, accuracy_drop_per_rd, td_def_drop_per_rd,
        output_retention, late_ssl_abs,
        late_activity_abs, activity_drop_per_rd, champ_round_activity,
        fade_score, degradation_score

    Only includes fighters with at least `min_fights` multi-round fights.
    Returns an empty DataFrame if round_stats.csv does not exist.
    """
    if not os.path.exists(ROUND_CSV):
        return pd.DataFrame(columns=[
            "fighter", "n_fights",
            "output_drop_per_rd", "accuracy_drop_per_rd", "td_def_drop_per_rd",
            "output_retention", "late_ssl_abs",
            "late_activity_abs", "activity_drop_per_rd", "champ_round_activity",
            "fade_score", "degradation_score",
        ])

    rdf = pd.read_csv(ROUND_CSV)
    rdf["date"] = pd.to_datetime(rdf["date"], errors="coerce")
    curves = _build_round_curves(rdf)

    # Count distinct fight_urls per fighter across both sides
    f0 = rdf.groupby("fighter_0")["fight_url"].nunique().rename("n_fights")
    f1 = rdf.groupby("fighter_1")["fight_url"].nunique().rename("n_fights")
    n_fights = (f0.add(f1, fill_value=0)).reset_index()
    n_fights.columns = ["fighter", "n_fights"]

    # Last fight date per fighter
    last0 = rdf.groupby("fighter_0")["date"].max().rename("last_fight_date").reset_index()
    last0.columns = ["fighter", "last_fight_date"]
    last1 = rdf.groupby("fighter_1")["date"].max().rename("last_fight_date").reset_index()
    last1.columns = ["fighter", "last_fight_date"]
    last_fight = (
        pd.concat([last0, last1], ignore_index=True)
        .groupby("fighter")["last_fight_date"]
        .max()
        .reset_index()
    )

    curves = curves.merge(n_fights, on="fighter", how="left")
    curves = curves.merge(last_fight, on="fighter", how="left")
    curves["n_fights"] = curves["n_fights"].fillna(0).astype(int)

    # Filter to fighters with enough data
    curves = curves[curves["n_fights"] >= min_fights].copy()
    curves = curves.dropna(subset=["degradation_score"])

    return curves.sort_values("degradation_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lookup helpers (used by predict.py and features.py)
# ---------------------------------------------------------------------------

def load_fatigue_profiles() -> dict[str, dict]:
    """
    Load fighter_fatigue.csv and return a dict keyed by normalised fighter name.
    Falls back to _DEFAULTS for unknown fighters.
    """
    if not os.path.exists(FATIGUE_CSV):
        return {}
    df = pd.read_csv(FATIGUE_CSV)
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = _norm(str(row.get("fighter", "")))
        if name:
            result[name] = {c: float(row.get(c, _DEFAULTS.get(c, 0.0))) for c in FATIGUE_COLS}
    return result


def lookup_fighter_fatigue(name: str, profiles: dict[str, dict]) -> dict:
    """Return fatigue profile for a fighter, falling back to defaults."""
    key = _norm(name)
    return profiles.get(key, dict(_DEFAULTS))


def load_round_curve_profiles(min_fights: int = 2) -> tuple[dict[str, dict], dict]:
    """
    Build per-fighter round-curve profiles from round_stats.csv and return:
        (lookup dict keyed by normalised name, defaults dict)

    Uses median fills for absolute metrics (champ_round_activity, late_activity_abs)
    so fighters without championship-round data score neutrally.
    Returns an empty lookup + zero defaults when round_stats.csv is absent.
    """
    curves = build_round_fatigue(min_fights=min_fights)

    zero_defaults: dict = {c: 0.0 for c in ROUND_CURVE_COLS}
    if curves.empty:
        return {}, zero_defaults

    champ_median = float(curves["champ_round_activity"].dropna().median() or 0.0)
    late_median  = float(curves["late_activity_abs"].dropna().median() or 0.0)

    curves = curves.copy()
    curves["champ_round_activity"] = curves["champ_round_activity"].fillna(champ_median)
    curves["late_activity_abs"]    = curves["late_activity_abs"].fillna(late_median)

    defaults: dict = {c: 0.0 for c in ROUND_CURVE_COLS}
    defaults["champ_round_activity"] = champ_median
    defaults["late_activity_abs"]    = late_median

    lookup: dict[str, dict] = {}
    for _, row in curves.iterrows():
        name = _norm(str(row.get("fighter", "")))
        if name:
            lookup[name] = {c: float(row.get(c, defaults[c])) for c in ROUND_CURVE_COLS}

    return lookup, defaults
