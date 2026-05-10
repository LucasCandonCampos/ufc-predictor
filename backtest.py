"""
Historical backtest: does the model have positive EV against sportsbook odds?

Methodology:
  - Use the time-based test set (last 20% of fights by date)
  - One perspective per fight: always "Red = a_fighter"
  - Model gives P(Red wins) — compare to de-vigged sportsbook implied prob
  - Simulate flat-bet and quarter-Kelly strategies
  - Report ROI, win rate, Brier score, calibration

No look-ahead: test set starts after 80th-percentile date used in training.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

FEATURES_CSV = os.path.join(ROOT, "data", "matchup_features.csv")
RAW_CSV      = os.path.join(ROOT, "data", "raw", "ufc-master.csv")
MODEL_PKL    = os.path.join(ROOT, "models", "model.pkl")

EV_THRESHOLD   = 0.05   # minimum edge to place a bet
KELLY_FRACTION = 0.25   # quarter-Kelly
BANKROLL_0     = 1000.0 # starting bankroll for Kelly simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_prob(odds: float) -> float:
    """American moneyline → raw implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def devig(prob_r: float, prob_b: float) -> tuple[float, float]:
    """Remove vig from two implied probabilities."""
    total = prob_r + prob_b
    if total <= 0:
        return 0.5, 0.5
    return prob_r / total, prob_b / total


def kelly_fraction(model_p: float, market_p: float) -> float:
    """Quarter-Kelly stake (as fraction of bankroll)."""
    if market_p <= 0 or market_p >= 1:
        return 0.0
    b = (1.0 / market_p) - 1.0   # decimal odds - 1
    q = 1.0 - model_p
    full_k = (b * model_p - q) / b
    return max(0.0, round(full_k * KELLY_FRACTION, 6))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading features and raw data...")
feat = pd.read_csv(FEATURES_CSV)
feat["date"] = pd.to_datetime(feat["date"])

raw = pd.read_csv(RAW_CSV)
raw["date"] = pd.to_datetime(raw["date"])

with open(MODEL_PKL, "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
feature_cols = bundle["feature_cols"]

print(f"  Features: {len(feat):,} rows  |  {len(feature_cols)} feature cols")
print(f"  Raw CSV:  {len(raw):,} fights")
print(f"  Model test accuracy (from training run): {bundle.get('test_accuracy', 'n/a'):.3f}")

# ---------------------------------------------------------------------------
# Determine test cutoff (same logic as train.py: 80th percentile of dates)
# ---------------------------------------------------------------------------

test_cut = feat["date"].quantile(0.80)
print(f"\nTest cutoff (80th pct): {test_cut.date()}")

feat_test = feat[feat["date"] > test_cut].copy()
print(f"Test rows (both perspectives): {len(feat_test):,}")

# ---------------------------------------------------------------------------
# Keep only the Red perspective so each fight appears once
# We identify Red rows by: a_fighter == R_fighter in raw CSV on the same date
# ---------------------------------------------------------------------------

raw_key = raw[["R_fighter", "B_fighter", "date", "R_odds", "B_odds", "Winner"]].copy()
raw_key = raw_key.dropna(subset=["R_odds", "B_odds"])
raw_key["date"] = pd.to_datetime(raw_key["date"])
raw_key["_winner_is_red"] = (raw_key["Winner"].str.strip().str.upper() == "RED").astype(int)

# Merge: a_fighter = R_fighter, b_fighter = B_fighter
merged = feat_test.merge(
    raw_key,
    left_on=["a_fighter", "b_fighter", "date"],
    right_on=["R_fighter", "B_fighter", "date"],
    how="inner",
)

print(f"After joining Red-perspective rows: {len(merged):,} fights")

# Sanity check: a_won should agree with winner_is_red
agree = (merged["a_won"] == merged["_winner_is_red"]).mean()
print(f"a_won ↔ actual winner agreement: {agree:.1%}  (should be ~100%)")

if agree < 0.90:
    print("WARNING: poor agreement — check fight matching logic")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Run model predictions on the test set
# ---------------------------------------------------------------------------

X_test = merged[feature_cols].copy()

# Fill any missing feature columns with 0
for col in feature_cols:
    if col not in X_test.columns:
        X_test[col] = 0.0
X_test = X_test.fillna(0.0)

merged["model_prob_r"] = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Convert sportsbook odds to de-vigged probabilities
# ---------------------------------------------------------------------------

merged["raw_imp_r"] = merged["R_odds"].apply(american_to_prob)
merged["raw_imp_b"] = merged["B_odds"].apply(american_to_prob)

devigged = merged.apply(
    lambda row: pd.Series(devig(row["raw_imp_r"], row["raw_imp_b"]), index=["mkt_prob_r", "mkt_prob_b"]),
    axis=1,
)
merged = pd.concat([merged, devigged], axis=1)

# Edge = model probability - fair market probability
merged["edge_r"] = merged["model_prob_r"] - merged["mkt_prob_r"]
merged["edge_b"] = (1.0 - merged["model_prob_r"]) - merged["mkt_prob_b"]

# Best side: whichever has larger positive edge
merged["best_edge"]    = merged[["edge_r", "edge_b"]].max(axis=1)
merged["bet_on_red"]   = merged["edge_r"] >= merged["edge_b"]

# ---------------------------------------------------------------------------
# Model calibration check
# ---------------------------------------------------------------------------

y_true  = merged["a_won"].values
y_prob  = merged["model_prob_r"].values

from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
acc   = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
auc   = roc_auc_score(y_true, y_prob)
brier = brier_score_loss(y_true, y_prob)
ll    = log_loss(y_true, y_prob)

print(f"\n{'─'*50}")
print(f"MODEL PERFORMANCE ON TEST SET ({len(merged):,} fights)")
print(f"{'─'*50}")
print(f"  Accuracy  : {acc:.1%}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Brier     : {brier:.4f}")
print(f"  Log Loss  : {ll:.4f}")
print(f"  Mean model prob R: {y_prob.mean():.3f}  (should be ~0.50 if balanced)")

# Odds sanity
print(f"\nOdds check (de-vigged):")
print(f"  Mean mkt_prob_r: {merged['mkt_prob_r'].mean():.3f}")
print(f"  Mean edge_r:     {merged['edge_r'].mean():.4f}")
print(f"  Mean |edge_r|:   {merged['edge_r'].abs().mean():.4f}")

# ---------------------------------------------------------------------------
# Betting simulation helpers
# ---------------------------------------------------------------------------

def simulate_flat(df: pd.DataFrame, ev_thresh: float = EV_THRESHOLD) -> dict:
    """
    Flat-bet $1 per fight on whichever side has edge > ev_thresh.
    Profit = (decimal_odds - 1) if win, else -1.
    """
    bets = df[df["best_edge"] >= ev_thresh].copy()
    if len(bets) == 0:
        return {"n_bets": 0, "roi": 0.0, "win_rate": 0.0, "total_wagered": 0.0, "profit": 0.0}

    # Determine which side we bet and the decimal odds for that side
    red_bets  = bets[bets["bet_on_red"]]
    blue_bets = bets[~bets["bet_on_red"]]

    def _pnl(row, is_red: bool) -> float:
        if is_red:
            won   = row["_winner_is_red"] == 1
            odds  = row["R_odds"]
        else:
            won   = row["_winner_is_red"] == 0
            odds  = row["B_odds"]
        if won:
            if odds > 0:
                return odds / 100.0
            else:
                return 100.0 / abs(odds)
        else:
            return -1.0

    red_pnl  = red_bets.apply(lambda r: _pnl(r, True),  axis=1)
    blue_pnl = blue_bets.apply(lambda r: _pnl(r, False), axis=1)
    all_pnl  = pd.concat([red_pnl, blue_pnl])

    total_wagered = len(bets)
    profit        = all_pnl.sum()
    wins          = (all_pnl > 0).sum()

    return {
        "n_bets":        len(bets),
        "n_red":         len(red_bets),
        "n_blue":        len(blue_bets),
        "win_rate":      wins / len(bets),
        "total_wagered": total_wagered,
        "profit":        profit,
        "roi":           profit / total_wagered,
        "avg_edge":      bets["best_edge"].mean(),
    }


def simulate_kelly(df: pd.DataFrame, ev_thresh: float = EV_THRESHOLD) -> dict:
    """Quarter-Kelly compounding simulation."""
    bets = df[df["best_edge"] >= ev_thresh].copy().sort_values("date")
    if len(bets) == 0:
        return {"n_bets": 0, "final_bankroll": BANKROLL_0, "roi": 0.0}

    bankroll = BANKROLL_0
    peak     = BANKROLL_0
    max_dd   = 0.0
    records  = []

    for _, row in bets.iterrows():
        is_red   = row["bet_on_red"]
        mkt_prob = row["mkt_prob_r"] if is_red else row["mkt_prob_b"]
        edge     = row["edge_r"]     if is_red else row["edge_b"]
        model_p  = row["model_prob_r"] if is_red else 1.0 - row["model_prob_r"]
        won      = (row["_winner_is_red"] == 1) if is_red else (row["_winner_is_red"] == 0)
        odds_am  = row["R_odds"] if is_red else row["B_odds"]

        frac  = kelly_fraction(model_p, mkt_prob)
        stake = bankroll * frac
        if stake < 0.01:
            continue

        if won:
            if odds_am > 0:
                pnl = stake * odds_am / 100.0
            else:
                pnl = stake * 100.0 / abs(odds_am)
        else:
            pnl = -stake

        bankroll += pnl
        bankroll  = max(bankroll, 0.01)  # floor at $0.01
        peak      = max(peak, bankroll)
        dd        = (peak - bankroll) / peak
        max_dd    = max(max_dd, dd)
        records.append({"bankroll": bankroll, "pnl": pnl, "won": won, "frac": frac})

    return {
        "n_bets":          len(records),
        "final_bankroll":  bankroll,
        "roi":             (bankroll - BANKROLL_0) / BANKROLL_0,
        "max_drawdown":    max_dd,
        "win_rate":        np.mean([r["won"] for r in records]),
        "avg_stake_pct":   np.mean([r["frac"] for r in records]),
    }


# ---------------------------------------------------------------------------
# Run simulations at several EV thresholds
# ---------------------------------------------------------------------------

print(f"\n{'─'*70}")
print(f"FLAT-BET SIMULATION  (test set: {merged['date'].min().date()} → {merged['date'].max().date()})")
print(f"Total test fights: {len(merged):,}")
print(f"{'─'*70}")
print(f"  {'EV%':>6}  {'Bets':>6}  {'Win%':>6}  {'ROI':>8}  {'Profit/$1':>10}  {'AvgEdge':>9}")
print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*9}")

for thresh in [0.03, 0.05, 0.07, 0.10, 0.15]:
    res = simulate_flat(merged, thresh)
    if res["n_bets"] == 0:
        print(f"  {thresh:>5.0%}  {'0':>6}  {'—':>6}  {'—':>8}  {'—':>10}  {'—':>9}")
    else:
        print(
            f"  {thresh:>5.0%}  {res['n_bets']:>6,}  {res['win_rate']:>6.1%}"
            f"  {res['roi']:>+8.1%}  {res['profit']:>+10.2f}  {res['avg_edge']:>9.3f}"
        )

print(f"\n{'─'*70}")
print(f"QUARTER-KELLY SIMULATION  (starting bankroll: ${BANKROLL_0:,.0f})")
print(f"{'─'*70}")
print(f"  {'EV%':>6}  {'Bets':>6}  {'Win%':>6}  {'Final $':>10}  {'ROI':>8}  {'MaxDD':>7}  {'AvgStk%':>8}")
print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*8}")

for thresh in [0.03, 0.05, 0.07, 0.10, 0.15]:
    res = simulate_kelly(merged, thresh)
    if res["n_bets"] == 0:
        print(f"  {thresh:>5.0%}  {'0':>6}  {'—':>6}  {'—':>10}  {'—':>8}  {'—':>7}  {'—':>8}")
    else:
        print(
            f"  {thresh:>5.0%}  {res['n_bets']:>6,}  {res['win_rate']:>6.1%}"
            f"  ${res['final_bankroll']:>9,.0f}  {res['roi']:>+8.1%}"
            f"  {res['max_drawdown']:>7.1%}  {res['avg_stake_pct']:>8.2%}"
        )

# ---------------------------------------------------------------------------
# Year-by-year breakdown (flat-bet at 5% threshold)
# ---------------------------------------------------------------------------

print(f"\n{'─'*60}")
print("YEAR-BY-YEAR FLAT-BET ROI  (EV threshold = 5%)")
print(f"{'─'*60}")
print(f"  {'Year':>6}  {'Fights':>7}  {'Bets':>6}  {'Win%':>6}  {'ROI':>8}")
print(f"  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}")

merged["year"] = merged["date"].dt.year
for yr, grp in merged.groupby("year"):
    res = simulate_flat(grp, 0.05)
    bets = res["n_bets"]
    if bets == 0:
        print(f"  {yr:>6}  {len(grp):>7,}  {'0':>6}  {'—':>6}  {'—':>8}")
    else:
        print(
            f"  {yr:>6}  {len(grp):>7,}  {bets:>6,}"
            f"  {res['win_rate']:>6.1%}  {res['roi']:>+8.1%}"
        )

# ---------------------------------------------------------------------------
# Probability distribution check
# ---------------------------------------------------------------------------

print(f"\n{'─'*50}")
print("EDGE DISTRIBUTION  (all test fights)")
print(f"{'─'*50}")
edges = merged["best_edge"]
for pct in [50, 75, 90, 95]:
    print(f"  {pct}th percentile edge: {edges.quantile(pct/100):+.3f}")

above_5 = (edges >= 0.05).mean()
above_10 = (edges >= 0.10).mean()
print(f"\n  Fights with edge ≥ 5%:  {above_5:.1%}")
print(f"  Fights with edge ≥ 10%: {above_10:.1%}")

print("\nDone.")
