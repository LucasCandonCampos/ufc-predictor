"""
EV scanner: compares model fight predictions to Polymarket implied probabilities
and flags fights where the statistical edge exceeds a threshold.

A positive edge means the market is underpricing a fighter relative to the model —
typically driven by hype, nationality bias, or recency bias rather than statistics.

Usage:
    python src/ev_scanner.py "UFC 314"
    python src/ev_scanner.py            # broad search for any open UFC markets
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))
from polymarket import get_card_markets, fuzzy_match_name, cost_to_buy, SPORTS_FEE_RATE
from predict import predict_matchup, DATA_DIR

log = logging.getLogger(__name__)

EV_THRESHOLD   = 0.12   # minimum edge to flag as a +EV opportunity
KELLY_FRACTION = 0.25   # quarter-Kelly discount for model uncertainty

# Only recommend bets where the market prices our pick in this range.
# Pick'em to moderate favourite (+15% and +7% ROI respectively in backtests).
# Moderate underdogs (35-45%) and heavy dogs (<35%) showed -2.8% to -20.5% ROI.
BET_MKT_PROB_MIN = 0.45
BET_MKT_PROB_MAX = 0.70


def _bettable_odds(mkt_prob: float) -> bool:
    """Return True if the market probability of our pick is in the profitable range."""
    return BET_MKT_PROB_MIN <= mkt_prob <= BET_MKT_PROB_MAX


def _kelly_size(edge: float, market_prob: float) -> float:
    """
    Quarter-Kelly bet fraction.

    Full Kelly: f* = (b·p - q) / b
    where b = net decimal odds = (1 / market_prob) - 1
          p = model probability of winning
          q = 1 - p
    """
    if edge <= 0 or market_prob <= 0 or market_prob >= 1:
        return 0.0
    model_p = market_prob + edge
    if model_p >= 1.0:
        return 0.0
    b = (1.0 / market_prob) - 1.0
    q = 1.0 - model_p
    full_kelly = (b * model_p - q) / b
    return max(0.0, round(full_kelly * KELLY_FRACTION, 4))


def scan_card(
    event_query: str | None = None,
    ev_threshold: float = EV_THRESHOLD,
) -> list[dict]:
    """
    Scan an upcoming UFC card for +EV betting opportunities on Polymarket.

    Args:
        event_query: e.g. "UFC 314" or "UFC Fight Night". None searches broadly.
        ev_threshold: minimum |edge| to include in results (default 7%).

    Returns:
        List of dicts sorted by absolute edge descending. Each dict contains:
            fighter_a, fighter_b, our_name_a, our_name_b,
            model_prob_a, model_prob_b,
            market_prob_a, market_prob_b,
            edge_a, edge_b, kelly_a, kelly_b,
            style_a, style_b, question
    """
    fights = get_card_markets(event_query)
    if not fights:
        log.warning("No Polymarket fight markets found for '%s'", event_query)
        return []

    # Load known fighter names from the raw CSV for fuzzy matching
    csv_path = os.path.join(DATA_DIR, "raw", "ufc-master.csv")
    try:
        import pandas as pd
        raw = pd.read_csv(csv_path, usecols=["R_fighter", "B_fighter"])
        known_names = list(
            set(raw["R_fighter"].dropna().tolist() + raw["B_fighter"].dropna().tolist())
        )
    except Exception:
        known_names = []

    results = []
    for fight in fights:
        pm_a, pm_b = fight["fighter_a"], fight["fighter_b"]
        mkt_prob_a = fight["prob_a"]
        mkt_prob_b = fight["prob_b"]

        our_a = fuzzy_match_name(pm_a, known_names) or pm_a
        our_b = fuzzy_match_name(pm_b, known_names) or pm_b

        try:
            pred = predict_matchup(our_a, our_b, verbose=False)
        except (ValueError, Exception) as exc:
            log.warning("Skipping %s vs %s: %s", pm_a, pm_b, exc)
            continue

        model_prob_a = pred["prob_a"]
        model_prob_b = 1.0 - model_prob_a

        # Edge = model probability minus true cost to buy that outcome
        edge_a = round(model_prob_a - cost_to_buy(mkt_prob_a), 4)
        edge_b = round(model_prob_b - cost_to_buy(mkt_prob_b), 4)

        if max(abs(edge_a), abs(edge_b)) < ev_threshold:
            continue

        # bet_a/bet_b: True only when edge is positive AND market odds are in profitable range
        bet_a = edge_a >= ev_threshold and _bettable_odds(mkt_prob_a)
        bet_b = edge_b >= ev_threshold and _bettable_odds(mkt_prob_b)

        results.append({
            "fighter_a":     pm_a,
            "fighter_b":     pm_b,
            "our_name_a":    our_a,
            "our_name_b":    our_b,
            "model_prob_a":  round(model_prob_a, 4),
            "model_prob_b":  round(model_prob_b, 4),
            "market_prob_a": mkt_prob_a,
            "market_prob_b": mkt_prob_b,
            "edge_a":        edge_a,
            "edge_b":        edge_b,
            "kelly_a":       _kelly_size(edge_a, mkt_prob_a),
            "kelly_b":       _kelly_size(edge_b, mkt_prob_b),
            "bet_a":         bet_a,
            "bet_b":         bet_b,
            "style_a":       pred.get("style_a", ""),
            "style_b":       pred.get("style_b", ""),
            "weight_class":  pred.get("weight_class", ""),
            "question":      fight["question"],
        })

    results.sort(key=lambda r: max(abs(r["edge_a"]), abs(r["edge_b"])), reverse=True)
    return results


def format_scan_results(
    results: list[dict],
    event_query: str | None = None,
    total_scanned: int = 0,
) -> str:
    """Format EV scan results as a plain-text CLI report."""
    title  = event_query.upper() if event_query else "UFC"
    border = "=" * 56
    header = f"{border}\n  EV SCAN — {title}"
    if total_scanned:
        header += f"  ({total_scanned} fights scanned)"
    header += f"\n{border}"

    if not results:
        return header + "\n  No significant EV opportunities found.\n"

    lines = [header]
    for i, r in enumerate(results, 1):
        na, nb = r["fighter_a"], r["fighter_b"]
        ma, mb = r["model_prob_a"], r["model_prob_b"]
        pa, pb = r["market_prob_a"], r["market_prob_b"]
        ea, eb = r["edge_a"], r["edge_b"]
        ka, kb = r["kelly_a"], r["kelly_b"]

        lines.append(f"\n  #{i}  {na}  vs  {nb}")
        lines.append(f"  Model : {ma:.0%} / {mb:.0%}")
        lines.append(f"  Market: {pa:.0%} / {pb:.0%}")

        if r.get("bet_a"):
            lines.append(f"  [+EV] BET {na}   edge +{ea:.1%}   kelly {ka:.1%}")
        elif ea >= EV_THRESHOLD:
            lines.append(f"  [SKIP] {na}  edge +{ea:.1%} but mkt {pa:.0%} outside 45-70% range")
        if r.get("bet_b"):
            lines.append(f"  [+EV] BET {nb}   edge +{eb:.1%}   kelly {kb:.1%}")
        elif eb >= EV_THRESHOLD:
            lines.append(f"  [SKIP] {nb}  edge +{eb:.1%} but mkt {pb:.0%} outside 45-70% range")
        if ea <= -EV_THRESHOLD:
            lines.append(f"  [---] {na} OVERPRICED by {-ea:.1%}")
        if eb <= -EV_THRESHOLD:
            lines.append(f"  [---] {nb} OVERPRICED by {-eb:.1%}")

    lines += [
        f"\n{border}",
        f"  Kelly = 25% of full Kelly  |  Edge = model − Polymarket cost (3% fee)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s  %(levelname)s  %(message)s",
        level=logging.INFO,
    )
    query       = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    scan_results = scan_card(query)
    print(format_scan_results(scan_results, query, total_scanned=len(scan_results)))
