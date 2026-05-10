"""
Polymarket API client for UFC fight markets.
No authentication required for read-only access.

API structure:
  GET /public-search?q=ufc returns {"events": [...], "pagination": {...}}
  Each event represents one fight and contains several markets (winner, rounds O/U, method).
  We extract only the fight-winner market (outcomes = two fighter names, not Yes/No/Over/Under).
"""

import re
import json
import logging
import requests
from rapidfuzz import process, fuzz

log = logging.getLogger(__name__)

GAMMA_BASE      = "https://gamma-api.polymarket.com"
SPORTS_FEE_RATE = 0.03   # Polymarket sports taker fee rate

_SKIP_OUTCOMES = {"yes", "no", "over", "under"}


def _parse_json_field(val) -> list:
    """Parse a field that may already be a list or a JSON-encoded string."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def search_ufc_events(query: str = "ufc", limit: int = 100) -> list[dict]:
    """
    Fetch UFC events from Polymarket Gamma API.

    Each event in the response represents one fight on a UFC card and
    contains several child markets (winner, rounds O/U, method, etc.).
    """
    url    = f"{GAMMA_BASE}/public-search"
    params = {"q": query, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.error("Polymarket API error: %s", exc)
        return []

    if isinstance(data, dict):
        return data.get("events", [])
    return []


def _find_winner_market(markets: list[dict]) -> dict | None:
    """
    Find the fight-winner market within an event's market list.

    The winner market is identified by having exactly two outcomes that are
    fighter names (not Yes/No/Over/Under) and not being closed/resolved.
    """
    for m in markets:
        if m.get("closed") or m.get("resolved"):
            continue
        outcomes = _parse_json_field(m.get("outcomes", []))
        if (
            len(outcomes) == 2
            and all(isinstance(o, str) for o in outcomes)
            and not any(o.lower() in _SKIP_OUTCOMES for o in outcomes)
        ):
            return m
    return None


def parse_fight_markets(events: list[dict]) -> list[dict]:
    """
    Extract fight-winner matchup info from a list of Polymarket event dicts.

    Returns list of dicts:
        fighter_a, fighter_b, prob_a, prob_b, market_id, question, event_title
    """
    fights = []
    for event in events:
        if event.get("closed") or event.get("archived"):
            continue

        winner_market = _find_winner_market(event.get("markets", []))
        if not winner_market:
            continue

        outcomes   = _parse_json_field(winner_market.get("outcomes", []))
        prices_raw = _parse_json_field(winner_market.get("outcomePrices", []))

        try:
            prices = [float(p) for p in prices_raw]
        except (TypeError, ValueError):
            continue

        if len(prices) < 2 or len(outcomes) < 2:
            continue

        fights.append({
            "fighter_a":   outcomes[0].strip(),
            "fighter_b":   outcomes[1].strip(),
            "prob_a":      round(prices[0], 4),
            "prob_b":      round(prices[1], 4),
            "market_id":   winner_market.get("id", ""),
            "question":    winner_market.get("question") or event.get("title", ""),
            "event_title": event.get("title", ""),
        })

    return fights


def get_card_markets(event_query: str | None = None) -> list[dict]:
    """
    High-level: search Polymarket for upcoming UFC fight markets.

    Args:
        event_query: e.g. "UFC 314" or "UFC Fight Night". Defaults to "ufc".
    """
    query  = event_query or "ufc"
    events = search_ufc_events(query=query, limit=100)
    fights = parse_fight_markets(events)
    log.info("Found %d active fight winner markets for query '%s'", len(fights), query)
    return fights


def fuzzy_match_name(
    name: str,
    known_names: list[str],
    threshold: int = 70,
) -> str | None:
    """
    Fuzzy-match a Polymarket fighter name to our database of known fighter names.
    Returns the best match or None if confidence falls below threshold.
    """
    if not known_names or not name:
        return None
    result = process.extractOne(name, known_names, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def cost_to_buy(prob: float, fee_rate: float = SPORTS_FEE_RATE) -> float:
    """
    True cost to acquire one outcome share on Polymarket, including taker fee.

    Polymarket sports fee: fee = rate × p × (1 - p)
    Total cost per share = p + fee
    """
    return prob + fee_rate * prob * (1.0 - prob)
