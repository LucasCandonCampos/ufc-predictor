"""
Scrape new UFC events from UFCStats.com and append to ufc-master.csv.

Usage:
    python src/scraper.py                # append new fights
    python src/scraper.py --dry-run      # preview without writing

After running, retrain the model:
    python src/cluster.py data/raw/ufc-master.csv
    python src/features.py data/raw/ufc-master.csv
    python src/train.py
"""

import os
import sys
import re
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, date
from bs4 import BeautifulSoup

BASE       = "http://ufcstats.com"
EVENTS_URL = f"{BASE}/statistics/events/completed?page=all"
DELAY      = 1.5   # seconds between requests (be polite)

ROOT     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "data", "raw", "ufc-master.csv")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str) -> BeautifulSoup:
    headers = {"User-Agent": "Mozilla/5.0 (ufc-updater/1.0; research only)"}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            time.sleep(DELAY)
            return BeautifulSoup(r.text, "html.parser")
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
            time.sleep(DELAY * 3)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _x_of_y(s: str) -> tuple[int, int]:
    """'15 of 30' → (15, 30); '---' or '' → (0, 0)."""
    m = re.match(r"(\d+)\s+of\s+(\d+)", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _parse_time_str(s: str) -> float:
    """'3:17' → 3 + 17/60 minutes."""
    m = re.match(r"(\d+):(\d+)", s.strip())
    return (int(m.group(1)) + int(m.group(2)) / 60.0) if m else 0.0


def _parse_ctrl_secs(s: str) -> int:
    """'4:13' → 253 seconds; '--' or '' → 0."""
    m = re.match(r"(\d+):(\d+)", s.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def _split_cell(cell) -> tuple[str, str]:
    """Split a two-fighter cell on '|' → (val_fighter_0, val_fighter_1)."""
    parts = [p.strip() for p in cell.get_text(separator="|", strip=True).split("|") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return (parts[0], parts[0]) if parts else ("", "")


def _method_to_finish(method: str) -> str:
    """Map UFCStats method text to the finish codes used in ufc-master.csv."""
    m = method.strip()
    if "Doctor" in m:                       return "TKO - Doctor's Stoppage"
    if "KO/TKO" in m or "KO" in m:         return "KO/TKO"
    if "Submission" in m or "Sub" in m:     return "SUB"
    if "Unanimous" in m or "U-DEC" in m:   return "U-DEC"
    if "Split" in m     or "S-DEC" in m:   return "S-DEC"
    if "Majority" in m  or "M-DEC" in m:   return "M-DEC"
    if "DQ" in m or "Disqualification" in m: return "DQ"
    if "No Contest" in m or "NC" in m:     return "NC"
    if "Decision" in m:                    return "U-DEC"  # fallback
    return m


# ---------------------------------------------------------------------------
# Step 1 — events list
# ---------------------------------------------------------------------------

def fetch_new_events(after: date) -> list[dict]:
    """Return events strictly after `after`, sorted oldest-first."""
    soup = _get(EVENTS_URL)
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        link     = row.select_one("a.b-link")
        date_el  = row.select_one("span.b-statistics__date")
        loc_cell = row.select("td")
        if not link or not date_el:
            continue
        try:
            event_date = datetime.strptime(date_el.get_text(strip=True), "%B %d, %Y").date()
        except ValueError:
            continue
        if event_date > after:
            location = loc_cell[1].get_text(strip=True) if len(loc_cell) > 1 else ""
            events.append({
                "name":     link.get_text(strip=True),
                "date":     event_date,
                "url":      link["href"],
                "location": location,
            })
    return sorted(events, key=lambda x: x["date"])


# ---------------------------------------------------------------------------
# Step 2 — fight list for one event
# ---------------------------------------------------------------------------

def fetch_event_fights(event_url: str) -> list[dict]:
    """Return fight metadata from an event detail page."""
    soup   = _get(event_url)
    fights = []

    for row in soup.select("tr.b-fight-details__table-row__hover"):
        fight_url = row.get("data-link", "").strip()
        if not fight_url:
            continue
        cells = row.select("td")
        if len(cells) < 10:
            continue

        # Fighter names (cells[1] = two <p> tags; first = winner)
        names = [a.get_text(strip=True) for a in cells[1].select("a.b-link")]
        if len(names) < 2:
            continue

        # W/L flags — winner is first fighter listed (green flag)
        flags = [i.get_text(strip=True).lower() for i in cells[0].select("i.b-flag__text")]
        winner_idx = 0  # default: first fighter won
        if flags:
            # Check if "draw" or "nc" appears
            if all(f in ("draw", "nc", "no contest") for f in flags[:2]):
                winner_idx = -1  # draw / NC
            elif len(flags) >= 2 and flags[0] != "win":
                winner_idx = 1

        method       = cells[7].get_text(strip=True)
        finish_round = cells[8].get_text(strip=True)
        finish_time  = cells[9].get_text(strip=True)
        weight_class = cells[6].get_text(strip=True)

        fights.append({
            "fight_url":    fight_url,
            "fighter_0":    names[0],   # first listed (winner in almost all cases)
            "fighter_1":    names[1],
            "winner_idx":   winner_idx, # 0 = fighter_0 won, 1 = fighter_1 won, -1 = draw/NC
            "weight_class": weight_class,
            "method":       method,
            "finish_round": int(finish_round) if finish_round.isdigit() else 1,
            "finish_time":  finish_time,
        })

    return fights


# ---------------------------------------------------------------------------
# Step 3 — per-fight stats from fight detail page
# ---------------------------------------------------------------------------

def fetch_fight_stats(fight_url: str) -> dict | None:
    """
    Scrape the fight detail page. Returns:
    {
      "fighter_0": name, "fighter_1": name,
      "winner_name": str | None,
      "method": str, "no_of_rounds": int,
      "total_secs": int, "finish_round": int, "finish_time": str,
      "title_bout": bool, "finish_details": str,
      "stats_0": {sig_str_landed, sig_str_att, td_landed, td_att, sub_att, kd},
      "stats_1": {...},
    }
    Returns None if the page cannot be parsed.
    """
    soup = _get(fight_url)

    # Fighter names and W/L from the person divs
    persons = soup.select("div.b-fight-details__person")
    if len(persons) < 2:
        return None

    def _person_name(div) -> str:
        el = div.select_one("h3.b-fight-details__person-name a") or \
             div.select_one("h3.b-fight-details__person-name")
        return el.get_text(strip=True) if el else ""

    def _person_status(div) -> str:
        el = div.select_one("i.b-fight-details__person-status")
        return el.get_text(strip=True).upper() if el else ""

    name_0   = _person_name(persons[0])
    name_1   = _person_name(persons[1])
    status_0 = _person_status(persons[0])   # "W", "L", "D", "NC"

    if status_0 == "W":
        winner_name = name_0
    elif status_0 == "L":
        winner_name = name_1
    else:
        winner_name = None  # draw / NC

    # Fight metadata from the content block
    detail_text = soup.select_one("div.b-fight-details__content")
    method, finish_round, finish_time_str, no_of_rounds = "", 1, "0:00", 3
    title_bout, finish_details = False, ""
    if detail_text:
        txt = detail_text.get_text(separator="|", strip=True)
        m = re.search(r"Method:\|([^|]+)", txt);         method         = m.group(1).strip() if m else ""
        m = re.search(r"Round:\|(\d+)", txt);            finish_round   = int(m.group(1)) if m else 1
        m = re.search(r"Time:\|(\d+:\d+)", txt);         finish_time_str= m.group(1) if m else "0:00"
        m = re.search(r"Time format:\|(\d+)", txt);      no_of_rounds   = int(m.group(1)) if m else 3
        m = re.search(r"Details:\|([^|]+)", txt);        finish_details = m.group(1).strip() if m else ""

    # Total fight time in seconds
    finish_min  = _parse_time_str(finish_time_str)
    total_secs  = int(((finish_round - 1) * 5 + finish_min) * 60)

    # Sum per-round stats from Table 0 (Totals section)
    tables = soup.select("table.b-fight-details__table")
    if not tables:
        return None

    agg = {"sig_str_landed": [0, 0], "sig_str_att":  [0, 0],
           "td_landed":      [0, 0], "td_att":       [0, 0],
           "sub_att":        [0, 0], "kd":           [0, 0],
           "ctrl_secs":      [0, 0],
           "head_landed":    [0, 0], "body_landed":  [0, 0], "leg_landed": [0, 0]}

    # Per-round data — one entry per round in table order
    per_round: list[dict] = []
    rd_idx = 0

    for row in tables[0].select("tr"):
        cells = row.select("td")
        if not cells or len(cells) < 8:
            continue
        # Cell index:  0=names, 1=KD, 2=sig_str(x/y), 3=sig_str%, 4=total_str,
        #              5=TD(x/y), 6=TD%, 7=sub_att, 8=rev, 9=ctrl
        v0, v1 = _split_cell(cells[1])
        kd0 = int(v0) if v0.isdigit() else 0
        kd1 = int(v1) if v1.isdigit() else 0
        agg["kd"][0] += kd0
        agg["kd"][1] += kd1

        s0, s1 = _split_cell(cells[2])
        l0, a0 = _x_of_y(s0);  l1, a1 = _x_of_y(s1)
        agg["sig_str_landed"][0] += l0;  agg["sig_str_att"][0] += a0
        agg["sig_str_landed"][1] += l1;  agg["sig_str_att"][1] += a1

        t0, t1 = _split_cell(cells[5])
        tl0, ta0 = _x_of_y(t0);  tl1, ta1 = _x_of_y(t1)
        agg["td_landed"][0] += tl0;  agg["td_att"][0] += ta0
        agg["td_landed"][1] += tl1;  agg["td_att"][1] += ta1

        u0, u1 = _split_cell(cells[7])
        sa0 = int(u0) if u0.isdigit() else 0
        sa1 = int(u1) if u1.isdigit() else 0
        agg["sub_att"][0] += sa0
        agg["sub_att"][1] += sa1

        ctrl0 = ctrl1 = 0
        if len(cells) > 9:
            c0, c1 = _split_cell(cells[9])
            ctrl0 = _parse_ctrl_secs(c0)
            ctrl1 = _parse_ctrl_secs(c1)
            agg["ctrl_secs"][0] += ctrl0
            agg["ctrl_secs"][1] += ctrl1

        rd_idx += 1
        per_round.append({
            "round":   rd_idx,
            "ssl_0":   l0,   "ssa_0":  a0,
            "tdl_0":   tl0,  "tda_0":  ta0,
            "kd_0":    kd0,  "ctrl_0": ctrl0,
            "ssl_1":   l1,   "ssa_1":  a1,
            "tdl_1":   tl1,  "tda_1":  ta1,
            "kd_1":    kd1,  "ctrl_1": ctrl1,
        })

    # Table 1: Significant Strikes breakdown (head / body / leg)
    # Columns: 0=names, 1=sig_str(x/y), 2=head(x/y), 3=body(x/y), 4=leg(x/y), ...
    if len(tables) > 1:
        for row in tables[1].select("tr"):
            cells = row.select("td")
            if not cells or len(cells) < 5:
                continue
            h0, h1 = _split_cell(cells[2]); hl0, _ = _x_of_y(h0); hl1, _ = _x_of_y(h1)
            b0, b1 = _split_cell(cells[3]); bl0, _ = _x_of_y(b0); bl1, _ = _x_of_y(b1)
            g0, g1 = _split_cell(cells[4]); gl0, _ = _x_of_y(g0); gl1, _ = _x_of_y(g1)
            agg["head_landed"][0] += hl0;  agg["head_landed"][1] += hl1
            agg["body_landed"][0] += bl0;  agg["body_landed"][1] += bl1
            agg["leg_landed"][0]  += gl0;  agg["leg_landed"][1]  += gl1

    def _stats(idx: int) -> dict:
        sl = agg["sig_str_landed"][idx];  sa = agg["sig_str_att"][idx]
        tl = agg["td_landed"][idx];       ta = agg["td_att"][idx]
        return {
            "sig_str_landed": sl,
            "sig_str_att":    sa,
            "sig_str_pct":    sl / sa if sa > 0 else 0.0,
            "td_landed":      tl,
            "td_att":         ta,
            "td_pct":         tl / ta if ta > 0 else 0.0,
            "sub_att":        agg["sub_att"][idx],
            "kd":             agg["kd"][idx],
            "ctrl_secs":      agg["ctrl_secs"][idx],
            "head_landed":    agg["head_landed"][idx],
            "body_landed":    agg["body_landed"][idx],
            "leg_landed":     agg["leg_landed"][idx],
        }

    return {
        "fighter_0":     name_0,
        "fighter_1":     name_1,
        "winner_name":   winner_name,
        "method":        method,
        "no_of_rounds":  no_of_rounds,
        "total_secs":    total_secs,
        "finish_round":  finish_round,
        "finish_time":   finish_time_str,
        "title_bout":    title_bout,
        "finish_details": finish_details,
        "stats_0":       _stats(0),
        "stats_1":       _stats(1),
        "per_round":     per_round,   # list of dicts, one per round; empty if only totals row
    }


# ---------------------------------------------------------------------------
# Step 4 — update cumulative career stats
# ---------------------------------------------------------------------------

def _get_latest_fighter_row(name: str, df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Return (most_recent_row, corner) for a fighter in the existing CSV, or (None, '')."""
    name_l = name.strip().lower()
    r_mask = df["R_fighter"].str.strip().str.lower() == name_l
    b_mask = df["B_fighter"].str.strip().str.lower() == name_l
    subset = pd.concat([
        df[r_mask].assign(_c="R"),
        df[b_mask].assign(_c="B"),
    ]).sort_values("date")
    if subset.empty:
        return None, ""
    row = subset.iloc[-1]
    return row, row["_c"]


def _safe(row, col, default=0.0):
    """Get a numeric value from a row dict/Series, falling back to default."""
    v = row.get(col, default) if isinstance(row, dict) else row.get(col, default)
    try:
        f = float(v)
        return f if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def update_career_stats(
    old_row: pd.Series | None,
    corner: str,
    fight_stats: dict,
    fight_time_secs: int,
    finish_rounds: int,
    won: bool,
    is_draw: bool,
    method: str,
    prefix: str,
) -> dict:
    """
    Compute updated cumulative stats for one fighter after a new fight.
    `prefix` is "R_" or "B_" for the output column names.
    Returns a dict of column→value for all stats we track.
    """
    p = prefix  # shorter alias

    def g(col, default=0.0):
        if old_row is None:
            return default
        return _safe(old_row, f"{corner}_{col}", default)

    # Existing record counts
    n_wins   = int(g("wins"))
    n_losses = int(g("losses"))
    n_draws  = int(g("draw"))
    n_fights = n_wins + n_losses + n_draws

    # Time-based update for per-15-min stats
    old_rounds   = int(g("total_rounds_fought"))
    old_time_min = old_rounds * 5.0                   # approximate (each round = 5 min)
    new_time_min = fight_time_secs / 60.0
    total_time   = old_time_min + new_time_min

    def upd_per15(old_avg: float, new_count: float) -> float:
        if total_time <= 0:
            return new_count * (15 / max(new_time_min, 1))
        return (old_avg * old_time_min + new_count * 15) / total_time

    def upd_pct(old_pct: float, new_pct: float) -> float:
        if n_fights <= 0:
            return new_pct
        return (old_pct * n_fights + new_pct) / (n_fights + 1)

    # Update averages
    fs = fight_stats
    new_sig_landed  = upd_per15(g("avg_SIG_STR_landed"), fs["sig_str_landed"])
    new_sig_pct     = upd_pct(  g("avg_SIG_STR_pct"),   fs["sig_str_pct"])
    new_td_landed   = upd_per15(g("avg_TD_landed"),      fs["td_landed"])
    new_td_pct      = upd_pct(  g("avg_TD_pct"),         fs["td_pct"])
    new_sub_att     = upd_per15(g("avg_SUB_ATT"),        fs["sub_att"])
    new_avg_kd      = upd_per15(g("avg_KD",          0.0), fs["kd"])
    new_avg_ctrl    = upd_per15(g("avg_ctrl_secs",   0.0), fs["ctrl_secs"])
    new_avg_head    = upd_per15(g("avg_head_landed", 0.0), fs["head_landed"])
    new_avg_body    = upd_per15(g("avg_body_landed", 0.0), fs["body_landed"])
    new_avg_leg     = upd_per15(g("avg_leg_landed",  0.0), fs["leg_landed"])

    # Win/loss/draw counts
    if is_draw:
        new_wins, new_losses, new_draws = n_wins, n_losses, n_draws + 1
    elif won:
        new_wins, new_losses, new_draws = n_wins + 1, n_losses, n_draws
    else:
        new_wins, new_losses, new_draws = n_wins, n_losses + 1, n_draws

    # Win method tallies
    w_ko      = int(g("win_by_KO/TKO"))
    w_sub     = int(g("win_by_Submission"))
    w_dec_u   = int(g("win_by_Decision_Unanimous"))
    w_dec_s   = int(g("win_by_Decision_Split"))
    w_dec_m   = int(g("win_by_Decision_Majority"))
    w_tko_doc = int(g("win_by_TKO_Doctor_Stoppage"))
    if won:
        m = method.lower()
        if "doctor" in m:                        w_tko_doc += 1
        elif "ko" in m or "tko" in m:            w_ko      += 1
        elif "sub" in m or "submission" in m:    w_sub     += 1
        elif "unanimous" in m or "u-dec" in m:   w_dec_u   += 1
        elif "split" in m    or "s-dec" in m:    w_dec_s   += 1
        elif "majority" in m or "m-dec" in m:    w_dec_m   += 1

    # Streaks
    win_streak  = int(g("current_win_streak"))
    lose_streak = int(g("current_lose_streak"))
    longest     = int(g("longest_win_streak"))
    if is_draw:
        win_streak = lose_streak = 0
    elif won:
        win_streak += 1;  lose_streak = 0;  longest = max(longest, win_streak)
    else:
        lose_streak += 1; win_streak  = 0

    new_total_rounds  = old_rounds + finish_rounds
    new_title_bouts   = int(g("total_title_bouts"))

    return {
        f"{p}wins":                       new_wins,
        f"{p}losses":                     new_losses,
        f"{p}draw":                       new_draws,
        f"{p}win_by_KO/TKO":             w_ko,
        f"{p}win_by_Submission":          w_sub,
        f"{p}win_by_Decision_Unanimous":  w_dec_u,
        f"{p}win_by_Decision_Split":      w_dec_s,
        f"{p}win_by_Decision_Majority":   w_dec_m,
        f"{p}win_by_TKO_Doctor_Stoppage": w_tko_doc,
        f"{p}avg_SIG_STR_pct":           round(new_sig_pct, 4),
        f"{p}avg_SIG_STR_landed":        round(new_sig_landed, 4),
        f"{p}avg_TD_pct":               round(new_td_pct, 4),
        f"{p}avg_TD_landed":            round(new_td_landed, 4),
        f"{p}avg_SUB_ATT":              round(new_sub_att, 4),
        f"{p}avg_KD":                   round(new_avg_kd, 4),
        f"{p}avg_ctrl_secs":            round(new_avg_ctrl, 1),
        f"{p}avg_head_landed":          round(new_avg_head, 4),
        f"{p}avg_body_landed":          round(new_avg_body, 4),
        f"{p}avg_leg_landed":           round(new_avg_leg, 4),
        f"{p}total_rounds_fought":      new_total_rounds,
        f"{p}current_win_streak":       win_streak,
        f"{p}current_lose_streak":      lose_streak,
        f"{p}longest_win_streak":       longest,
        f"{p}total_title_bouts":        new_title_bouts,
        # Physical attributes — carry forward unchanged
        f"{p}Reach_cms":    g("Reach_cms",  np.nan),
        f"{p}Height_cms":   g("Height_cms", np.nan),
        f"{p}Weight_lbs":   g("Weight_lbs", np.nan),
        f"{p}Stance":       (old_row.get(f"{corner}_Stance", "") if old_row is not None else ""),
        f"{p}age":          g("age", np.nan),
    }


# ---------------------------------------------------------------------------
# Step 5 — build one CSV row per fight
# ---------------------------------------------------------------------------

_WEIGHT_TO_RANK_COL = {
    "Bantamweight":            "Bantamweight_rank",
    "Featherweight":           "Featherweight_rank",
    "Flyweight":               "Flyweight_rank",
    "Heavyweight":             "Heavyweight_rank",
    "Light Heavyweight":       "Light Heavyweight_rank",
    "Lightweight":             "Lightweight_rank",
    "Middleweight":            "Middleweight_rank",
    "Welterweight":            "Welterweight_rank",
    "Women's Bantamweight":    "Women's Bantamweight_rank",
    "Women's Featherweight":   "Women's Featherweight_rank",
    "Women's Flyweight":       "Women's Flyweight_rank",
    "Women's Strawweight":     "Women's Strawweight_rank",
}


def build_row(
    event_date: date,
    event_location: str,
    fight: dict,
    fight_detail: dict,
    r_stats: dict,
    b_stats: dict,
    r_name: str,
    b_name: str,
) -> dict:
    """Assemble a single row dict matching ufc-master.csv column layout."""
    method  = fight_detail["method"]
    finish  = _method_to_finish(method)
    wc      = fight["weight_class"]
    gender  = "FEMALE" if wc.startswith("Women") else "MALE"
    is_draw = fight["winner_idx"] == -1

    if is_draw:
        winner_col = "Draw"
    else:
        winner_col = "Red"  # R_fighter = winner by our assignment

    row: dict = {
        "date":               str(event_date),
        "location":           event_location,
        "country":            np.nan,
        "weight_class":       wc,
        "gender":             gender,
        "title_bout":         fight_detail.get("title_bout", False),
        "Winner":             winner_col,
        "finish":             finish,
        "finish_details":     fight_detail.get("finish_details", ""),
        "finish_round":       fight_detail["finish_round"],
        "finish_round_time":  fight_detail["finish_time"],
        "no_of_rounds":       fight_detail["no_of_rounds"],
        "total_fight_time_secs": fight_detail["total_secs"],
        "empty_arena":        False,
        "R_fighter":          r_name,
        "B_fighter":          b_name,
    }

    # Merge per-fighter stat dicts
    row.update(r_stats)
    row.update(b_stats)

    # Rank columns → NaN (would need live ranking data)
    for prefix in ("R_", "B_"):
        for rank_col in _WEIGHT_TO_RANK_COL.values():
            row[f"{prefix}{rank_col}"] = np.nan
        row[f"{prefix}Pound-for-Pound_rank"] = np.nan
        row[f"{prefix}match_weightclass_rank"] = np.nan
        row[f"{prefix}ev"]   = np.nan
        row[f"{prefix}odds"] = np.nan

    # Odds columns
    for col in ("r_dec_odds","r_ko_odds","r_sub_odds","b_dec_odds","b_ko_odds","b_sub_odds"):
        row[col] = np.nan

    # Diff columns
    def _d(rc, bc, default=0.0):
        rv = r_stats.get(f"R_{rc}", row.get(f"R_{rc}", np.nan))
        bv = b_stats.get(f"B_{bc}", row.get(f"B_{bc}", np.nan))
        try:
            return float(rv) - float(bv)
        except (TypeError, ValueError):
            return np.nan

    row["age_dif"]                = _d("age",              "age")
    row["reach_dif"]              = _d("Reach_cms",        "Reach_cms")
    row["height_dif"]             = _d("Height_cms",       "Height_cms")
    row["sig_str_dif"]            = _d("avg_SIG_STR_landed","avg_SIG_STR_landed")
    row["avg_td_dif"]             = _d("avg_TD_landed",    "avg_TD_landed")
    row["avg_sub_att_dif"]        = _d("avg_SUB_ATT",      "avg_SUB_ATT")
    row["ko_dif"]                 = _d("win_by_KO/TKO",    "win_by_KO/TKO")
    row["sub_dif"]                = _d("win_by_Submission","win_by_Submission")
    row["win_dif"]                = _d("wins",             "wins")
    row["loss_dif"]               = _d("losses",           "losses")
    row["total_round_dif"]        = _d("total_rounds_fought","total_rounds_fought")
    row["win_streak_dif"]         = _d("current_win_streak","current_win_streak")
    row["lose_streak_dif"]        = _d("current_lose_streak","current_lose_streak")
    row["longest_win_streak_dif"] = _d("longest_win_streak","longest_win_streak")
    row["total_title_bout_dif"]   = _d("total_title_bouts","total_title_bouts")
    row["better_rank"]            = np.nan

    return row


# ---------------------------------------------------------------------------
# Round stats writer
# ---------------------------------------------------------------------------

ROUND_CSV = os.path.join(ROOT, "data", "round_stats.csv")
_ROUND_COLS = [
    "fight_url", "date", "fighter_0", "fighter_1",
    "round",
    "ssl_0", "ssa_0", "tdl_0", "tda_0", "kd_0", "ctrl_0",
    "ssl_1", "ssa_1", "tdl_1", "tda_1", "kd_1", "ctrl_1",
]


def _append_round_stats(
    per_round: list[dict],
    fight_url: str,
    fight_date,
    fighter_0: str,
    fighter_1: str,
) -> None:
    """Append per-round stats rows to round_stats.csv."""
    rows = []
    for rd in per_round:
        rows.append({
            "fight_url": fight_url,
            "date":      str(fight_date),
            "fighter_0": fighter_0,
            "fighter_1": fighter_1,
            **rd,
        })
    new_df = pd.DataFrame(rows, columns=_ROUND_COLS)
    write_header = not os.path.exists(ROUND_CSV)
    new_df.to_csv(ROUND_CSV, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    print("Loading existing dataset...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df.copy()  # defragment so .assign() doesn't warn
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    last_date = df["date"].max().date()
    print(f"  Current dataset: {len(df):,} fights  |  last date: {last_date}")

    print(f"\nFetching events after {last_date}...")
    new_events = fetch_new_events(last_date)
    if not new_events:
        print("Dataset is already up to date.")
        return
    print(f"  Found {len(new_events)} new event(s):")
    for e in new_events:
        print(f"    {e['date']}  {e['name']}")

    new_rows: list[dict] = []
    existing_cols = list(df.columns)

    for ev in new_events:
        print(f"\n{'='*60}")
        print(f"Scraping: {ev['name']}  ({ev['date']})")
        try:
            fights = fetch_event_fights(ev["url"])
        except Exception as exc:
            print(f"  ERROR fetching event fights: {exc}")
            continue

        print(f"  {len(fights)} fights found")

        for fight in fights:
            print(f"  → {fight['fighter_0']} vs {fight['fighter_1']} ...", end=" ", flush=True)
            try:
                detail = fetch_fight_stats(fight["fight_url"])
            except Exception as exc:
                print(f"SKIP (error: {exc})")
                continue
            if detail is None:
                print("SKIP (parse failed)")
                continue

            # Use winner_name from the fight detail page (determined via W/L status flag)
            # rather than assuming fighter_0 is always the winner.
            winner_name = detail.get("winner_name")
            is_draw     = winner_name is None

            if is_draw:
                r_name    = detail["fighter_0"]
                b_name    = detail["fighter_1"]
                r_stats_d = detail["stats_0"]
                b_stats_d = detail["stats_1"]
            elif winner_name.strip().lower() == detail["fighter_0"].strip().lower():
                r_name    = detail["fighter_0"]
                b_name    = detail["fighter_1"]
                r_stats_d = detail["stats_0"]
                b_stats_d = detail["stats_1"]
            else:
                # fighter_1 is the actual winner — swap R/B assignment and stats
                r_name    = detail["fighter_1"]
                b_name    = detail["fighter_0"]
                r_stats_d = detail["stats_1"]
                b_stats_d = detail["stats_0"]

            r_won = not is_draw
            b_won = False

            # Look up existing career stats
            r_old, r_c = _get_latest_fighter_row(r_name, df)
            b_old, b_c = _get_latest_fighter_row(b_name, df)
            if r_c == "":
                r_c = "R"
            if b_c == "":
                b_c = "B"

            r_stats = update_career_stats(
                r_old, r_c, r_stats_d,
                detail["total_secs"], detail["finish_round"],
                won=r_won, is_draw=is_draw,
                method=detail["method"], prefix="R_",
            )
            b_stats = update_career_stats(
                b_old, b_c, b_stats_d,
                detail["total_secs"], detail["finish_round"],
                won=b_won, is_draw=is_draw,
                method=detail["method"], prefix="B_",
            )

            row = build_row(
                ev["date"], ev["location"],
                fight, detail,
                r_stats, b_stats,
                r_name, b_name,
            )

            # Set R_fighter / B_fighter (may differ from r_name/b_name if draw logic tweaks)
            row["R_fighter"] = r_name
            row["B_fighter"] = b_name
            new_rows.append(row)

            # Save per-round data if the table had multiple rows (one per round)
            per_rd = detail.get("per_round", [])
            if len(per_rd) > 1:
                _append_round_stats(
                    per_rd, fight["fight_url"], ev["date"],
                    r_name, b_name,
                )

            print(f"OK  ({detail['method']}, R{detail['finish_round']} {detail['finish_time']})")

    if not new_rows:
        print("\nNo new fights were scraped.")
        return

    print(f"\n{'='*60}")
    print(f"Scraped {len(new_rows)} new fight(s) across {len(new_events)} event(s).")

    if dry_run:
        print("\n[DRY RUN — not writing to CSV]")
        new_df = pd.DataFrame(new_rows)
        print(new_df[["date","R_fighter","B_fighter","Winner","finish","weight_class"]].to_string(index=False))
        return

    # Align columns to existing CSV
    new_df = pd.DataFrame(new_rows).reindex(columns=existing_cols)
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
    updated = pd.concat([df, new_df], ignore_index=True).sort_values("date")
    updated.to_csv(CSV_PATH, index=False)
    print(f"Saved → {CSV_PATH}  ({len(updated):,} total fights)")
    print("\nNext steps:")
    print("  python src/cluster.py data/raw/ufc-master.csv")
    print("  python src/features.py data/raw/ufc-master.csv")
    print("  python src/train.py")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
