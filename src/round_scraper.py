"""
Historical per-round stats scraper.

Visits every completed event on UFCStats.com, scrapes per-round striking
and grappling numbers for each fight, and writes them to round_stats.csv.

Schema written:
    fight_url, date, fighter_0, fighter_1,
    round,
    ssl_0, ssa_0, tdl_0, tda_0, kd_0, ctrl_0,
    ssl_1, ssa_1, tdl_1, tda_1, kd_1, ctrl_1

Run once to build the database, then incrementally to add new fights:
    python src/round_scraper.py            # all events
    python src/round_scraper.py --recent   # last 365 days only
    python src/round_scraper.py --dry-run  # print count without writing
"""

import os
import sys
import re
import time
import argparse
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(format="%(asctime)s  %(levelname)s  %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

BASE       = "http://ufcstats.com"
EVENTS_URL = f"{BASE}/statistics/events/completed?page=all"
DELAY      = 1.2   # seconds between requests

ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ROUND_CSV = os.path.join(ROOT, "data", "round_stats.csv")

_ROUND_COLS = [
    "fight_url", "date", "fighter_0", "fighter_1",
    "round",
    "ssl_0", "ssa_0", "tdl_0", "tda_0", "kd_0", "ctrl_0",
    "ssl_1", "ssa_1", "tdl_1", "tda_1", "kd_1", "ctrl_1",
]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url: str) -> BeautifulSoup:
    headers = {"User-Agent": "Mozilla/5.0 (ufc-round-scraper/1.0; research)"}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            time.sleep(DELAY)
            return BeautifulSoup(r.text, "html.parser")
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed {url}: {exc}") from exc
            time.sleep(DELAY * 4)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _x_of_y(s: str) -> tuple[int, int]:
    m = re.match(r"(\d+)\s+of\s+(\d+)", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _split_cell(cell) -> tuple[str, str]:
    parts = [p.strip() for p in cell.get_text(separator="|", strip=True).split("|") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return (parts[0], parts[0]) if parts else ("", "")


def _parse_ctrl_secs(s: str) -> int:
    m = re.match(r"(\d+):(\d+)", s.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def fetch_all_events(since: date | None = None) -> list[dict]:
    """Return list of {name, date, url} for all (or recent) events."""
    log.info("Fetching event list from UFCStats…")
    soup   = _get(EVENTS_URL)
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        link    = row.select_one("a.b-link")
        date_el = row.select_one("span.b-statistics__date")
        if not link or not date_el:
            continue
        try:
            ev_date = datetime.strptime(date_el.get_text(strip=True), "%B %d, %Y").date()
        except ValueError:
            continue
        if since and ev_date < since:
            continue
        events.append({"name": link.get_text(strip=True), "date": ev_date, "url": link["href"]})
    log.info("Found %d events%s", len(events),
             f" since {since}" if since else " (all time)")
    return sorted(events, key=lambda e: e["date"])


def fetch_fight_urls(event_url: str) -> list[str]:
    soup  = _get(event_url)
    urls  = []
    for row in soup.select("tr.b-fight-details__table-row__hover"):
        url = row.get("data-link", "").strip()
        if url:
            urls.append(url)
    return urls


def scrape_fight_rounds(fight_url: str, event_date: date) -> list[dict] | None:
    """
    Return a list of per-round dicts for this fight, or None if unparseable.
    """
    soup = _get(fight_url)

    # Fighter names
    persons = soup.select("div.b-fight-details__person")
    if len(persons) < 2:
        return None

    def _name(div):
        el = div.select_one("h3.b-fight-details__person-name a") or \
             div.select_one("h3.b-fight-details__person-name")
        return el.get_text(strip=True) if el else ""

    f0 = _name(persons[0])
    f1 = _name(persons[1])
    if not f0 or not f1:
        return None

    tables = soup.select("table.b-fight-details__table")
    if not tables:
        return None

    rows   = []
    rd_idx = 0

    for row in tables[0].select("tr"):
        cells = row.select("td")
        if not cells or len(cells) < 8:
            continue

        # KD
        v0, v1 = _split_cell(cells[1])
        kd0 = int(v0) if v0.isdigit() else 0
        kd1 = int(v1) if v1.isdigit() else 0

        # Sig strikes
        s0, s1   = _split_cell(cells[2])
        l0, a0   = _x_of_y(s0)
        l1, a1   = _x_of_y(s1)

        # Takedowns
        t0, t1   = _split_cell(cells[5])
        tl0, ta0 = _x_of_y(t0)
        tl1, ta1 = _x_of_y(t1)

        # Sub attempts
        u0, u1 = _split_cell(cells[7])
        sa0 = int(u0) if u0.isdigit() else 0
        sa1 = int(u1) if u1.isdigit() else 0

        # Control time
        ctrl0 = ctrl1 = 0
        if len(cells) > 9:
            c0, c1 = _split_cell(cells[9])
            ctrl0  = _parse_ctrl_secs(c0)
            ctrl1  = _parse_ctrl_secs(c1)

        rd_idx += 1
        rows.append({
            "fight_url": fight_url,
            "date":      str(event_date),
            "fighter_0": f0,
            "fighter_1": f1,
            "round":     rd_idx,
            "ssl_0": l0,  "ssa_0": a0,  "tdl_0": tl0, "tda_0": ta0,
            "kd_0":  kd0, "ctrl_0": ctrl0,
            "ssl_1": l1,  "ssa_1": a1,  "tdl_1": tl1, "tda_1": ta1,
            "kd_1":  kd1, "ctrl_1": ctrl1,
        })

    # Require at least 2 rounds to be useful for degradation analysis
    return rows if len(rows) >= 2 else None


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def load_done_urls() -> set[str]:
    if not os.path.exists(ROUND_CSV):
        return set()
    try:
        df = pd.read_csv(ROUND_CSV, usecols=["fight_url"])
        return set(df["fight_url"].dropna().unique())
    except Exception:
        return set()


def append_rows(rows: list[dict]) -> None:
    df  = pd.DataFrame(rows, columns=_ROUND_COLS)
    hdr = not os.path.exists(ROUND_CSV)
    df.to_csv(ROUND_CSV, mode="a", header=hdr, index=False)


def run(since: date | None = None, dry_run: bool = False) -> None:
    done_urls = load_done_urls()
    log.info("%d fights already in round_stats.csv — will skip", len(done_urls))

    events      = fetch_all_events(since=since)
    total_new   = 0
    total_skipped = 0
    total_events  = len(events)

    for ev_i, ev in enumerate(events, 1):
        log.info("[%d/%d] %s (%s)", ev_i, total_events, ev["name"], ev["date"])
        try:
            fight_urls = fetch_fight_urls(ev["url"])
        except Exception as exc:
            log.warning("  Skipping event (fetch failed): %s", exc)
            continue

        for fight_url in fight_urls:
            if fight_url in done_urls:
                total_skipped += 1
                continue
            try:
                rows = scrape_fight_rounds(fight_url, ev["date"])
            except Exception as exc:
                log.warning("  Fight scrape failed %s: %s", fight_url, exc)
                rows = None

            if rows is None:
                log.debug("  No usable round data: %s", fight_url)
                done_urls.add(fight_url)  # don't retry
                continue

            if not dry_run:
                append_rows(rows)
            done_urls.add(fight_url)
            total_new += 1
            log.info("  Saved %d rounds — %s vs %s",
                     len(rows), rows[0]["fighter_0"], rows[0]["fighter_1"])

    log.info(
        "Done. New fights saved: %d  |  Skipped (already done): %d",
        total_new, total_skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape per-round UFC stats")
    parser.add_argument("--recent",  action="store_true",
                        help="Only scrape events from the last 365 days")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count fights without writing anything")
    args = parser.parse_args()

    since = (date.today() - timedelta(days=365)) if args.recent else None
    run(since=since, dry_run=args.dry_run)
