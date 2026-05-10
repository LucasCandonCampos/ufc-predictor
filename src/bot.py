"""
Telegram bot interface for the UFC matchup predictor.

Setup:
    export TELEGRAM_BOT_TOKEN=<your_token>
    export ANTHROPIC_API_KEY=<your_key>   # optional — enables LLM summary
    export ADMIN_CHAT_ID=<your_chat_id>   # optional — restricts /update to one user

Run:
    python src/bot.py

Commands:
    /predict [Fighter A] vs [Fighter B]   — predict a matchup
    /top [n]                              — top-n fighters by Glicko-2 rating (default 10)
    /stats                                — model info and accuracy
    /scan [event]                         — scan a UFC card for +EV Polymarket bets
    /update                               — scrape new fights and retrain (admin only)
    /help                                 — usage guide
"""

import asyncio
import logging
import os
import sys

import pandas as pd
from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Load .env from the project root before anything else reads env vars
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_env_file = os.path.join(_ROOT, ".env")
if os.path.exists(_env_file):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Fallback: parse the .env file manually (no extra deps needed)
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

sys.path.insert(0, os.path.dirname(__file__))
from predict import predict_matchup, _factor_line
from ev_scanner import scan_card, EV_THRESHOLD

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

ROOT     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")

# Weight classes where the model has negative ROI — predictions shown but no bet recommended
_NO_BET_DIVISIONS = {
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Heavyweight",   # -19.5% ROI: one-punch variance makes model stats unreliable
}


def _is_no_bet_division(weight_class: str) -> bool:
    return any(weight_class.startswith(wc) or wc in weight_class
               for wc in _NO_BET_DIVISIONS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escape characters reserved in Telegram HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _admin_chat_id() -> int | None:
    raw = os.getenv("ADMIN_CHAT_ID", "")
    return int(raw) if raw.strip().lstrip("-").isdigit() else None


def _is_admin(update: Update) -> bool:
    admin = _admin_chat_id()
    return admin is None or update.effective_chat.id == admin


# ---------------------------------------------------------------------------
# Response formatter
# ---------------------------------------------------------------------------

def format_prediction(result: dict) -> str:
    """Build an HTML-formatted Telegram message from a predict_matchup result dict."""
    na       = _esc(result["name_a"])
    nb       = _esc(result["name_b"])
    winner   = _esc(result["winner"])
    conf     = result["confidence"]
    style_a  = _esc(result["style_a"])
    style_b  = _esc(result["style_b"])
    ga       = result.get("glicko_a", 1500.0)
    gb       = result.get("glicko_b", 1500.0)
    rda      = result.get("glicko_rd_a", 350.0)
    rdb      = result.get("glicko_rd_b", 350.0)
    top_a    = result["top_for_a"]
    top_b    = result["top_for_b"]
    summary  = result.get("summary")
    wc       = result.get("weight_class", "")
    no_bet   = _is_no_bet_division(wc)

    lines = [
        f"🥊 <b>{na}</b>  vs  <b>{nb}</b>",
        f"<i>{style_a} vs {style_b}</i>",
        f"⚡ Ratings:  {na}: <b>{ga:.0f}</b> ±{rda:.0f}  |  {nb}: <b>{gb:.0f}</b> ±{rdb:.0f}",
        "",
        f"🏆 <b>{winner}</b> wins  ({conf:.0%} confidence)",
        "",
        f"<b>Favours {na}:</b>",
    ]

    for label, val, shap in top_a:
        lines.append("  " + _esc(_factor_line(label, val, shap, "a")))

    if not top_a:
        lines.append("  (no dominant factors)")

    lines += ["", f"<b>Favours {nb}:</b>"]
    for label, val, shap in top_b:
        lines.append("  " + _esc(_factor_line(label, val, shap, "b")))

    if not top_b:
        lines.append("  (no dominant factors)")

    # Finish method and round prediction
    finish_method       = result.get("finish_method")
    finish_method_probs = result.get("finish_method_probs") or {}
    finish_round        = result.get("finish_round")
    finish_round_probs  = result.get("finish_round_probs")  or {}

    if finish_method and finish_method_probs:
        # Sort methods by probability descending
        sorted_methods = sorted(finish_method_probs.items(), key=lambda x: x[1], reverse=True)
        # Abbreviate "Submission" → "Sub"
        def _abbrev_method(m: str) -> str:
            return m.replace("Submission", "Sub")
        method_parts = " | ".join(f"{_abbrev_method(m)} ({p:.0%})" for m, p in sorted_methods)
        lines += ["", f"🎯 Method: {_esc(method_parts)}"]

        if finish_method != "Decision" and finish_round_probs:
            # Sort rounds by probability descending, abbreviate display
            sorted_rounds = sorted(finish_round_probs.items(), key=lambda x: x[1], reverse=True)
            def _abbrev_round(r: str) -> str:
                return r.replace("Round ", "R").replace("4 or 5", "4+")
            round_parts = " | ".join(f"{_abbrev_round(r)} ({p:.0%})" for r, p in sorted_rounds)
            lines.append(f"🔔 Round:  {_esc(round_parts)}")

    if no_bet:
        lines += [
            "",
            f"⚠️ <b>No bet recommended</b> — {_esc(wc)} has shown negative ROI in backtests. "
            "Prediction shown for informational purposes only.",
        ]

    if summary:
        lines += ["", f"💬 <i>{_esc(summary)}</i>"]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline runner (used by /update)
# ---------------------------------------------------------------------------

async def _run(cmd: list[str], cwd: str) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined output tail)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    stdout, _ = await proc.communicate()
    tail = stdout.decode(errors="replace")[-1200:]
    return proc.returncode, tail


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def predict_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/predict Fighter A vs Fighter B"""
    raw = " ".join(context.args).strip()

    if not raw or " vs " not in raw.lower():
        await update.message.reply_text(
            "Usage: <code>/predict [Fighter A] vs [Fighter B]</code>\n"
            "Example: <code>/predict Jon Jones vs Stipe Miocic</code>",
            parse_mode="HTML",
        )
        return

    idx    = raw.lower().index(" vs ")
    name_a = raw[:idx].strip()
    name_b = raw[idx + 4:].strip()

    thinking = await update.message.reply_text("⏳ Analysing matchup…")

    try:
        result = predict_matchup(name_a, name_b, verbose=False)
        await thinking.edit_text(format_prediction(result), parse_mode="HTML")
    except ValueError as exc:
        await thinking.edit_text(f"❌ {_esc(str(exc))}", parse_mode="HTML")
    except Exception as exc:
        log.exception("Prediction error for '%s vs %s'", name_a, name_b)
        await thinking.edit_text(
            f"❌ Something went wrong: <code>{_esc(str(exc))}</code>",
            parse_mode="HTML",
        )


async def top_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/top [n]  —  top-n fighters by current Glicko-2 rating"""
    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 30))
        except ValueError:
            pass

    path = os.path.join(DATA_DIR, "fighter_ratings.csv")
    if not os.path.exists(path):
        await update.message.reply_text(
            "❌ Ratings file not found. Run the pipeline first.",
            parse_mode="HTML",
        )
        return

    df = pd.read_csv(path).head(n)
    lines = [f"🏅 <b>Top {n} fighters by Glicko-2 rating</b>", ""]
    for i, row in df.iterrows():
        medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(int(i), f"{int(i)+1}.")
        lines.append(
            f"{medal} <b>{_esc(row['fighter'])}</b>  "
            f"{row['glicko_rating']:.0f} ±{row['glicko_rd']:.0f}"
        )

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/stats  —  model metadata and accuracy"""
    import pickle
    model_path = os.path.join(ROOT, "models", "model.pkl")
    if not os.path.exists(model_path):
        await update.message.reply_text("❌ No trained model found.", parse_mode="HTML")
        return

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    acc    = bundle.get("test_accuracy", float("nan"))
    auc    = bundle.get("test_roc_auc",  float("nan"))
    brier  = bundle.get("test_brier",    float("nan"))
    cutoff = bundle.get("train_cutoff",  "unknown")

    ratings_path = os.path.join(DATA_DIR, "fighter_ratings.csv")
    n_fighters = len(pd.read_csv(ratings_path)) if os.path.exists(ratings_path) else "?"

    features_path = os.path.join(DATA_DIR, "matchup_features.csv")
    n_fights = "?"
    date_range = ""
    if os.path.exists(features_path):
        df = pd.read_csv(features_path, usecols=["date"])
        n_fights = f"{len(df) // 2:,}"
        date_range = f"{df['date'].min()[:10]} → {df['date'].max()[:10]}"

    lines = [
        "📊 <b>Model stats</b>",
        "",
        f"Test accuracy : <b>{acc:.1%}</b>",
        f"ROC-AUC       : <b>{auc:.4f}</b>",
        f"Brier score   : <b>{brier:.4f}</b>",
        f"Train cutoff  : {_esc(str(cutoff))}",
        "",
        f"Fights in dataset : {n_fights}  ({_esc(date_range)})",
        f"Fighters rated    : {n_fighters}",
        "",
        "Features: style matchup · Glicko-2 · recent form · "
        "absorbed stats · weight class · age · reach · experience",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


def _format_scan_telegram(results: list, event_query: str | None) -> str:
    """Build an HTML-formatted Telegram message from ev_scanner results."""
    title = _esc(event_query.upper() if event_query else "UFC")

    if not results:
        return (
            f"🔍 <b>EV SCAN — {title}</b>\n\n"
            "No significant EV opportunities found on Polymarket.\n"
            "<i>Either the card isn't listed yet or the market prices are in line with the model.</i>"
        )

    # Count actionable opportunities (exclude filtered divisions + bad odds ranges)
    opp_count = sum(
        1 for r in results
        if not _is_no_bet_division(r.get("weight_class", ""))
        and (r.get("bet_a") or r.get("bet_b"))
    )
    lines = [
        f"🔍 <b>EV SCAN — {title}</b>",
        f"<i>{len(results)} divergent fight(s)  ·  {opp_count} actionable +EV opportunity/ies</i>",
        "",
    ]

    for i, r in enumerate(results, 1):
        na  = _esc(r["fighter_a"])
        nb  = _esc(r["fighter_b"])
        ma  = r["model_prob_a"]
        mb  = r["model_prob_b"]
        pa  = r["market_prob_a"]
        pb  = r["market_prob_b"]
        ea  = r["edge_a"]
        eb  = r["edge_b"]
        ka  = r["kelly_a"]
        kb  = r["kelly_b"]
        sa  = _esc(r.get("style_a", ""))
        sb  = _esc(r.get("style_b", ""))
        wc  = r.get("weight_class", "")
        no_bet = _is_no_bet_division(wc)

        lines.append(f"<b>#{i}  {na} vs {nb}</b>")
        if sa and sb:
            lines.append(f"<i>{sa} vs {sb}</i>")
        lines.append(f"Model  : {ma:.0%} / {mb:.0%}")
        lines.append(f"Market : {pa:.0%} / {pb:.0%}")

        if no_bet:
            lines.append(f"⚠️ <i>No bet — {_esc(wc)} excluded (negative historical ROI)</i>")
        else:
            if r.get("bet_a"):
                lines.append(
                    f"✅ <b>BET {na}</b>  edge <b>+{ea:.1%}</b>  kelly <b>{ka:.1%}</b>"
                )
            elif ea >= EV_THRESHOLD:
                lines.append(
                    f"⚠️ <i>{na} edge +{ea:.1%} — skipped (mkt {pa:.0%} is outside 45-70% profitable range)</i>"
                )
            if r.get("bet_b"):
                lines.append(
                    f"✅ <b>BET {nb}</b>  edge <b>+{eb:.1%}</b>  kelly <b>{kb:.1%}</b>"
                )
            elif eb >= EV_THRESHOLD:
                lines.append(
                    f"⚠️ <i>{nb} edge +{eb:.1%} — skipped (mkt {pb:.0%} is outside 45-70% profitable range)</i>"
                )
            if ea <= -EV_THRESHOLD:
                lines.append(f"📉 {na} overpriced by {-ea:.1%}")
            if eb <= -EV_THRESHOLD:
                lines.append(f"📉 {nb} overpriced by {-eb:.1%}")

        lines.append("")

    lines.append(
        "<i>Edge = model prob − Polymarket cost (3% taker fee included)  ·  "
        "Kelly = 25% of full Kelly</i>"
    )
    return "\n".join(lines)


_CARDIO_SHRINK_K = 6  # Bayesian shrinkage constant — score * n/(n+k)
                      # k=6: 3 fights → 33% weight, 10 fights → 63%, 20 fights → 77%


def _load_women_fighters() -> set[str]:
    """Build a set of normalised names for fighters who competed in women's divisions."""
    csv = os.path.join(DATA_DIR, "raw", "ufc-master.csv")
    if not os.path.exists(csv):
        return set()
    try:
        df = pd.read_csv(csv, usecols=["R_fighter", "B_fighter", "weight_class"])
        womens = df["weight_class"].str.contains("Women", na=False)
        names: set[str] = set()
        for col in ("R_fighter", "B_fighter"):
            names.update(df[womens][col].dropna().str.strip().str.lower())
        return names
    except Exception:
        return set()


def _load_cardio_profiles(min_fights: int = 5) -> pd.DataFrame:
    """
    Load round fatigue profiles, filter women's divisions, and add an adjusted_score
    column that shrinks the raw degradation_score toward zero based on sample size.

    min_fights=5 for rankings (quality floor); use min_fights=3 for individual lookups
    so fighters with fewer UFC appearances (e.g. Chimaev) can still be compared.
    """
    from fatigue import build_round_fatigue
    import datetime
    df = build_round_fatigue(min_fights=min_fights)
    if df.empty:
        return df

    # Remove women's division fighters
    women = _load_women_fighters()
    if women:
        df = df[~df["fighter"].str.strip().str.lower().isin(women)].copy()

    # Remove fighters inactive for more than two years
    cutoff = pd.Timestamp(datetime.date.today()) - pd.DateOffset(years=2)
    if "last_fight_date" in df.columns:
        last = pd.to_datetime(df["last_fight_date"], errors="coerce")
        df = df[last >= cutoff].copy()

    n = df["n_fights"]
    df["adjusted_score"] = df["degradation_score"] * (n / (n + _CARDIO_SHRINK_K))

    if "fade_score" in df.columns and "late_activity_abs" in df.columns:
        # bad_cardio_score = fade × (1 - late_ssl_percentile).
        # Uses late_ssl (strikes only) for the percentile so pure strikers aren't
        # unfairly penalised for having low grappling output. A fighter who fades
        # 40% but still lands lots of strikes late is pacing; one who fades 40%
        # AND lands few strikes late is a true gasser.
        late_ssl_pct = df["late_ssl_abs"].rank(pct=True, ascending=True) if "late_ssl_abs" in df.columns else pd.Series(0.5, index=df.index)
        df["bad_cardio_raw"] = df["fade_score"] * (1.0 - late_ssl_pct)
        df["bad_cardio_adjusted"] = df["bad_cardio_raw"] * (n / (n + _CARDIO_SHRINK_K))
    return df


def _no_data_msg() -> str:
    return (
        "❌ No round-by-round data found.\n\n"
        "Run <code>python src/round_scraper.py</code> to build it."
    )


def _cardio_bar(score: float) -> str:
    filled = max(0, min(10, round((score + 1) / 2 * 10)))
    return "█" * filled + "░" * (10 - filled)


def _cardio_confidence(n_fights: int) -> str:
    """
    Return a confidence label based on how many multi-round fights were used.
    More fights = tighter slope estimate = higher confidence.
    """
    if n_fights >= 15:
        return "🟢 High"
    if n_fights >= 8:
        return "🟡 Medium"
    if n_fights >= 5:
        return "🟠 Low"
    return "🔴 Very low"


def _fade_bar(fade: float) -> str:
    """Bar showing how much of the fighter's early output they lose late. More = worse."""
    filled = max(0, min(10, round(fade * 10)))
    return "▓" * filled + "░" * (10 - filled)


def _fade_row(rank: int, row: pd.Series) -> str:
    """Row format for /worstcardio — shows fade % and late-round output level."""
    fade     = row.get("fade_score", float("nan"))
    n_fights = int(row["n_fights"])
    bar      = _fade_bar(fade if pd.notna(fade) else 0.0)
    conf     = _cardio_confidence(n_fights)

    act_v  = row.get("late_activity_abs", float("nan"))
    ssl_v  = row.get("late_ssl_abs", float("nan"))

    parts = []
    if pd.notna(fade):
        parts.append(f"fades {fade:.0%} from own early pace")
    if pd.notna(act_v):
        parts.append(f"late output {act_v:.1f}/rd")
    if pd.notna(ssl_v):
        parts.append(f"{ssl_v:.1f} str/rd late")
    detail = "  ·  ".join(parts)

    return (
        f"{rank}. <b>{_esc(row['fighter'])}</b>  {conf} ({n_fights} fights)\n"
        f"   {bar}  {fade:.0%} fade\n"
        f"   <i>{_esc(detail)}</i>"
    )


def _cardio_row(rank: int, row: pd.Series) -> str:
    raw_score  = row["degradation_score"]
    adj_score  = row.get("adjusted_score", raw_score)
    n_fights   = int(row["n_fights"])
    bar        = _cardio_bar(adj_score)
    confidence = _cardio_confidence(n_fights)

    act_v  = row.get("late_activity_abs", float("nan"))    # grappling-inclusive late output
    ssl_v  = row.get("late_ssl_abs", float("nan"))          # strikes only (for reference)
    slope_v = row.get("activity_drop_per_rd", float("nan")) # combined activity slope

    parts = []
    if not pd.isna(act_v):
        parts.append(f"late output {act_v:.1f}/rd")
    if not pd.isna(ssl_v):
        parts.append(f"{ssl_v:.1f} str")
    if not pd.isna(slope_v):
        parts.append(f"slope {slope_v:+.1f}/rd")
    detail = "  ·  ".join(parts)

    return (
        f"{rank}. <b>{_esc(row['fighter'])}</b>  {confidence} ({n_fights} fights)\n"
        f"   {bar}  {adj_score:+.2f}  <i>(raw {raw_score:+.2f})</i>\n"
        f"   <i>{_esc(detail)}</i>"
    )


_CARDIO_FOOTER = (
    "<i>Score = late-round combined output (strikes + takedowns + control time) + slope. "
    "Grapplers and wrestlers get full credit for takedowns and control. "
    "Higher = dominant output deep into fights.</i>"
)


async def bestcardio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/bestcardio [n]  —  fighters whose performance holds up best round-to-round"""
    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 30))
        except ValueError:
            pass

    profiles = _load_cardio_profiles()
    if profiles.empty:
        await update.message.reply_text(_no_data_msg(), parse_mode="HTML")
        return

    df = profiles.sort_values("adjusted_score", ascending=False).head(n).reset_index(drop=True)
    lines = [
        f"<b>🫁 Best cardio — top {n}</b>",
        "<i>Fighters whose output, accuracy and TD defense hold up deepest into fights</i>",
        "",
    ]
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(_cardio_row(rank, row))
    lines += ["", _CARDIO_FOOTER]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def worstcardio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/worstcardio [n]  —  fighters who fade most relative to their own early-round output"""
    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 30))
        except ValueError:
            pass

    profiles = _load_cardio_profiles()
    if profiles.empty:
        await update.message.reply_text(_no_data_msg(), parse_mode="HTML")
        return

    sort_col = "bad_cardio_adjusted" if "bad_cardio_adjusted" in profiles.columns else "adjusted_score"
    df = profiles.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)
    lines = [
        f"<b>😮‍💨 Worst cardio — top {n} faders</b>",
        "<i>Fighters whose combined output drops most in late rounds relative to their own early pace</i>",
        "",
    ]
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(_fade_row(rank, row))
    lines += [
        "",
        "<i>Fade = % drop in combined output (strikes + TDs + control) from rounds 1-2 to rounds 3+. "
        "Measured against each fighter's own early pace, so low-volume fighters aren't unfairly penalised.</i>",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def comparecardio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/comparecardio Fighter A vs Fighter B"""
    raw = " ".join(context.args).strip()
    if not raw or " vs " not in raw.lower():
        await update.message.reply_text(
            "Usage: <code>/comparecardio [Fighter A] vs [Fighter B]</code>\n"
            "Example: <code>/comparecardio Khabib Nurmagomedov vs Conor McGregor</code>",
            parse_mode="HTML",
        )
        return

    idx    = raw.lower().index(" vs ")
    name_a = raw[:idx].strip()
    name_b = raw[idx + 4:].strip()

    # Use a lower min_fights threshold for individual lookups — fighters with
    # just 3-4 UFC fights (e.g. recent signees, Chimaev) should still be comparable.
    profiles = _load_cardio_profiles(min_fights=3)
    if profiles.empty:
        await update.message.reply_text(_no_data_msg(), parse_mode="HTML")
        return

    import difflib

    def _norm(n): return str(n).strip().lower()

    lookup = {_norm(r["fighter"]): r for _, r in profiles.iterrows()}
    all_keys = list(lookup.keys())

    def _find(name: str) -> tuple[str | None, pd.Series | None]:
        key = _norm(name)
        # 1. Exact match
        if key in lookup:
            return lookup[key]["fighter"], lookup[key]
        # 2. Substring match (handles nicknames / partial names)
        matches = [(f, v) for f, v in lookup.items() if key in f or f in key]
        if matches:
            best = min(matches, key=lambda x: abs(len(x[0]) - len(key)))
            return best[1]["fighter"], best[1]
        # 3. Fuzzy match — catches typos like "Oliviera" vs "Oliveira"
        close = difflib.get_close_matches(key, all_keys, n=1, cutoff=0.75)
        if close:
            return lookup[close[0]]["fighter"], lookup[close[0]]
        return None, None

    fname_a, row_a = _find(name_a)
    fname_b, row_b = _find(name_b)

    missing = []
    if row_a is None:
        missing.append(name_a)
    if row_b is None:
        missing.append(name_b)
    if missing:
        await update.message.reply_text(
            f"❌ No round data found for: <b>{_esc(', '.join(missing))}</b>\n"
            "<i>Fighter may not have enough multi-round fights in the dataset.</i>",
            parse_mode="HTML",
        )
        return

    score_a = row_a["degradation_score"]
    score_b = row_b["degradation_score"]
    adj_a   = row_a.get("adjusted_score", score_a)
    adj_b   = row_b.get("adjusted_score", score_b)
    better  = fname_a if adj_a >= adj_b else fname_b
    diff    = abs(adj_a - adj_b)

    def _metric_line(label: str, val_a, val_b, higher_is_better: bool = True) -> str:
        if pd.isna(val_a) or pd.isna(val_b):
            return f"  {label}: <i>insufficient data</i>"
        winner = fname_a if (val_a >= val_b) == higher_is_better else fname_b
        edge = abs(val_a - val_b)
        return (
            f"  {label}:  {_esc(fname_a)} {val_a:+.3f}  vs  {_esc(fname_b)} {val_b:+.3f}"
            f"  → <b>{_esc(winner)}</b> +{edge:.3f}"
        )

    nf_a = int(row_a["n_fights"])
    nf_b = int(row_b["n_fights"])
    conf_a = _cardio_confidence(nf_a)
    conf_b = _cardio_confidence(nf_b)

    def _pct(v) -> str:
        return f"{v:.0%}" if pd.notna(v) else "N/A"

    lines = [
        f"<b>🫁 Cardio Comparison</b>",
        f"<b>{_esc(fname_a)}</b>  vs  <b>{_esc(fname_b)}</b>",
        "",
        f"{_esc(fname_a)}  {conf_a} ({nf_a} fights):",
        f"  {_cardio_bar(adj_a)}  {adj_a:+.2f}  <i>(raw {score_a:+.2f})</i>",
        "",
        f"{_esc(fname_b)}  {conf_b} ({nf_b} fights):",
        f"  {_cardio_bar(adj_b)}  {adj_b:+.2f}  <i>(raw {score_b:+.2f})</i>",
        "",
        "<b>Breakdown:</b>",
        _metric_line("Late output/rd (incl. grappling)",
                     row_a.get("late_activity_abs", float("nan")),
                     row_b.get("late_activity_abs", float("nan")), higher_is_better=True),
        _metric_line("Late strikes/rd",
                     row_a.get("late_ssl_abs", float("nan")),
                     row_b.get("late_ssl_abs", float("nan")), higher_is_better=True),
        _metric_line("Activity slope/rd", row_a.get("activity_drop_per_rd", float("nan")),
                                           row_b.get("activity_drop_per_rd", float("nan")), higher_is_better=True),
        _metric_line("Strike retention", row_a.get("output_retention", float("nan")),
                                          row_b.get("output_retention", float("nan")), higher_is_better=True),
        "",
        f"🏆 <b>{_esc(better)}</b> has better cardio by <b>{diff:.2f}</b> points",
        "",
        _CARDIO_FOOTER,
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def scan_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/scan [event]  —  scan a UFC card for +EV Polymarket opportunities"""
    event_query = " ".join(context.args).strip() or None
    label       = event_query or "upcoming UFC events"

    thinking = await update.message.reply_text(
        f"⏳ Scanning <b>{_esc(label)}</b> on Polymarket…",
        parse_mode="HTML",
    )

    try:
        results = scan_card(event_query)
        text    = _format_scan_telegram(results, event_query)
        await thinking.edit_text(text, parse_mode="HTML")
    except Exception as exc:
        log.exception("Scan error for '%s'", event_query)
        await thinking.edit_text(
            f"❌ Scan failed: <code>{_esc(str(exc))}</code>",
            parse_mode="HTML",
        )


async def update_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/update  —  scrape new fights and retrain (admin only)"""
    if not _is_admin(update):
        await update.message.reply_text("❌ This command is restricted to admins.")
        return

    csv_path = os.path.join(DATA_DIR, "raw", "ufc-master.csv")
    py = sys.executable

    # Step 1 — scraper
    msg = await update.message.reply_text("⏳ <b>Step 1/3</b>  Scraping new fights…", parse_mode="HTML")
    rc, out = await _run([py, "src/scraper.py", csv_path], cwd=ROOT)
    if rc != 0:
        await msg.edit_text(
            f"❌ Scraper failed:\n<pre>{_esc(out[-800:])}</pre>", parse_mode="HTML"
        )
        return

    # Parse how many fights were added from scraper output
    new_fights = next(
        (l for l in out.splitlines() if "new fight" in l.lower() or "added" in l.lower()),
        "",
    )

    # Step 2 — features
    await msg.edit_text("⏳ <b>Step 2/3</b>  Engineering features…", parse_mode="HTML")
    rc, out = await _run([py, "src/features.py", csv_path], cwd=ROOT)
    if rc != 0:
        await msg.edit_text(
            f"❌ features.py failed:\n<pre>{_esc(out[-800:])}</pre>", parse_mode="HTML"
        )
        return

    # Step 3 — retrain
    await msg.edit_text("⏳ <b>Step 3/3</b>  Retraining model…", parse_mode="HTML")
    rc, out = await _run([py, "src/train.py"], cwd=ROOT)
    if rc != 0:
        await msg.edit_text(
            f"❌ train.py failed:\n<pre>{_esc(out[-800:])}</pre>", parse_mode="HTML"
        )
        return

    # Parse final accuracy from train output
    acc_line = next(
        (l for l in out.splitlines() if "accuracy" in l.lower() and "calibrat" in l.lower()),
        "",
    )

    # Reload predict module so new feature code takes effect without a bot restart.
    # This handles the case where /update retrains a model with new features that
    # the currently-loaded predict.py code doesn't know how to compute yet.
    try:
        import importlib
        import predict as _predict_mod
        importlib.reload(_predict_mod)
        # Re-bind the symbols bot.py imported directly at startup
        import bot as _self
        _self.predict_matchup = _predict_mod.predict_matchup
        _self._factor_line    = _predict_mod._factor_line
    except Exception:
        pass

    # Invalidate the in-process model cache so next /predict uses the new model
    try:
        from predict import _cache
        _cache.clear()
    except Exception:
        pass

    summary_lines = ["✅ <b>Update complete</b>", ""]
    if new_fights:
        summary_lines.append(f"🆕 {_esc(new_fights)}")
    if acc_line:
        summary_lines.append(f"📈 {_esc(acc_line.strip())}")

    await msg.edit_text("\n".join(summary_lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# Chin / durability ranking helpers
# ---------------------------------------------------------------------------

def _load_chin_profiles(min_fights: int = 5, exclude_inactive: bool = True) -> pd.DataFrame:
    """
    Build chin profiles, filter women's divisions, and optionally filter inactive fighters.
    min_fights=5 + exclude_inactive=True for rankings.
    min_fights=3 + exclude_inactive=False for /comparechin individual lookups.
    """
    import datetime
    from chin import build_chin_profiles
    df = build_chin_profiles(min_fights=min_fights)
    if df.empty:
        return df

    # Remove women's division fighters
    women = _load_women_fighters()
    if women:
        df = df[~df["fighter"].str.strip().str.lower().isin(women)].copy()

    # Remove fighters inactive for more than two years (rankings only).
    # last_fight_date comes directly from round_stats.csv inside build_chin_profiles —
    # no secondary join needed, so the filter is reliable.
    if exclude_inactive and "last_fight_date" in df.columns:
        cutoff = pd.Timestamp(datetime.date.today()) - pd.DateOffset(years=2)
        last = pd.to_datetime(df["last_fight_date"], errors="coerce")
        df = df[last >= cutoff].copy()

    return df.sort_values("chin_score", ascending=False).reset_index(drop=True)


def _chin_bar(score: float) -> str:
    """Visual bar for chin score (0-1 scale → 10 blocks)."""
    filled = max(0, min(10, round(score * 10)))
    return "█" * filled + "░" * (10 - filled)


def _chin_row(rank: int, row: pd.Series) -> str:
    score    = row["chin_score"]
    surplus  = row.get("kd_surplus_adj", float("nan"))
    ko_rate  = row.get("ko_loss_rate", float("nan"))
    wc_rate  = row.get("wc_ko_rate", float("nan"))
    kd_abs   = int(row.get("kd_absorbed", 0))
    n        = int(row.get("n_fights", 0))
    wc       = str(row.get("weight_class", "")).strip()
    bar      = _chin_bar(score)

    parts = []
    if pd.notna(surplus):
        parts.append(f"Opp-adj: {surplus:+.2f} KD/fight")
    if pd.notna(ko_rate) and pd.notna(wc_rate):
        parts.append(f"KO: {ko_rate:.0%} (wc avg {wc_rate:.0%})")
    elif pd.notna(ko_rate):
        parts.append(f"KO vuln: {ko_rate:.0%}")
    parts.append(f"dropped {kd_abs}x in {n} fights")
    detail = "  ·  ".join(parts)

    wc_label = f"  <i>{_esc(wc)}</i>" if wc else ""
    return (
        f"{rank}. <b>{_esc(row['fighter'])}</b>{wc_label}  ({n} fights)\n"
        f"   {bar}  {score:.2f}\n"
        f"   <i>{_esc(detail)}</i>"
    )


_CHIN_FOOTER = (
    "<i>Chin score: 55% opponent-adjusted KD surplus + 45% KO/TKO loss rate (wc-normalised). "
    "KD surplus = actual drops absorbed minus expected given each opponent's power — "
    "getting dropped 0× against hard punchers counts more than 0× against pillow-hands. "
    "Higher = tougher chin.</i>"
)


async def bestchin_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/bestchin [n]  —  fighters hardest to knock down or stop"""
    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 30))
        except ValueError:
            pass

    profiles = _load_chin_profiles()
    if profiles.empty:
        await update.message.reply_text(
            "❌ No chin data available. Make sure <code>round_stats.csv</code> exists.",
            parse_mode="HTML",
        )
        return

    df = profiles.head(n).reset_index(drop=True)
    lines = [
        f"<b>🪨 Best chin — top {n}</b>",
        "<i>Fighters who absorb the most punishment without getting dropped or stopped</i>",
        "",
    ]
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(_chin_row(rank, row))
    lines += ["", _CHIN_FOOTER]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def worstchin_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/worstchin [n]  —  most KO-vulnerable fighters"""
    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 30))
        except ValueError:
            pass

    profiles = _load_chin_profiles()
    if profiles.empty:
        await update.message.reply_text(
            "❌ No chin data available. Make sure <code>round_stats.csv</code> exists.",
            parse_mode="HTML",
        )
        return

    df = profiles.sort_values("chin_score", ascending=True).head(n).reset_index(drop=True)
    lines = [
        f"<b>💀 Worst chin — top {n} most KO-vulnerable</b>",
        "<i>Fighters with highest knockdown rate and KO loss rate</i>",
        "",
    ]
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(_chin_row(rank, row))
    lines += ["", _CHIN_FOOTER]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def comparechin_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/comparechin Fighter A vs Fighter B"""
    raw = " ".join(context.args).strip()
    if not raw or " vs " not in raw.lower():
        await update.message.reply_text(
            "Usage: <code>/comparechin [Fighter A] vs [Fighter B]</code>\n"
            "Example: <code>/comparechin Nate Diaz vs Conor McGregor</code>",
            parse_mode="HTML",
        )
        return

    idx    = raw.lower().index(" vs ")
    name_a = raw[:idx].strip()
    name_b = raw[idx + 4:].strip()

    profiles = _load_chin_profiles(min_fights=3, exclude_inactive=False)
    if profiles.empty:
        await update.message.reply_text(
            "❌ No chin data available. Make sure <code>round_stats.csv</code> exists.",
            parse_mode="HTML",
        )
        return

    import difflib

    def _norm(n): return str(n).strip().lower()
    lookup   = {_norm(r["fighter"]): r for _, r in profiles.iterrows()}
    all_keys = list(lookup.keys())

    def _find(name: str) -> tuple[str | None, pd.Series | None]:
        key = _norm(name)
        if key in lookup:
            return lookup[key]["fighter"], lookup[key]
        matches = [(f, v) for f, v in lookup.items() if key in f or f in key]
        if matches:
            best = min(matches, key=lambda x: abs(len(x[0]) - len(key)))
            return best[1]["fighter"], best[1]
        close = difflib.get_close_matches(key, all_keys, n=1, cutoff=0.75)
        if close:
            return lookup[close[0]]["fighter"], lookup[close[0]]
        return None, None

    found_a, row_a = _find(name_a)
    found_b, row_b = _find(name_b)

    missing = []
    if row_a is None:
        missing.append(name_a)
    if row_b is None:
        missing.append(name_b)
    if missing:
        await update.message.reply_text(
            f"❌ No chin data found for: {' / '.join(missing)}\n"
            "Fighter may not have enough fights in the dataset (min 3).",
            parse_mode="HTML",
        )
        return

    def _fmt(row: pd.Series, name: str) -> list[str]:
        score    = row["chin_score"]
        surplus  = row.get("kd_surplus_adj", float("nan"))
        kd100    = row.get("kd_per_100", float("nan"))
        ko_rate  = row.get("ko_loss_rate", float("nan"))
        wc_rate  = row.get("wc_ko_rate", float("nan"))
        kd_abs   = int(row.get("kd_absorbed", 0))
        ssl_abs  = int(row.get("ssl_absorbed", 0))
        n        = int(row.get("n_fights", 0))
        wc       = str(row.get("weight_class", "")).strip()
        bar      = _chin_bar(score)

        wc_label = f"  <i>{_esc(wc)}</i>" if wc else ""
        lines = [f"<b>{_esc(name)}</b>{wc_label}  ({n} fights)"]
        lines.append(f"   {bar}  Chin score: <b>{score:.2f}</b>")
        if pd.notna(surplus):
            lines.append(f"   Opp-adj KD:  <b>{surplus:+.3f}</b>/fight  (vs expected given opponent power)")
        if pd.notna(kd100):
            lines.append(f"   KD rate:     <b>{kd100:.2f}</b> per 100 strikes absorbed")
        if pd.notna(ko_rate):
            wc_context = f"  (wc avg {wc_rate:.0%})" if pd.notna(wc_rate) else ""
            lines.append(f"   KO vuln:     <b>{ko_rate:.0%}</b> of losses by KO/TKO{_esc(wc_context)}")
        lines.append(f"   Dropped:     <b>{kd_abs}x</b> in {n} fights  ({ssl_abs} sig strikes absorbed)")
        return lines

    lines_a = _fmt(row_a, found_a)
    lines_b = _fmt(row_b, found_b)

    score_a = row_a["chin_score"]
    score_b = row_b["chin_score"]
    diff    = abs(score_a - score_b)
    if diff < 0.05:
        edge_line = "⚖️ <i>Virtually identical chin scores</i>"
    elif score_a > score_b:
        edge_line = f"🪨 <b>{_esc(found_a)}</b> has the significantly tougher chin (+{diff:.2f})"
    else:
        edge_line = f"🪨 <b>{_esc(found_b)}</b> has the significantly tougher chin (+{diff:.2f})"

    lines = (
        ["<b>🪨 Chin Comparison</b>", ""]
        + lines_a
        + [""]
        + lines_b
        + ["", edge_line, "", _CHIN_FOOTER]
    )
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# KO power ranking helpers
# ---------------------------------------------------------------------------

def _load_ko_power_profiles(min_fights: int = 5, exclude_inactive: bool = True) -> pd.DataFrame:
    """Load KO power profiles, filter women's fighters and optionally inactive fighters."""
    import datetime
    from chin import build_ko_power_profiles
    df = build_ko_power_profiles(min_fights=min_fights)
    if df.empty:
        return df

    women = _load_women_fighters()
    if women:
        df = df[~df["fighter"].str.strip().str.lower().isin(women)].copy()

    if exclude_inactive and "last_fight_date" in df.columns:
        cutoff = pd.Timestamp(datetime.date.today()) - pd.DateOffset(years=2)
        last = pd.to_datetime(df["last_fight_date"], errors="coerce")
        df = df[last >= cutoff].copy()

    return df.sort_values("power_score", ascending=False).reset_index(drop=True)


def _power_bar(score: float) -> str:
    """Visual bar for power score (0-1 → 10 blocks)."""
    filled = max(0, min(10, round(score * 10)))
    return "▓" * filled + "░" * (10 - filled)


def _power_row(rank: int, row: pd.Series) -> str:
    score     = row["power_score"]
    kd_off    = row.get("kd_per_100_off", float("nan"))
    ko_rate   = row.get("ko_win_rate", float("nan"))
    wc_rate   = row.get("wc_ko_rate", float("nan"))
    n         = int(row.get("n_fights", 0))
    wc        = str(row.get("weight_class", "")).strip()
    bar       = _power_bar(score)

    parts = []
    if pd.notna(kd_off):
        parts.append(f"KD: {kd_off:.2f}/100 str")
    if pd.notna(ko_rate) and pd.notna(wc_rate):
        parts.append(f"KO wins: {ko_rate:.0%} (wc avg {wc_rate:.0%})")
    detail = "  ·  ".join(parts)

    wc_label = f"  <i>{_esc(wc)}</i>" if wc else ""
    return (
        f"{rank}. <b>{_esc(row['fighter'])}</b>{wc_label}  ({n} fights)\n"
        f"   {bar}  {score:.2f}\n"
        f"   <i>{_esc(detail)}</i>"
    )


_POWER_FOOTER = (
    "<i>Power score: 50% offensive KD rate per 100 sig strikes + 50% KO win rate "
    "(both weight-class-normalised and Bayesian-shrunk). Higher = more dangerous finisher.</i>"
)


async def bestkopower_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/kopower [n] / /bestpower [n]  —  top-n most dangerous KO finishers"""
    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 30))
        except ValueError:
            pass

    profiles = _load_ko_power_profiles()
    if profiles.empty:
        await update.message.reply_text(
            "❌ No KO power data available. Make sure <code>round_stats.csv</code> exists.",
            parse_mode="HTML",
        )
        return

    df = profiles.head(n).reset_index(drop=True)
    lines = [
        f"<b>💥 Best striking power — top {n}</b>",
        "<i>Fighters most dangerous to face — high KD rate and KO finish rate, "
        "normalised for weight class</i>",
        "",
    ]
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(_power_row(rank, row))
    lines += ["", _POWER_FOOTER]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def worstpower_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/worstpower [n]  —  fighters who rarely finish by KO/TKO"""
    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 30))
        except ValueError:
            pass

    profiles = _load_ko_power_profiles()
    if profiles.empty:
        await update.message.reply_text(
            "❌ No KO power data available. Make sure <code>round_stats.csv</code> exists.",
            parse_mode="HTML",
        )
        return

    df = profiles.sort_values("power_score", ascending=True).head(n).reset_index(drop=True)
    lines = [
        f"<b>🧻 Weakest striking power — top {n}</b>",
        "<i>Fighters least likely to finish with strikes — low KD rate and KO win rate "
        "relative to their weight class</i>",
        "",
    ]
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(_power_row(rank, row))
    lines += ["", _POWER_FOOTER]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def comparepower_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/comparepower Fighter A vs Fighter B"""
    raw = " ".join(context.args).strip()
    if not raw or " vs " not in raw.lower():
        await update.message.reply_text(
            "Usage: <code>/comparepower [Fighter A] vs [Fighter B]</code>\n"
            "Example: <code>/comparepower Alex Pereira vs Israel Adesanya</code>",
            parse_mode="HTML",
        )
        return

    idx    = raw.lower().index(" vs ")
    name_a = raw[:idx].strip()
    name_b = raw[idx + 4:].strip()

    profiles = _load_ko_power_profiles(min_fights=3, exclude_inactive=False)
    if profiles.empty:
        await update.message.reply_text(
            "❌ No KO power data available. Make sure <code>round_stats.csv</code> exists.",
            parse_mode="HTML",
        )
        return

    import difflib

    def _norm(n): return str(n).strip().lower()
    lookup   = {_norm(r["fighter"]): r for _, r in profiles.iterrows()}
    all_keys = list(lookup.keys())

    def _find(name: str) -> tuple[str | None, pd.Series | None]:
        key = _norm(name)
        if key in lookup:
            return lookup[key]["fighter"], lookup[key]
        matches = [(f, v) for f, v in lookup.items() if key in f or f in key]
        if matches:
            best = min(matches, key=lambda x: abs(len(x[0]) - len(key)))
            return best[1]["fighter"], best[1]
        close = difflib.get_close_matches(key, all_keys, n=1, cutoff=0.75)
        if close:
            return lookup[close[0]]["fighter"], lookup[close[0]]
        return None, None

    found_a, row_a = _find(name_a)
    found_b, row_b = _find(name_b)

    missing = []
    if row_a is None: missing.append(name_a)
    if row_b is None: missing.append(name_b)
    if missing:
        await update.message.reply_text(
            f"❌ No power data found for: <b>{_esc(', '.join(missing))}</b>\n"
            "Fighter may not have enough fights in the dataset (min 3).",
            parse_mode="HTML",
        )
        return

    def _fmt_power(row: pd.Series, name: str) -> list[str]:
        score    = row["power_score"]
        kd_off   = row.get("kd_per_100_off", float("nan"))
        kd_raw   = row.get("kd_per_100_off_raw", float("nan"))
        ko_win   = row.get("ko_win_rate", float("nan"))
        ko_adj   = row.get("ko_win_rate_adj", float("nan"))
        wc_rate  = row.get("wc_ko_rate", float("nan"))
        ko_wins  = int(row.get("ko_wins", 0))
        tot_wins = int(row.get("total_wins", 0))
        n        = int(row.get("n_fights", 0))
        wc       = str(row.get("weight_class", "")).strip()
        bar      = _power_bar(score)

        wc_label = f"  <i>{_esc(wc)}</i>" if wc else ""
        lines = [f"<b>{_esc(name)}</b>{wc_label}  ({n} fights)"]
        lines.append(f"   {bar}  Power score: <b>{score:.2f}</b>")
        if pd.notna(kd_off):
            raw_str = f"  (raw {kd_raw:.2f})" if pd.notna(kd_raw) else ""
            lines.append(f"   KD rate:     <b>{kd_off:.2f}</b>/100 str{_esc(raw_str)}")
        if pd.notna(ko_win):
            wc_ctx = f"  (wc avg {wc_rate:.0%})" if pd.notna(wc_rate) else ""
            adj_str = f"  → {ko_adj:.2f}× class avg" if pd.notna(ko_adj) else ""
            lines.append(f"   KO win rate: <b>{ko_win:.0%}</b>{_esc(wc_ctx)}{_esc(adj_str)}")
        lines.append(f"   KO wins:     <b>{ko_wins}</b> of {tot_wins} career wins")
        return lines

    lines_a = _fmt_power(row_a, found_a)
    lines_b = _fmt_power(row_b, found_b)

    score_a = row_a["power_score"]
    score_b = row_b["power_score"]
    diff    = abs(score_a - score_b)
    if diff < 0.05:
        edge_line = "⚖️ <i>Virtually identical power scores</i>"
    elif score_a > score_b:
        edge_line = f"💥 <b>{_esc(found_a)}</b> is the significantly more dangerous finisher (+{diff:.2f})"
    else:
        edge_line = f"💥 <b>{_esc(found_b)}</b> is the significantly more dangerous finisher (+{diff:.2f})"

    lines = (
        ["<b>💥 Striking Power Comparison</b>", ""]
        + lines_a
        + [""]
        + lines_b
        + ["", edge_line, "", _POWER_FOOTER]
    )
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help"""
    await update.message.reply_text(
        "<b>UFC Matchup Predictor</b>\n\n"
        "<b>/predict</b> [Fighter A] <b>vs</b> [Fighter B]\n"
        "  Predicts the winner with calibrated confidence, Glicko-2 ratings, "
        "key factors, and an optional AI fight preview.\n\n"
        "<b>/scan</b> [event]\n"
        "  Scans a UFC card on Polymarket and flags fights where the model diverges "
        "from the market price — potential +EV opportunities.\n"
        "  Example: <code>/scan UFC 314</code>  or just <code>/scan</code> for open markets.\n\n"
        "<b>/top</b> [n]\n"
        "  Top-n fighters by current Glicko-2 skill rating (default 10, max 30).\n\n"
        "<b>/bestcardio</b> [n]\n"
        "  Fighters whose striking output, accuracy, and TD defense hold up best round-to-round (default 10, max 30).\n\n"
        "<b>/worstcardio</b> [n]\n"
        "  Fighters who fade the most as fights go deep (default 10, max 30).\n\n"
        "<b>/comparecardio</b> [Fighter A] <b>vs</b> [Fighter B]\n"
        "  Side-by-side cardio breakdown of two fighters across all round-curve metrics.\n\n"
        "<b>/bestchin</b> [n]\n"
        "  Fighters who absorb the most punishment without getting dropped or stopped (default 10, max 30).\n\n"
        "<b>/worstchin</b> [n]\n"
        "  Most KO-vulnerable fighters — high KD rate and/or high KO loss rate (default 10, max 30).\n\n"
        "<b>/comparechin</b> [Fighter A] <b>vs</b> [Fighter B]\n"
        "  Side-by-side chin/durability breakdown — opponent-adjusted KD surplus, KO vulnerability, career context.\n\n"
        "<b>/bestpower</b> [n]\n"
        "  Most dangerous KO finishers — ranked by offensive KD rate and KO win rate, both weight-class-normalised (default 10, max 30).\n\n"
        "<b>/worstpower</b> [n]\n"
        "  Fighters who almost never finish by KO/TKO — lowest offensive KD and KO win rate relative to their division (default 10, max 30).\n\n"
        "<b>/comparepower</b> [Fighter A] <b>vs</b> [Fighter B]\n"
        "  Side-by-side KO power breakdown — offensive KD rate, KO win rate, weight-class context.\n\n"
        "<b>/stats</b>\n"
        "  Model accuracy, dataset size, and feature summary.\n\n"
        "<b>/update</b>  <i>(admin)</i>\n"
        "  Scrape the latest UFC results, re-engineer features, and retrain the model.\n\n"
        "<i>Fighter names are fuzzy-matched — small typos are fine.</i>\n\n"
        "Examples:\n"
        "<code>/predict Khabib Nurmagomedov vs Conor McGregor</code>\n"
        "<code>/comparechin Nate Diaz vs Conor McGregor</code>\n"
        "<code>/scan UFC 314</code>",
        parse_mode="HTML",
    )


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start"""
    await help_handler(update, context)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_BOT_COMMANDS = [
    BotCommand("predict",        "Predict a matchup — /predict Fighter A vs Fighter B"),
    BotCommand("scan",           "Scan a UFC card for +EV Polymarket bets — /scan UFC 314"),
    BotCommand("top",            "Top fighters by Glicko-2 rating — /top 10"),
    BotCommand("bestcardio",     "Fighters with best round-to-round cardio — /bestcardio 10"),
    BotCommand("worstcardio",    "Fighters who fade most late in fights — /worstcardio 10"),
    BotCommand("comparecardio",  "Compare two fighters' cardio — /comparecardio A vs B"),
    BotCommand("bestchin",       "Fighters hardest to knock down or stop — /bestchin 10"),
    BotCommand("worstchin",      "Most KO-vulnerable fighters — /worstchin 10"),
    BotCommand("comparechin",    "Compare two fighters' chin/durability — /comparechin A vs B"),
    BotCommand("kopower",        "Most dangerous KO finishers — /kopower 10"),
    BotCommand("bestpower",      "Most dangerous KO finishers — /bestpower 10"),
    BotCommand("worstpower",     "Fighters who rarely finish by KO — /worstpower 10"),
    BotCommand("comparepower",   "Compare two fighters' KO power — /comparepower A vs B"),
    BotCommand("stats",          "Model accuracy and dataset info"),
    BotCommand("update",         "Scrape new fights and retrain (admin)"),
    BotCommand("help",           "Show command guide"),
]


async def _set_commands(app: Application) -> None:
    await app.bot.set_my_commands(_BOT_COMMANDS)
    log.info("Bot command menu registered (%d commands)", len(_BOT_COMMANDS))


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN is not set.")
        print("Get a token from @BotFather, then run:")
        print("  export TELEGRAM_BOT_TOKEN=<your_token>")
        sys.exit(1)

    app = (
        Application.builder()
        .token(token)
        .post_init(_set_commands)
        .build()
    )
    app.add_handler(CommandHandler("start",       start_handler))
    app.add_handler(CommandHandler("help",        help_handler))
    app.add_handler(CommandHandler("predict",     predict_handler))
    app.add_handler(CommandHandler("scan",        scan_handler))
    app.add_handler(CommandHandler("top",            top_handler))
    app.add_handler(CommandHandler("bestcardio",     bestcardio_handler))
    app.add_handler(CommandHandler("worstcardio",    worstcardio_handler))
    app.add_handler(CommandHandler("comparecardio",  comparecardio_handler))
    app.add_handler(CommandHandler("bestchin",       bestchin_handler))
    app.add_handler(CommandHandler("worstchin",      worstchin_handler))
    app.add_handler(CommandHandler("comparechin",    comparechin_handler))
    app.add_handler(CommandHandler("kopower",        bestkopower_handler))
    app.add_handler(CommandHandler("bestpower",      bestkopower_handler))
    app.add_handler(CommandHandler("worstpower",     worstpower_handler))
    app.add_handler(CommandHandler("comparepower",   comparepower_handler))
    app.add_handler(CommandHandler("stats",          stats_handler))
    app.add_handler(CommandHandler("update",         update_handler))

    log.info("Bot started — polling for updates")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
