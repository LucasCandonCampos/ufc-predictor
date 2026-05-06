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
    /update                               — scrape new fights and retrain (admin only)
    /help                                 — usage guide
"""

import asyncio
import logging
import os
import sys

import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

sys.path.insert(0, os.path.dirname(__file__))
from predict import predict_matchup, _factor_line

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

ROOT     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")


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


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help"""
    await update.message.reply_text(
        "<b>UFC Matchup Predictor</b>\n\n"
        "<b>/predict</b> [Fighter A] <b>vs</b> [Fighter B]\n"
        "  Predicts the winner with calibrated confidence, Glicko-2 ratings, "
        "key factors, and an optional AI fight preview.\n\n"
        "<b>/top</b> [n]\n"
        "  Top-n fighters by current Glicko-2 skill rating (default 10, max 30).\n\n"
        "<b>/stats</b>\n"
        "  Model accuracy, dataset size, and feature summary.\n\n"
        "<b>/update</b>  <i>(admin)</i>\n"
        "  Scrape the latest UFC results, re-engineer features, and retrain the model.\n\n"
        "<i>Fighter names are fuzzy-matched — small typos are fine.</i>\n\n"
        "Example:\n"
        "<code>/predict Khabib Nurmagomedov vs Conor McGregor</code>",
        parse_mode="HTML",
    )


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start"""
    await help_handler(update, context)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN is not set.")
        print("Get a token from @BotFather, then run:")
        print("  export TELEGRAM_BOT_TOKEN=<your_token>")
        sys.exit(1)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start",   start_handler))
    app.add_handler(CommandHandler("help",    help_handler))
    app.add_handler(CommandHandler("predict", predict_handler))
    app.add_handler(CommandHandler("top",     top_handler))
    app.add_handler(CommandHandler("stats",   stats_handler))
    app.add_handler(CommandHandler("update",  update_handler))

    log.info("Bot started — polling for updates")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
