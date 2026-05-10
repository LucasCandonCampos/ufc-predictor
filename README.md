# UFC Fight Predictor

A machine learning system that predicts UFC fight outcomes, explains its reasoning with SHAP values, and surfaces betting edges against Polymarket odds — all accessible through a Telegram bot.

## What It Does

- **Predicts fight winners** with 67.7% accuracy on held-out test data (2023–2026)
- **Predicts finish method** (KO/TKO, Submission, Decision) at 55.6% accuracy
- **Predicts finish round** at 50.0% accuracy
- **Explains each prediction** using SHAP feature attributions rendered in plain English
- **Scans for +EV bets** by comparing model probabilities against Polymarket implied odds
- **Ranks fighters** on durability (chin), KO power, and cardio/late-round output
- **Updates itself** — scrapes new UFC events, retrains models, reloads live without a restart

---

## Architecture

```
UFC Stats website
      │
      ▼
 scraper.py          ← scrapes fight-by-fight stats, per-round data, career averages
      │
      ▼
ufc-master.csv       ← 7,253 fights (2010–present), one row per fight
round_stats.csv      ← per-round sig strikes, TDs, KDs for every recorded fight
      │
      ├──► cluster.py      ← KMeans style clustering → 4 archetypes per fighter
      ├──► fatigue.py      ← round-curve regression → cardio/fade profiles
      ├──► chin.py         ← Bayesian opponent-adjusted chin + KO power scores
      │
      ▼
 features.py         ← engineers 65 matchup-level delta features, Glicko-2 ratings
      │
      ▼
matchup_features.csv ← 14,506 rows (each fight doubled: A vs B and B vs A)
      │
      ▼
  train.py           ← XGBoost + isotonic calibration; 3 separate models
      │
      ▼
 model.pkl           ← winner prediction
 method_model.pkl    ← KO / Sub / Decision
 round_model.pkl     ← finish round
      │
      ▼
  predict.py         ← loads models, builds feature vector, runs SHAP
      │
      ▼
   bot.py            ← Telegram interface
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Scraping | `requests`, `beautifulsoup4` |
| Data | `pandas`, `numpy` |
| ML models | `xgboost` |
| Calibration | `scikit-learn` (isotonic regression) |
| Explainability | `shap` |
| Fighter ratings | Glicko-2 (custom implementation) |
| Style clustering | KMeans (`scikit-learn`) |
| Fuzzy name matching | `rapidfuzz` |
| Betting markets | Polymarket API (custom client) |
| LLM summaries | `anthropic` (Claude, optional) |
| Bot | `python-telegram-bot` |

---

## Pipeline

Each step is a standalone script that reads from and writes to disk.

### Step 1 — Scrape (`scraper.py`)
Fetches all UFC events from [ufcstats.com](http://ufcstats.com), parses fighter stats per fight and per round, and appends new rows to `data/raw/ufc-master.csv`. Correctly assigns the winner as R_fighter and the loser as B_fighter using the W/L status flags on each fight detail page — not by assuming the event listing order.

### Step 2 — Feature Engineering (`features.py`)
Builds `data/matchup_features.csv`. Every fight becomes two symmetric rows (A vs B, B vs A) to give the model both perspectives. Key operations:

- Computes exponentially time-decayed career stat averages (recent fights weighted more heavily, decay rate 0.13/year)
- Assigns each fighter a combat style via KMeans clustering (see below)
- Computes historical style-matchup win rates in strict date order to prevent leakage
- Runs a full Glicko-2 rating system across the dataset
- Computes round-by-round output degradation curves from `round_stats.csv`
- Loads Bayesian opponent-adjusted chin and KO power scores from `chin.py`
- Outputs 65 delta features (Fighter A minus Fighter B) per matchup

### Step 3 — Train (`train.py`)
Three XGBoost models trained on a time-based split (no data leakage):

- **Train** set: fights up to ~2022
- **Calibration** set: ~2022–2023 (isotonic regression to fix probability overconfidence)
- **Test** set: ~2023–2026 (never touched during training)

---

## Feature Engineering Detail

### Combat Style Clustering
KMeans (k=4) on career striking accuracy, takedown accuracy, takedown volume, and submission attempts. Each fighter is assigned one of:

| Style | Description |
|---|---|
| **Pressure Striker** | High-output aggressive striker |
| **Volume Striker** | Technical striker, high accuracy |
| **Wrestler/Grappler** | Takedown-heavy, submission threats |
| **Well-Rounded** | No dominant dimension |

Historical style-matchup win rates (e.g. "Pressure Strikers win 58% vs Wrestlers") are computed from the full dataset and used as a direct feature.

### Glicko-2 Rating System
Each fighter has a rating, rating deviation (RD), and volatility that update after every fight. RD captures uncertainty — a fighter on a long layoff has higher RD. The `glicko_rating_delta` feature captures the current skill gap between two fighters.

### Cardio / Late-Round Degradation (`fatigue.py`)
Linear regression of significant strikes landed vs. round number for each fighter, fit on their per-round history. Features include:

- `output_drop_per_rd` — slope of strike output across rounds (negative = fades)
- `accuracy_drop_per_rd` — slope of striking accuracy across rounds
- `degradation_score` — composite: positive = gets stronger late, negative = fades badly
- `champ_round_activity` — output specifically in rounds 4–5 (championship rounds)

### Chin / Durability (`chin.py`)
Opponent-quality-adjusted knockdown absorption. Rather than raw KDs absorbed per strike, the model computes how many KDs a fighter *should* have absorbed given who they fought:

1. Each opponent's offensive KD rate is Bayesian-shrunk toward the pool mean (K=3 fights prior) to prevent small-sample outliers from dominating
2. `kd_surplus_per_fight` = actual KDs absorbed − expected KDs given opponent KD rates
3. `kd_surplus_adj` further shrinks toward 0 by fight count (fighters with fewer bouts are regressed to neutral)
4. Combined with weight-class-normalised KO loss rate into a composite `chin_score`

### KO Power (`chin.py`)
Symmetric to chin, measuring offensive danger:

- `kd_per_100_off` — knockdowns landed per 100 significant strikes thrown
- `ko_win_rate_adj` — fraction of wins by KO/TKO, normalised by weight-class KO frequency (Heavyweights KO at 51.8%, Flyweights at 23.6%)
- `power_score` — composite percentile (50% offensive KD rate + 50% KO win rate)

### Full Feature List (65 features)

| Category | Features |
|---|---|
| Striking | Accuracy delta, volume delta |
| Grappling | Takedown accuracy delta, TD volume delta, submission attempts delta |
| Finishing | KO finish rate delta |
| Physical | Reach delta, age delta |
| Experience | Fight count delta, days since last fight delta |
| Ratings | Glicko-2 rating delta, RD delta |
| Style | Style archetype (one-hot), historical style-matchup win rate |
| Cardio | Output drop/rd delta, accuracy drop/rd delta, late-round output delta, fade score delta, championship round output delta |
| Chin | Composite chin score delta, opp-adjusted KD surplus delta, KO vulnerability delta |
| KO Power | Composite power score delta, offensive KD rate delta, KO win rate delta |
| Context | Weight class (ordinal), low-data flags |

---

## Model Performance

All metrics on the held-out test set (2023–2026, ~2,888 fights, never used during training or calibration).

| Model | Task | Test Accuracy | ROC-AUC |
|---|---|---|---|
| `model.pkl` | Winner prediction | **67.7%** | 0.729 |
| `method_model.pkl` | KO/Sub/Decision | **55.6%** | — |
| `round_model.pkl` | Finish round | **50.0%** | — |

Time-series cross-validation (5 expanding folds) shows consistent improvement as the dataset grows, from 62.5% (2017 data) to 68.6% (2023+ data), reflecting both more data and better features.

Calibration is handled via isotonic regression on a held-out calibration set. The model's stated probabilities match observed win rates within ±3% across the full probability range.

---

## EV Scanner (`ev_scanner.py`)

Compares model win probabilities against Polymarket implied odds for upcoming UFC fights. Flags bets where:

- **Model edge** > 12% (model probability minus market implied probability)
- **Market price** for our pick is between 45–70% (backtests show negative ROI outside this range)

Bet sizing uses quarter-Kelly criterion to account for model uncertainty. The scanner pulls live Polymarket markets via the Gamma API, fuzzy-matches fighter names, and returns ranked opportunities with Kelly-recommended bet sizes.

---

## Telegram Bot Commands

| Command | Description |
|---|---|
| `/predict [A] vs [B]` | Full matchup prediction with SHAP explanation and finish forecast |
| `/top [n]` | Top-n fighters by Glicko-2 rating (default 10) |
| `/stats` | Model accuracy, dataset size, last update date |
| `/scan [event]` | Scan Polymarket for +EV bets on an upcoming card |
| `/bestchin [n]` | Top-n most durable fighters by opponent-adjusted chin score |
| `/worstchin [n]` | Most KO-vulnerable fighters |
| `/comparechin A vs B` | Side-by-side chin/durability comparison |
| `/bestpower [n]` | Top-n hardest punchers by KO power score |
| `/worstpower [n]` | Fighters with least KO power |
| `/comparepower A vs B` | Side-by-side KO power comparison |
| `/bestcardio [n]` | Fighters who perform best in late rounds |
| `/worstcardio [n]` | Fighters who fade the most |
| `/comparecardio A vs B` | Side-by-side cardio/fade comparison |
| `/update` | Scrape new fights, rebuild features, retrain (admin only) |
| `/help` | Full command reference |

---

## Setup

### Requirements
- Python 3.11+
- A Telegram bot token ([BotFather](https://t.me/botfather))
- Anthropic API key (optional — enables plain-English LLM summaries)

### Install

```bash
git clone https://github.com/LucasCandonCampos/ufc-predictor.git
cd ufc-predictor
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in:
#   TELEGRAM_BOT_TOKEN=...
#   ANTHROPIC_API_KEY=...   (optional)
#   ADMIN_CHAT_ID=...        (optional — restricts /update to one user)
```

### Run the full pipeline (first time)

```bash
python src/scraper.py data/raw/ufc-master.csv   # fetch fight data
python src/cluster.py data/raw/ufc-master.csv   # fit style clusters
python src/features.py data/raw/ufc-master.csv  # engineer features
python src/train.py                              # train models
python src/bot.py                                # start bot
```

If you already have a trained model, just run `python src/bot.py`. The `/update` command handles the full pipeline automatically from within Telegram.

---

## Project Structure

```
ufc-predictor/
├── src/
│   ├── scraper.py        # UFC Stats scraper + career stat accumulation
│   ├── cluster.py        # KMeans fight-style clustering
│   ├── fatigue.py        # Round-curve regression, cardio/fade profiling
│   ├── chin.py           # Opponent-adjusted chin + KO power scoring
│   ├── features.py       # Full feature engineering pipeline
│   ├── train.py          # XGBoost training + calibration
│   ├── predict.py        # Inference + SHAP explanations
│   ├── ev_scanner.py     # Polymarket EV scanner
│   ├── polymarket.py     # Polymarket API client
│   ├── calibration.py    # Custom calibrated model wrapper
│   ├── bot.py            # Telegram bot
│   └── round_scraper.py  # Standalone per-round stats scraper
├── models/
│   ├── model.pkl          # Winner prediction model
│   ├── method_model.pkl   # Finish method model
│   ├── round_model.pkl    # Finish round model
│   ├── style_kmeans.pkl   # Style cluster model
│   └── style_scaler.pkl   # Feature scaler for clustering
├── data/
│   ├── raw/ufc-master.csv       # Full fight dataset (7,253 fights)
│   ├── round_stats.csv          # Per-round stats for all fights
│   ├── matchup_features.csv     # Engineered training features
│   ├── fighter_ratings.csv      # Glicko-2 ratings
│   ├── fighter_fatigue.csv      # Cardio/fade profiles
│   ├── fighters_with_style.csv  # Style assignments
│   └── fighter_ew_stats.csv     # Exponentially-weighted career averages
├── .env.example
├── requirements.txt
└── backtest.py
```

---

## Data Source

Fight statistics sourced from [UFCStats.com](http://ufcstats.com), covering 7,253 fights from 2010 to present. The dataset includes cumulative career averages pre-computed up to each fight date (no leakage), per-round breakdowns, physical attributes, and fight outcomes.
