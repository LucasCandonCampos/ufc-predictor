"""
Step 3: Train XGBoost binary classifier on matchup features.

Split is date-based (70 / 10 / 20):
  - Train  (0–70 pct): XGBoost fitting
  - Cal   (70–80 pct): isotonic calibration — never seen by the base model
  - Test  (80–100 pct): final evaluation

Usage:
    python src/train.py

Reads:
    data/matchup_features.csv

Outputs:
    models/model.pkl             — {"model": CalibratedClassifierCV, "feature_cols": list, ...}
    data/feature_importance.png
    data/calibration.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from calibration import _CalibratedModel

ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_CSV = os.path.join(DATA_DIR, "matchup_features.csv")

EXCLUDE_COLS = {
    "fight_id", "a_fighter", "b_fighter", "date", "weight_class",
    "a_won", "a_style", "b_style", "style_matchup",
}

XGBOOST_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

CV_FOLDS   = 5
CAL_METHOD = "isotonic"


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_features() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"Loaded {len(df):,} rows  |  {len(feature_cols)} feature columns")
    return df, feature_cols


def date_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    test_frac:  float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Three-way chronological split.
      Train  [0, train_frac]       — XGBoost fitting
      Cal    (train_frac, test_frac] — isotonic calibration (never seen by base model)
      Test   (test_frac, 1]         — final evaluation
    """
    train_cut = df["date"].quantile(train_frac)
    test_cut  = df["date"].quantile(test_frac)

    train = df[df["date"] <= train_cut].copy()
    cal   = df[(df["date"] > train_cut) & (df["date"] <= test_cut)].copy()
    test  = df[df["date"] >  test_cut].copy()

    print(
        f"Train: {len(train):,} rows  ({train['date'].min().date()} → {train['date'].max().date()})\n"
        f"Cal:   {len(cal):,}  rows  ({cal['date'].min().date()} → {cal['date'].max().date()})\n"
        f"Test:  {len(test):,}  rows  ({test['date'].min().date()} → {test['date'].max().date()})\n"
        f"Cutoffs: {train_cut.date()} / {test_cut.date()}"
    )
    return train, cal, test, train_cut, test_cut


# ---------------------------------------------------------------------------
# Time-series cross-validation  (diagnostic — does not change the saved model)
# ---------------------------------------------------------------------------

def time_series_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = CV_FOLDS,
) -> list[dict]:
    """
    Expanding-window time-series CV across the full dataset.
    Each fold trains on everything up to a cutoff, validates on the next window.
    The final fold's validation window matches the held-out test set (80–100 pct).

    Purely diagnostic: used to estimate variance in accuracy across time, not
    to select the model or tune hyperparameters.
    """
    # n_folds+1 evenly spaced quantiles; last training cutoff = 80 pct
    train_fracs = np.linspace(0.40, 0.80, n_folds)
    val_fracs   = list(np.linspace(0.40, 0.80, n_folds)[1:]) + [1.0]

    train_cuts = [df["date"].quantile(q) for q in train_fracs]
    val_cuts   = [df["date"].quantile(q) for q in val_fracs]

    print(f"\n{'─' * 62}")
    print(f"TIME-SERIES CV  ({n_folds} expanding folds — diagnostic only)")
    print(f"{'─' * 62}")
    print(f"  {'Fold':<5}  {'Train →':<14}  {'Val window':<26}  {'Acc':>6}  {'AUC':>7}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*26}  {'─'*6}  {'─'*7}")

    results = []
    for i in range(n_folds):
        tr = df[df["date"] <= train_cuts[i]]
        va = df[(df["date"] > train_cuts[i]) & (df["date"] <= val_cuts[i])]

        if len(tr) < 500 or len(va) < 100:
            continue

        m = xgb.XGBClassifier(**XGBOOST_PARAMS)
        m.fit(tr[feature_cols], tr["a_won"].values)

        y_va  = va["a_won"].values
        prob  = m.predict_proba(va[feature_cols])[:, 1]
        acc   = accuracy_score(y_va, (prob >= 0.5).astype(int))
        auc   = roc_auc_score(y_va, prob)
        results.append({"fold": i + 1, "accuracy": acc, "roc_auc": auc})

        val_range = f"{train_cuts[i].date()} → {val_cuts[i].date()}"
        print(f"  {i+1:<5}  {train_cuts[i].date()!s:<14}  {val_range:<26}  "
              f"{acc:>5.1%}  {auc:>7.4f}")

    if results:
        accs = [r["accuracy"] for r in results]
        aucs = [r["roc_auc"]  for r in results]
        print(f"  {'─'*5}  {'─'*14}  {'─'*26}  {'─'*6}  {'─'*7}")
        print(f"  {'Mean':<5}  {'':14}  {'':26}  {np.mean(accs):>5.1%}  {np.mean(aucs):>7.4f}")
        print(f"  {'±std':<5}  {'':14}  {'':26}  {np.std(accs):>5.1%}  {np.std(aucs):>7.4f}")

    return results


# ---------------------------------------------------------------------------
# Training & calibration
# ---------------------------------------------------------------------------

def train_model(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> xgb.XGBClassifier:
    X = train[feature_cols]
    y = train["a_won"].values
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)
    print(f"Training complete  ({model.n_estimators} trees, max_depth={model.max_depth})")
    return model


def calibrate_model(
    model: xgb.XGBClassifier,
    cal: pd.DataFrame,
    feature_cols: list[str],
) -> _CalibratedModel:
    """
    Post-hoc isotonic calibration on the held-out calibration set.
    The base XGBoost is never retrained — calibration is a monotone mapping
    applied on top of its raw probabilities.
    """
    X_cal = cal[feature_cols]
    y_cal = cal["a_won"].values
    raw_probs = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)
    print(f"Calibration complete  (method=isotonic, cal_rows={len(cal):,})")
    return _CalibratedModel(model, iso)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_raw: xgb.XGBClassifier,
    model_cal: _CalibratedModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    print("\n" + "=" * 58)
    print("EVALUATION RESULTS")
    print("=" * 58)

    results = {}
    X_test = test[feature_cols]
    y_test = test["a_won"].values

    raw_prob = model_raw.predict_proba(X_test)[:, 1]
    cal_prob = model_cal.predict_proba(X_test)[:, 1]
    raw_pred = (raw_prob >= 0.5).astype(int)
    cal_pred = (cal_prob >= 0.5).astype(int)

    # Training metrics (raw model on training set)
    X_train = train[feature_cols]
    y_train = train["a_won"].values
    tr_prob = model_raw.predict_proba(X_train)[:, 1]
    tr_pred = (tr_prob >= 0.5).astype(int)

    print(f"\nTRAIN  (raw XGBoost, {len(train):,} rows)")
    print(f"  Accuracy : {accuracy_score(y_train, tr_pred):.4f}  ({accuracy_score(y_train, tr_pred):.1%})")
    print(f"  ROC-AUC  : {roc_auc_score(y_train, tr_prob):.4f}")
    print(f"  Log Loss : {log_loss(y_train, tr_prob):.4f}")

    print(f"\nTEST   (raw XGBoost)   |   CALIBRATED (isotonic)")
    print(f"  Accuracy : {accuracy_score(y_test, raw_pred):.1%}                   {accuracy_score(y_test, cal_pred):.1%}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, raw_prob):.4f}                {roc_auc_score(y_test, cal_prob):.4f}")
    print(f"  Log Loss : {log_loss(y_test, raw_prob):.4f}                {log_loss(y_test, cal_prob):.4f}")
    print(f"  Brier    : {brier_score_loss(y_test, raw_prob):.4f}                {brier_score_loss(y_test, cal_prob):.4f}")

    test_acc = accuracy_score(y_test, cal_pred)
    if test_acc >= 0.62:
        verdict = f"PASS — exceeds 62% target by {test_acc - 0.62:.1%}"
    elif test_acc >= 0.60:
        verdict = f"PASS — meets >60% target"
    else:
        verdict = f"BELOW TARGET — {test_acc:.1%} < 60%"
    print(f"\nTarget check: {verdict}")

    results["test"] = {
        "accuracy": test_acc,
        "roc_auc":  roc_auc_score(y_test, cal_prob),
        "log_loss": log_loss(y_test, cal_prob),
        "brier":    brier_score_loss(y_test, cal_prob),
    }
    return results


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _unwrap(model) -> xgb.XGBClassifier:
    """Return the underlying XGBoost from any calibration wrapper."""
    return getattr(model, "estimator", model)


def save_feature_importance(
    model,
    feature_cols: list[str],
) -> None:
    xgb_model = _unwrap(model)
    scores = xgb_model.get_booster().get_score(importance_type="gain")
    importance = (
        pd.Series(scores, name="gain")
          .reindex(feature_cols)
          .fillna(0)
          .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, max(6, len(importance) * 0.3)))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance (gain)")
    ax.set_xlabel("Gain")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(DATA_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature importance chart → {path}")

    print("\nTop 10 features by gain:")
    for feat, val in importance.sort_values(ascending=False).head(10).items():
        print(f"  {feat:<40}  {val:.1f}")


def save_calibration_plot(
    model_raw: xgb.XGBClassifier,
    model_cal: _CalibratedModel,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    X_test = test[feature_cols]
    y_test = test["a_won"].values

    fig, ax = plt.subplots(figsize=(6, 6))
    for model, label, color in [
        (model_raw, "XGBoost (raw)",          "steelblue"),
        (model_cal, "Calibrated (isotonic)",   "orangered"),
    ]:
        prob = model.predict_proba(X_test)[:, 1]
        pt, pp = calibration_curve(y_test, prob, n_bins=10, strategy="uniform")
        ax.plot(pp, pt, marker="o", label=label, color=color)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual win rate)")
    ax.set_title("Calibration: Before vs After Isotonic Correction")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(DATA_DIR, "calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Calibration plot → {path}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model_cal: _CalibratedModel,
    feature_cols: list[str],
    test_cut: pd.Timestamp,
    results: dict,
) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {
        "model":         model_cal,    # CalibratedClassifierCV; predict_proba() is calibrated
        "feature_cols":  feature_cols,
        "train_cutoff":  str(test_cut.date()),
        "test_accuracy": results["test"]["accuracy"],
        "test_roc_auc":  results["test"]["roc_auc"],
        "test_brier":    results["test"]["brier"],
    }
    path = os.path.join(MODELS_DIR, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df, feature_cols = load_features()
    train, cal, test, train_cut, test_cut = date_split(df)

    print("\nRunning time-series CV (diagnostic)...")
    time_series_cv(df, feature_cols)

    print("\nTraining XGBoost...")
    model_raw = train_model(train, feature_cols)

    print("\nCalibrating probabilities...")
    model_cal = calibrate_model(model_raw, cal, feature_cols)

    results = evaluate(model_raw, model_cal, train, test, feature_cols)

    print("\nGenerating charts...")
    save_feature_importance(model_raw, feature_cols)
    save_calibration_plot(model_raw, model_cal, test, feature_cols)

    save_model(model_cal, feature_cols, test_cut, results)
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
