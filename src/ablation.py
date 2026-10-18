"""
Ablation study: does FinBERT sentiment add predictive value over price features?

Design (interim report, Sections 3.6-3.8):
  - Per ticker x horizon (1, 3, 5 trading days) and per architecture (XGBoost,
    LSTM), train matched models on
      price                        : technical price/volume features only
      price+sentiment              : + daily FinBERT features from all tagged news
      price+sentiment+company      : + features from headlines naming the company
    Hyperparameters, seeds and data splits are identical across feature sets.
  - Walk-forward evaluation: test on 2021, 2022, 2023, each time training on all
    earlier data, with an h-day purge gap so no training label overlaps the test year.
  - Classification target: down / stable / up. The "stable" band is the 33rd
    percentile of |forward return| in each training window (no test information).
  - Regression target: forward return (RMSE, plus directional accuracy).
  - Significance: paired moving-block bootstrap of the metric difference between
    each sentiment set and the price-only model on the same test days.

Usage:
    python -m src.ablation                  # XGBoost and LSTM
    python -m src.ablation --model lstm
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier, XGBRegressor

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RANDOM_SEED, RESULTS_DIR, STUDY_TICKERS
from src.features import (
    COMPANY_SENTIMENT_FEATURES,
    HORIZONS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    build_all,
)

OUTPUT_DIR = RESULTS_DIR / "ablation"
TEST_YEARS = [2021, 2022, 2023]
CLASS_NAMES = ["down", "stable", "up"]

FEATURE_SETS: Dict[str, List[str]] = {
    "price": PRICE_FEATURES,
    "price+sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
    "price+sentiment+company": PRICE_FEATURES + SENTIMENT_FEATURES + COMPANY_SENTIMENT_FEATURES,
}

# Fixed, conservative settings shared by every model (no per-set tuning)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_BLOCK = 10  # trading days; respects autocorrelation of overlapping targets


def label_movement(returns: np.ndarray, band: float) -> np.ndarray:
    """0 = down, 1 = stable, 2 = up."""
    return np.where(returns > band, 2, np.where(returns < -band, 0, 1))


def walk_forward_splits(
    dates: pd.Series, horizon: int, test_years: List[int] = TEST_YEARS
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """(test_year, train_idx, test_idx); drops the last `horizon` training rows (purge)."""
    years = dates.dt.year.values
    splits = []
    for year in test_years:
        train_idx = np.where(years < year)[0]
        train_idx = train_idx[:-horizon] if horizon > 0 else train_idx
        test_idx = np.where(years == year)[0]
        if len(train_idx) and len(test_idx):
            splits.append((year, train_idx, test_idx))
    return splits


def xgboost_fit_predict(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: List[str],
    target: str,
    y_train_cls: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[float]]]:
    """Gradient-boosted trees: classifier + regressor. Returns (class, return %, gain)."""
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    clf = XGBClassifier(objective="multi:softprob", num_class=3, **XGB_PARAMS)
    clf.fit(train[features], y_train_cls)
    reg = XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
    reg.fit(train[features], train[target] * 100)  # percent units
    gain = clf.get_booster().get_score(importance_type="gain")
    return (
        clf.predict(test[features]),
        reg.predict(test[features]),
        [gain.get(f, 0.0) for f in features],
    )


def _lstm_fit_predict(*args, **kwargs):
    from src.lstm_model import lstm_fit_predict  # imports torch only when needed

    return lstm_fit_predict(*args, **kwargs)


MODEL_RUNNERS: Dict[str, Callable] = {
    "xgboost": xgboost_fit_predict,
    "lstm": _lstm_fit_predict,
    "lstm_balanced": lambda *args: _lstm_fit_predict(*args, class_weighted=True),
}


def run_ticker_horizon(
    df: pd.DataFrame,
    horizon: int,
    model_name: str = "xgboost",
    feature_sets: Optional[Dict[str, List[str]]] = None,
    test_years: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample predictions for every feature set, and feature importances."""
    fit_predict = MODEL_RUNNERS[model_name]
    target = f"fwd_ret_{horizon}"
    data = df.dropna(subset=[target]).reset_index(drop=True)

    pred_rows, importance_rows = [], []
    feature_sets = feature_sets or FEATURE_SETS
    for year, train_idx, test_idx in walk_forward_splits(data["date"], horizon,
                                                         test_years or TEST_YEARS):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        band = float(np.quantile(np.abs(train[target]), 1 / 3))
        y_train_cls = label_movement(train[target].values, band)
        y_test_cls = label_movement(test[target].values, band)
        majority = int(np.bincount(y_train_cls, minlength=3).argmax())

        for set_name, features in feature_sets.items():
            y_pred, ret_pred, importance = fit_predict(
                data, train_idx, test_idx, features, target, y_train_cls, horizon
            )
            pred_rows.append(
                pd.DataFrame(
                    {
                        "date": test["date"].values,
                        "test_year": year,
                        "feature_set": set_name,
                        "y_true": y_test_cls,
                        "y_pred": y_pred,
                        "y_majority": majority,
                        "ret_true": test[target].values * 100,
                        "ret_pred": ret_pred,
                        "stable_band_pct": band * 100,
                    }
                )
            )
            if importance is not None:
                importance_rows.append(
                    pd.DataFrame({"test_year": year, "feature_set": set_name,
                                  "feature": features, "importance_gain": importance})
                )
    importance_df = (pd.concat(importance_rows, ignore_index=True) if importance_rows
                     else pd.DataFrame())
    return pd.concat(pred_rows, ignore_index=True), importance_df


# ---------------------------------------------------------------------------
# Metrics and significance
# ---------------------------------------------------------------------------


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    kw = dict(labels=[0, 1, 2], average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, **kw),
        "recall_macro": recall_score(y_true, y_pred, **kw),
        "f1_macro": f1_score(y_true, y_pred, **kw),
    }


def regression_metrics(ret_true, ret_pred) -> Dict[str, float]:
    ret_true, ret_pred = np.asarray(ret_true), np.asarray(ret_pred)
    return {
        "rmse": float(np.sqrt(np.mean((ret_true - ret_pred) ** 2))),
        "rmse_zero_forecast": float(np.sqrt(np.mean(ret_true**2))),
        "directional_accuracy": float(np.mean(np.sign(ret_pred) == np.sign(ret_true))),
    }


def _macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def block_bootstrap_delta(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_new: np.ndarray,
    metric: Callable,
    block: int = BOOTSTRAP_BLOCK,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """Paired moving-block bootstrap of metric(new) - metric(base) on the same test days."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_blocks = int(np.ceil(n / block))
    deltas = np.empty(n_samples)
    for i in range(n_samples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        deltas[i] = metric(y_true[idx], pred_new[idx]) - metric(y_true[idx], pred_base[idx])
    observed = metric(y_true, pred_new) - metric(y_true, pred_base)
    p_two_sided = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return {
        "delta": observed,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": float(min(1.0, p_two_sided)),
    }


def benjamini_hochberg(p_values) -> np.ndarray:
    """False-discovery-rate adjusted p-values (q-values) across all comparisons."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def summarise(preds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pooled metrics, per-year metrics, and sentiment-vs-price deltas with bootstrap CIs."""
    keys = ["ticker", "horizon", "feature_set"]
    summary_rows, yearly_rows, delta_rows = [], [], []

    for key, g in preds.groupby(keys):
        row = dict(zip(keys, key), n_test_days=len(g))
        row.update(classification_metrics(g["y_true"], g["y_pred"]))
        row.update(regression_metrics(g["ret_true"], g["ret_pred"]))
        row["majority_baseline_accuracy"] = accuracy_score(g["y_true"], g["y_majority"])
        row["majority_baseline_f1_macro"] = _macro_f1(g["y_true"], g["y_majority"])
        summary_rows.append(row)
        for year, gy in g.groupby("test_year"):
            yearly_rows.append({**dict(zip(keys, key)), "test_year": year,
                                **classification_metrics(gy["y_true"], gy["y_pred"]),
                                **regression_metrics(gy["ret_true"], gy["ret_pred"])})

    for (ticker, horizon), g in preds.groupby(["ticker", "horizon"]):
        order = ["series", "date"]  # keeps pooled ("ALL") rows paired and in time order
        base = g[g["feature_set"] == "price"].sort_values(order)
        for set_name in g["feature_set"].unique():
            if set_name == "price":
                continue
            new = g[g["feature_set"] == set_name].sort_values(order)
            f1 = block_bootstrap_delta(base["y_true"].values, base["y_pred"].values,
                                       new["y_pred"].values, _macro_f1)
            rmse = block_bootstrap_delta(base["ret_true"].values, base["ret_pred"].values,
                                         new["ret_pred"].values, _rmse)
            delta_rows.append({
                "ticker": ticker, "horizon": horizon, "compared_to_price": set_name,
                **{f"f1_{k}": v for k, v in f1.items()},
                **{f"rmse_{k}": v for k, v in rmse.items()},
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(yearly_rows), pd.DataFrame(delta_rows)


def plot_deltas(deltas: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [s for s in FEATURE_SETS if s != "price"]
    tickers = sorted(t for t in deltas["ticker"].unique() if t != "ALL") + ["ALL"]
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 4.2), sharey=True)
    width = 0.38
    for ax, h in zip(np.atleast_1d(axes), HORIZONS):
        x = np.arange(len(tickers))
        for i, set_name in enumerate(sets):
            d = deltas[(deltas["horizon"] == h) & (deltas["compared_to_price"] == set_name)]
            d = d.set_index("ticker").loc[tickers]
            err = np.vstack([d["f1_delta"] - d["f1_ci_low"], d["f1_ci_high"] - d["f1_delta"]])
            ax.bar(x + (i - 0.5) * width, d["f1_delta"], width, yerr=err, capsize=3,
                   label=set_name.replace("price+", "+ "))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, tickers)
        ax.set_title(f"{h}-day horizon")
    np.atleast_1d(axes)[0].set_ylabel("Δ macro-F1 vs price-only (95% CI)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_ablation(model_name: str = "xgboost", tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_all(tickers)
    print(f"[OK] {model_name}: {len(dataset):,} ticker-days, "
          f"{dataset['date'].min().date()} to {dataset['date'].max().date()}")

    all_preds, all_importance = [], []
    for ticker in tickers:
        df = dataset[dataset["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        for h in HORIZONS:
            preds, importance = run_ticker_horizon(df, h, model_name)
            all_preds.append(preds.assign(ticker=ticker, horizon=h))
            if not importance.empty:
                all_importance.append(importance.assign(ticker=ticker, horizon=h))
            print(f"[OK] {model_name} {ticker} h={h}: {preds['date'].nunique()} test days", flush=True)

    preds = pd.concat(all_preds, ignore_index=True)
    preds["series"] = preds["ticker"]
    # "ALL" = every ticker's test days pooled, for one overall answer per horizon
    summary, yearly, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])

    preds.to_csv(out_dir / "predictions.csv", index=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    yearly.to_csv(out_dir / "metrics_by_test_year.csv", index=False)
    deltas.to_csv(out_dir / "ablation_deltas.csv", index=False)
    if all_importance:
        importance = pd.concat(all_importance, ignore_index=True)
        (importance.groupby(["ticker", "horizon", "feature_set", "feature"])["importance_gain"]
         .mean().reset_index().to_csv(out_dir / "feature_importance.csv", index=False))
    plot_deltas(deltas, out_dir / "ablation_delta_f1.png")

    pd.set_option("display.width", 200)
    print(f"\n[{model_name}] Macro-F1 (pooled 2021-2023 test days):")
    print(summary.pivot_table(index=["ticker", "horizon"], columns="feature_set",
                              values="f1_macro").round(3).to_string())
    print(f"\n[{model_name}] Sentiment vs price-only (negative RMSE delta = better):")
    print(deltas[["ticker", "horizon", "compared_to_price", "f1_delta", "f1_ci_low", "f1_ci_high",
                  "f1_p_value", "f1_q_bh", "rmse_delta", "rmse_p_value"]].round(3).to_string(index=False))
    print(f"\n[OK] Results written to {out_dir}")
    return summary


def compare_architectures() -> pd.DataFrame:
    """Side-by-side pooled (ALL tickers) results for every model that has been run."""
    frames = []
    for model_name in MODEL_RUNNERS:
        path = OUTPUT_DIR / model_name / "metrics_summary.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(model=model_name))
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    pooled = summary[summary["ticker"] == "ALL"]
    table = pooled.pivot_table(index=["model", "horizon"], columns="feature_set",
                               values=["f1_macro", "rmse"]).round(3)
    table.to_csv(OUTPUT_DIR / "architecture_comparison.csv")
    print("\nArchitecture comparison (all tickers pooled):")
    print(table.to_string())
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Price-only vs sentiment ablation")
    parser.add_argument("--model", choices=list(MODEL_RUNNERS) + ["all"], default="all")
    args = parser.parse_args()
    for name in (list(MODEL_RUNNERS) if args.model == "all" else [args.model]):
        run_ablation(name)
    compare_architectures()
