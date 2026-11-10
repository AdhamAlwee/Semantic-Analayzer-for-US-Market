"""
Does sentiment predict the SIZE of the next move rather than its direction?

News is often found to predict volatility better than direction. This module keeps
the main study's data, splits, hyperparameters and bootstrap, and changes only the
target to the absolute forward return |close[t+h] / close[t] - 1|:

  regression      XGBoost regressor of |return| in percent, scored by RMSE
  classification  "large move" = |return| above the median of the training window
                  (so the classes are balanced and no test information is used),
                  scored by binary macro-F1

Feature sets (fixed in advance):
  price           price features only
  price+sentiment price + daily FinBERT features from all tagged headlines
  price+company   price + FinBERT features from headlines naming the company only

Both same-day and one-day-lagged sentiment are tested. The price features already
include 20-day volatility and the high-low range, so the price-only baseline is a
strong volatility model and any gain must come on top of volatility clustering.

Usage:
    python -m src.volatility_target
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier, XGBRegressor

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_TICKERS
from src.ablation import (
    TEST_YEARS,
    XGB_PARAMS,
    benjamini_hochberg,
    block_bootstrap_delta,
    walk_forward_splits,
)
from src.features import (
    COMPANY_SENTIMENT_FEATURES,
    HORIZONS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    build_all,
)
from src.lag_robustness import lag_sentiment

OUTPUT_DIR = RESULTS_DIR / "volatility_target"

FEATURE_SETS: Dict[str, List[str]] = {
    "price": PRICE_FEATURES,
    "price+sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
    "price+company": PRICE_FEATURES + COMPANY_SENTIMENT_FEATURES,
}


def _binary_macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def run_ticker_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = f"fwd_ret_{horizon}"
    data = df.dropna(subset=[target]).reset_index(drop=True)
    data["abs_ret"] = data[target].abs() * 100  # percent

    rows = []
    for year, train_idx, test_idx in walk_forward_splits(data["date"], horizon, TEST_YEARS):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        threshold = float(np.median(train["abs_ret"]))
        y_train = (train["abs_ret"] > threshold).astype(int).values
        y_test = (test["abs_ret"] > threshold).astype(int).values
        for set_name, features in FEATURE_SETS.items():
            clf = XGBClassifier(objective="binary:logistic", **XGB_PARAMS)
            clf.fit(train[features], y_train)
            reg = XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
            reg.fit(train[features], train["abs_ret"])
            rows.append(pd.DataFrame({
                "date": test["date"].values,
                "test_year": year,
                "feature_set": set_name,
                "y_true": y_test,
                "y_pred": clf.predict(test[features]),
                "abs_true": test["abs_ret"].values,
                "abs_pred": reg.predict(test[features]),
                "abs_naive": float(train["abs_ret"].mean()),
            }))
    return pd.concat(rows, ignore_index=True)


def summarise(preds: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    summary_rows, delta_rows = [], []
    for (ticker, horizon, set_name), g in preds.groupby(["ticker", "horizon", "feature_set"]):
        summary_rows.append({
            "ticker": ticker, "horizon": horizon, "feature_set": set_name,
            "n_test_days": len(g),
            "f1_macro": _binary_macro_f1(g["y_true"], g["y_pred"]),
            "rmse": _rmse(g["abs_true"], g["abs_pred"]),
            "rmse_training_mean": _rmse(g["abs_true"], g["abs_naive"]),
        })
    for (ticker, horizon), g in preds.groupby(["ticker", "horizon"]):
        order = ["series", "date"]
        base = g[g["feature_set"] == "price"].sort_values(order)
        for set_name in FEATURE_SETS:
            if set_name == "price":
                continue
            new = g[g["feature_set"] == set_name].sort_values(order)
            f1 = block_bootstrap_delta(base["y_true"].values, base["y_pred"].values,
                                       new["y_pred"].values, _binary_macro_f1)
            rmse = block_bootstrap_delta(base["abs_true"].values, base["abs_pred"].values,
                                         new["abs_pred"].values, _rmse)
            delta_rows.append({
                "ticker": ticker, "horizon": horizon, "compared_to_price": set_name,
                **{f"f1_{k}": v for k, v in f1.items()},
                **{f"rmse_{k}": v for k, v in rmse.items()},
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def run(tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_all(tickers)

    summaries, all_deltas = [], []
    for timing, lag in [("same_day", False), ("lagged", True)]:
        all_preds = []
        for ticker in tickers:
            df = dataset[dataset["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if lag:
                df = lag_sentiment(df)
            for h in HORIZONS:
                all_preds.append(run_ticker_horizon(df, h).assign(ticker=ticker, horizon=h))
            print(f"[OK] {timing} {ticker}", flush=True)
        preds = pd.concat(all_preds, ignore_index=True)
        preds["series"] = preds["ticker"]
        summary, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
        summaries.append(summary.assign(timing=timing))
        all_deltas.append(deltas.assign(timing=timing))

    summary = pd.concat(summaries, ignore_index=True)
    deltas = pd.concat(all_deltas, ignore_index=True)
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "volatility_deltas.csv", index=False)

    pooled = deltas[deltas["ticker"] == "ALL"]
    cols = ["timing", "horizon", "compared_to_price", "f1_delta", "f1_ci_low", "f1_ci_high",
            "f1_q_bh", "rmse_delta", "rmse_ci_low", "rmse_ci_high", "rmse_q_bh"]
    table = pooled[cols].sort_values(["timing", "horizon", "compared_to_price"])
    table.to_csv(OUTPUT_DIR / "volatility_pooled.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nPooled metrics (all stocks):")
    print(summary[summary["ticker"] == "ALL"].round(4).to_string(index=False))
    print("\nPooled change vs price-only (negative RMSE delta = better):")
    print(table.round(4).to_string(index=False))
    print(f"Smallest F1 q: {deltas['f1_q_bh'].min():.3f}; smallest RMSE q: {deltas['rmse_q_bh'].min():.3f}")
    return table


if __name__ == "__main__":
    run()
