"""
Does the sentiment gain come from the scores or from the number of headlines?

With sentiment lagged by one day, six of the seven sentiment models (including
VADER, which labels 98.5% of headlines neutral) gave similar 3-day gains. Every
model shares the headline-count feature, so this module separates the two:

  price                    price features only (same model as the main experiment)
  price+count              price + log headline count
  price+scores             price + sentiment features without the headline count
  price+sentiment          price + all sentiment features (the main experiment)

XGBoost is run with same-day and with one-day-lagged features, using the same
splits, hyperparameters and bootstrap as src.ablation.

Usage:
    python -m src.count_ablation
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_TICKERS
from src.ablation import benjamini_hochberg, run_ticker_horizon, summarise
from src.features import HORIZONS, PRICE_FEATURES, SENTIMENT_FEATURES, build_all
from src.lag_robustness import lag_sentiment

OUTPUT_DIR = RESULTS_DIR / "count_ablation"
COUNT_FEATURE = "news_count_log"

FEATURE_SETS = {
    "price": PRICE_FEATURES,
    "price+count": PRICE_FEATURES + [COUNT_FEATURE],
    "price+scores": PRICE_FEATURES + [f for f in SENTIMENT_FEATURES if f != COUNT_FEATURE],
    "price+sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
}


def run_count_ablation(tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_all(tickers)

    all_deltas = []
    for timing, lag in [("same_day", False), ("lagged", True)]:
        all_preds = []
        for ticker in tickers:
            df = dataset[dataset["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if lag:
                df = lag_sentiment(df)
            for h in HORIZONS:
                preds, _ = run_ticker_horizon(df, h, "xgboost", feature_sets=FEATURE_SETS)
                all_preds.append(preds.assign(ticker=ticker, horizon=h))
            print(f"[OK] {timing} {ticker}", flush=True)
        preds = pd.concat(all_preds, ignore_index=True)
        preds["series"] = preds["ticker"]
        summary, _, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
        summary.assign(timing=timing).to_csv(OUTPUT_DIR / f"metrics_summary_{timing}.csv", index=False)
        all_deltas.append(deltas.assign(timing=timing))

    deltas = pd.concat(all_deltas, ignore_index=True)
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])
    deltas.to_csv(OUTPUT_DIR / "count_deltas.csv", index=False)

    pooled = deltas[deltas["ticker"] == "ALL"]
    cols = ["timing", "horizon", "compared_to_price", "f1_delta", "f1_ci_low", "f1_ci_high",
            "f1_p_value", "f1_q_bh"]
    table = pooled[cols].sort_values(["timing", "horizon", "compared_to_price"])
    table.to_csv(OUTPUT_DIR / "count_vs_scores_pooled.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPooled change in macro-F1 vs price-only:")
    print(table.round(3).to_string(index=False))
    print(f"Smallest q (all tickers): {deltas['f1_q_bh'].min():.3f}")
    return table


if __name__ == "__main__":
    run_count_ablation()
