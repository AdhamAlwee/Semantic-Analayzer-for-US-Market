"""
Does sentiment help once headlines that do not name the company are removed?

Only about 40-50% of the FNSPID headlines tagged to a ticker name the company or
its ticker; the rest are market round-ups or articles about other firms. The main
ablation adds company-only features ON TOP of all-headline sentiment. This module
instead REPLACES the all-headline features with the company-only ones, so the
model sees sentiment only from relevant headlines:

  price             price features only (same model as the main experiment)
  price+sentiment   price + all-headline sentiment (the main experiment)
  price+company     price + sentiment from headlines naming the company only

Direction target (down / stable / up), same splits, hyperparameters and bootstrap
as src.ablation, with same-day and one-day-lagged sentiment. XGBoost and the
class-weighted LSTM are run.

Usage:
    python -m src.relevance_ablation
    python -m src.relevance_ablation --model xgboost
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_TICKERS
from src.ablation import benjamini_hochberg, run_ticker_horizon, summarise
from src.features import (
    COMPANY_SENTIMENT_FEATURES,
    HORIZONS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    build_all,
)
from src.lag_robustness import lag_sentiment

OUTPUT_DIR = RESULTS_DIR / "relevance_ablation"

FEATURE_SETS = {
    "price": PRICE_FEATURES,
    "price+sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
    "price+company": PRICE_FEATURES + COMPANY_SENTIMENT_FEATURES,
}


def run(model_name: str, tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_all(tickers)

    all_deltas = []
    for timing, lag in [("same_day", False), ("lagged", True)]:
        all_preds = []
        for ticker in tickers:
            df = dataset[dataset["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if lag:
                df = lag_sentiment(df)
            for h in HORIZONS:
                preds, _ = run_ticker_horizon(df, h, model_name, feature_sets=FEATURE_SETS)
                all_preds.append(preds.assign(ticker=ticker, horizon=h))
            print(f"[OK] {model_name} {timing} {ticker}", flush=True)
        preds = pd.concat(all_preds, ignore_index=True)
        preds["series"] = preds["ticker"]
        summary, _, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
        summary.assign(timing=timing).to_csv(out_dir / f"metrics_summary_{timing}.csv", index=False)
        all_deltas.append(deltas.assign(timing=timing))

    deltas = pd.concat(all_deltas, ignore_index=True)
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])
    deltas.to_csv(out_dir / "relevance_deltas.csv", index=False)

    pooled = deltas[deltas["ticker"] == "ALL"]
    cols = ["timing", "horizon", "compared_to_price", "f1_delta", "f1_ci_low", "f1_ci_high",
            "f1_p_value", "f1_q_bh", "rmse_delta", "rmse_q_bh"]
    table = pooled[cols].sort_values(["timing", "horizon", "compared_to_price"])
    table.to_csv(out_dir / "relevance_pooled.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n[{model_name}] Pooled change in macro-F1 vs price-only:")
    print(table.round(4).to_string(index=False))
    print(f"[{model_name}] Smallest q (all tickers): {deltas['f1_q_bh'].min():.3f}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["xgboost", "lstm_balanced", "all"], default="all")
    args = parser.parse_args()
    for name in (["xgboost", "lstm_balanced"] if args.model == "all" else [args.model]):
        run(name)
