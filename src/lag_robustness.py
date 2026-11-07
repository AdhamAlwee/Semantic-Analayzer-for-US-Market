"""
Robustness check for FNSPID's date-only timestamps.

99.8% of FNSPID timestamps are midnight UTC, so the 16:00 market-close rule in
src.news_sentiment cannot act on the real publication time: a headline dated D is
assigned to trading day D even if it was published after D's close. It can then
feed the features of day D although it could only move the price after that close.

This module removes that possible look-ahead by using each day's sentiment one
trading day later:
  * ablation: the same experiment as src.ablation, with every sentiment feature on
    row t taken from day t-1. Price features, targets, splits and hyperparameters
    do not change, so the price-only model is the same as in the main experiment.
  * market agreement: FinBERT score of day T against the return close(T+1) ->
    close(T+2), which starts after any headline dated T has been published.
  * sentiment-model comparison: the XGBoost experiment of src.sentiment_downstream,
    with each model's sentiment lagged in the same way.

Usage:
    python -m src.lag_robustness
    python -m src.lag_robustness --model lstm_balanced
    python -m src.lag_robustness --model downstream
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
from src.ablation import (
    MODEL_RUNNERS,
    OUTPUT_DIR as ABLATION_DIR,
    benjamini_hochberg,
    run_ticker_horizon,
    summarise,
)
from src.features import (
    COMPANY_SENTIMENT_FEATURES,
    HORIZONS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    build_all,
)
from src.news_sentiment import fetch_prices
from src.sentiment_downstream import DEFAULT_MODELS
from src.source_comparison import (
    ALL_SOURCE_TICKER,
    MIN_HEADLINES,
    PAIR_SOURCES,
    PAIR_TICKERS,
    agreement_metrics,
    load_window_headlines,
)

OUTPUT_DIR = RESULTS_DIR / "lag_robustness"
LAG_DAYS = 1


def lag_sentiment(df: pd.DataFrame, days: int = LAG_DAYS) -> pd.DataFrame:
    """One ticker, sorted by date: sentiment features come from `days` trading days earlier."""
    out = df.copy()
    columns = SENTIMENT_FEATURES + COMPANY_SENTIMENT_FEATURES
    out[columns] = out[columns].shift(days).fillna(0)
    return out


def run_lagged_ablation(model_name: str, tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_all(tickers)

    all_preds = []
    for ticker in tickers:
        df = dataset[dataset["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        df = lag_sentiment(df)
        for h in HORIZONS:
            preds, _ = run_ticker_horizon(df, h, model_name)
            all_preds.append(preds.assign(ticker=ticker, horizon=h))
            print(f"[OK] {model_name} lagged {ticker} h={h}", flush=True)

    preds = pd.concat(all_preds, ignore_index=True)
    preds["series"] = preds["ticker"]
    summary, yearly, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])

    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    yearly.to_csv(out_dir / "metrics_by_test_year.csv", index=False)
    deltas.to_csv(out_dir / "ablation_deltas.csv", index=False)

    # Same-day (main experiment) next to lagged, for all stocks pooled
    same_day = pd.read_csv(ABLATION_DIR / model_name / "ablation_deltas.csv")
    keys = ["ticker", "horizon", "compared_to_price"]
    cols = keys + ["f1_delta", "f1_ci_low", "f1_ci_high", "f1_q_bh"]
    comparison = same_day[cols].merge(deltas[cols], on=keys, suffixes=("_same_day", "_lagged"))
    comparison.to_csv(out_dir / "same_day_vs_lagged.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n[{model_name}] Sentiment gain in macro-F1, same-day vs lagged:")
    print(comparison[comparison["ticker"] == "ALL"].round(3).to_string(index=False))
    print(f"[{model_name}] Smallest lagged q: {deltas['f1_q_bh'].min():.3f}")
    return comparison


def run_lagged_downstream(tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    """The XGBoost sentiment-model comparison of src.sentiment_downstream, lagged."""
    out_dir = OUTPUT_DIR / "sentiment_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    for i, model in enumerate(DEFAULT_MODELS):
        dataset = build_all(tickers, sentiment_model=model)
        feature_sets = {f"price+{model}": PRICE_FEATURES + SENTIMENT_FEATURES}
        if i == 0:
            feature_sets = {"price": PRICE_FEATURES, **feature_sets}
        for ticker in tickers:
            df = dataset[dataset["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            df = lag_sentiment(df)
            for h in HORIZONS:
                preds, _ = run_ticker_horizon(df, h, "xgboost", feature_sets=feature_sets)
                all_preds.append(preds.assign(ticker=ticker, horizon=h))
        print(f"[OK] lagged XGBoost done with {model} sentiment", flush=True)

    preds = pd.concat(all_preds, ignore_index=True)
    preds["series"] = preds["ticker"]
    summary, _, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
    deltas["sentiment_model"] = deltas["compared_to_price"].str.replace("price+", "", regex=False)
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    deltas.to_csv(out_dir / "deltas_vs_price.csv", index=False)

    pooled = deltas[deltas["ticker"] == "ALL"]
    table = pooled.pivot(index="sentiment_model", columns="horizon", values="f1_delta")
    table["min_q_pooled"] = pooled.groupby("sentiment_model")["f1_q_bh"].min()
    table.to_csv(out_dir / "downstream_comparison.csv")
    print("\nPooled change in macro-F1 vs price-only (lagged sentiment):")
    print(table.round(3).to_string())
    print(f"Smallest lagged q: {deltas['f1_q_bh'].min():.3f}")
    return table


def lagged_agreement_table(news: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in news.groupby("source"):
        if len(g) >= MIN_HEADLINES:
            rows.append({"source": source, "headlines": len(g),
                         **agreement_metrics(g, "lagged_ret_1d")})
    return pd.DataFrame(rows).sort_values("headlines", ascending=False)


def run_lagged_agreement() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    news = load_window_headlines()
    frames = []
    for ticker, group in news.groupby("ticker"):
        prices = fetch_prices(ticker).sort_values("date")
        close = prices.set_index(pd.to_datetime(prices["date"]))["close"]
        lagged = close.shift(-2) / close.shift(-1) - 1  # close(T+1) -> close(T+2)
        frames.append(group.assign(lagged_ret_1d=group["trading_day"].map(lagged)))
    news = pd.concat(frames, ignore_index=True).dropna(subset=["lagged_ret_1d"])

    intc = lagged_agreement_table(news[news["ticker"] == ALL_SOURCE_TICKER])
    pair = lagged_agreement_table(
        news[news["ticker"].isin(PAIR_TICKERS) & news["source"].isin(PAIR_SOURCES)]
    )
    intc.to_csv(OUTPUT_DIR / "agreement_intc_all_sources.csv", index=False)
    pair.to_csv(OUTPUT_DIR / "agreement_nasdaq_vs_benzinga.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nMarket agreement, close(T+1) -> close(T+2), INTC all outlets:")
    print(intc.round(3).to_string(index=False))
    print("\nMarket agreement, close(T+1) -> close(T+2), Nasdaq vs Benzinga:")
    print(pair.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lagged-sentiment robustness check")
    parser.add_argument("--model", choices=list(MODEL_RUNNERS) + ["downstream"], default="xgboost")
    args = parser.parse_args()
    if args.model == "downstream":
        run_lagged_downstream()
    else:
        run_lagged_agreement()
        run_lagged_ablation(args.model)
