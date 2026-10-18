"""
News-source comparison (interim report, Section 3.7, source axis).

FNSPID only has outlets other than nasdaq.com up to 2020-06-11, so everything
here uses 2018-01-01 .. 2020-06-11.

Part A - FinBERT classification consistency per source. There are no human
labels for FNSPID headlines, so two label-free measures are used:
  * market agreement: for non-neutral headlines, how often the sign of the
    FinBERT score matches the sign of the stock's 2-day reaction return
    close(T-1) -> close(T+1) around the headline's trading day T (FNSPID
    timestamps are date-only, so the window spans the whole publication day);
    plus the Spearman correlation between score and reaction return
  * cross-source consistency: on ticker-days covered by two outlets, the
    correlation of their daily mean sentiment
  Compared on INTC (the only stock all six outlets cover) and on the four
  stocks Benzinga and Nasdaq both cover.

Part B - downstream prediction per source: XGBoost ablation with sentiment
features built from ONE outlet (nasdaq.com vs benzinga.com), trained on
2018-2019 and tested on 2020-01-01 .. 2020-06-11. Small test set; exploratory.

Usage:
    python -m src.source_comparison
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_START
from src.ablation import benjamini_hochberg, run_ticker_horizon, summarise
from src.features import HORIZONS, PRICE_FEATURES, SENTIMENT_FEATURES, build_dataset
from src.news_sentiment import (
    DAILY_BY_SOURCE_PARQUET,
    SCORED_PARQUET,
    add_decayed_sentiment,
    assign_trading_day,
    fetch_prices,
)

OUTPUT_DIR = RESULTS_DIR / "source_comparison"
SOURCE_WINDOW_END = "2020-06-11"  # last non-nasdaq.com headline in FNSPID
MIN_HEADLINES = 200
ALL_SOURCE_TICKER = "INTC"
PAIR_SOURCES = ["nasdaq.com", "benzinga.com"]
PAIR_TICKERS = ["INTC", "MRK", "SBUX", "WFC"]


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def load_window_headlines() -> pd.DataFrame:
    """FinBERT-scored headlines in the source window, with trading day and reaction return."""
    scored = pd.read_parquet(SCORED_PARQUET)
    end = pd.Timestamp(SOURCE_WINDOW_END, tz="UTC") + pd.Timedelta(days=1)
    scored = scored[scored["published_utc"] < end]

    frames = []
    for ticker, group in scored.groupby("ticker"):
        prices = fetch_prices(ticker).sort_values("date").reset_index(drop=True)
        days = pd.DatetimeIndex(pd.to_datetime(prices["date"]))
        aligned = assign_trading_day(group, days)
        close = prices.set_index(pd.to_datetime(prices["date"]))["close"]
        reaction = close.shift(-1) / close.shift(1) - 1  # close(T-1) -> close(T+1)
        aligned["reaction_ret_2d"] = aligned["trading_day"].map(reaction)
        frames.append(aligned)
    return pd.concat(frames, ignore_index=True).dropna(subset=["reaction_ret_2d"])


def source_consistency_table(news: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in news.groupby("source"):
        if len(g) < MIN_HEADLINES:
            continue
        polar = g[g["finbert_label"] != "neutral"]
        hits = int((np.sign(polar["sentiment_score"]) == np.sign(polar["reaction_ret_2d"])).sum())
        low, high = wilson_interval(hits, len(polar))
        rho, p = spearmanr(g["sentiment_score"], g["reaction_ret_2d"])
        mix = g["finbert_label"].value_counts(normalize=True)
        rows.append({
            "source": source,
            "headlines": len(g),
            "tickers": g["ticker"].nunique(),
            "pct_positive": mix.get("positive", 0) * 100,
            "pct_neutral": mix.get("neutral", 0) * 100,
            "pct_negative": mix.get("negative", 0) * 100,
            "mean_score": g["sentiment_score"].mean(),
            "non_neutral_headlines": len(polar),
            "market_agreement": hits / len(polar) if len(polar) else np.nan,
            "agreement_ci_low": low,
            "agreement_ci_high": high,
            "spearman_score_vs_reaction": rho,
            "spearman_p": p,
        })
    return pd.DataFrame(rows).sort_values("headlines", ascending=False)


def cross_source_correlation(news: pd.DataFrame, sources: List[str]) -> pd.DataFrame:
    """Correlation of daily mean sentiment between outlets on shared ticker-days."""
    daily = (news[news["source"].isin(sources)]
             .groupby(["ticker", "trading_day", "source"])["sentiment_score"].mean()
             .unstack("source"))
    rows = []
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            both = daily[[a, b]].dropna()
            if len(both) >= 30:
                rho, p = spearmanr(both[a], both[b])
                rows.append({"source_a": a, "source_b": b, "shared_ticker_days": len(both),
                             "spearman_daily_sentiment": rho, "p_value": p})
    return pd.DataFrame(rows)


def source_daily_features(source: str, ticker: str) -> pd.DataFrame:
    """Daily features for one ticker from one outlet, with cross-day decay."""
    by_source = pd.read_parquet(DAILY_BY_SOURCE_PARQUET)
    d = by_source[(by_source["source"] == source) & (by_source["ticker"] == ticker)]
    days = pd.DatetimeIndex(pd.to_datetime(fetch_prices(ticker)["date"]))
    days = days[days <= pd.Timestamp(SOURCE_WINDOW_END)]
    d = d[d["trading_day"] <= pd.Timestamp(SOURCE_WINDOW_END)].drop(columns=["source", "ticker"])
    return add_decayed_sentiment(d, days).assign(ticker=ticker)


def downstream_by_source() -> Dict[str, pd.DataFrame]:
    all_preds = []
    for ticker in PAIR_TICKERS:
        for i, source in enumerate(PAIR_SOURCES):
            df = build_dataset(ticker, source_daily_features(source, ticker))
            df = df[df["date"] <= pd.Timestamp(SOURCE_WINDOW_END)].reset_index(drop=True)
            feature_sets = {f"price+{source}": PRICE_FEATURES + SENTIMENT_FEATURES}
            if i == 0:
                feature_sets = {"price": PRICE_FEATURES, **feature_sets}
            for h in HORIZONS:
                preds, _ = run_ticker_horizon(df, h, "xgboost", feature_sets=feature_sets,
                                              test_years=[2020])
                all_preds.append(preds.assign(ticker=ticker, horizon=h))

    preds = pd.concat(all_preds, ignore_index=True)
    preds["series"] = preds["ticker"]
    summary, _, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
    deltas["source"] = deltas["compared_to_price"].str.replace("price+", "", regex=False)
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    return {"predictions": preds, "summary": summary, "deltas": deltas}


def run_source_comparison() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    news = load_window_headlines()
    print(f"[OK] {len(news):,} headlines in {STUDY_START} .. {SOURCE_WINDOW_END}")

    intc_sources = source_consistency_table(news[news["ticker"] == ALL_SOURCE_TICKER])
    pair = news[news["ticker"].isin(PAIR_TICKERS) & news["source"].isin(PAIR_SOURCES)]
    pair_sources = source_consistency_table(pair)
    pair_by_ticker = pd.concat(
        [source_consistency_table(g).assign(ticker=t) for t, g in pair.groupby("ticker")],
        ignore_index=True,
    )
    intc_sources_list = intc_sources["source"].tolist()
    cross = pd.concat([
        cross_source_correlation(news[news["ticker"] == ALL_SOURCE_TICKER], intc_sources_list)
        .assign(scope="INTC"),
        cross_source_correlation(pair, PAIR_SOURCES).assign(scope="INTC+MRK+SBUX+WFC"),
    ], ignore_index=True)

    intc_sources.to_csv(OUTPUT_DIR / "consistency_intc_all_sources.csv", index=False)
    pair_sources.to_csv(OUTPUT_DIR / "consistency_nasdaq_vs_benzinga.csv", index=False)
    pair_by_ticker.to_csv(OUTPUT_DIR / "consistency_nasdaq_vs_benzinga_by_ticker.csv", index=False)
    cross.to_csv(OUTPUT_DIR / "cross_source_daily_correlation.csv", index=False)

    results = downstream_by_source()
    results["predictions"].to_csv(OUTPUT_DIR / "downstream_predictions.csv", index=False)
    results["summary"].to_csv(OUTPUT_DIR / "downstream_metrics_summary.csv", index=False)
    results["deltas"].to_csv(OUTPUT_DIR / "downstream_deltas_vs_price.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["source", "headlines", "pct_positive", "pct_neutral", "pct_negative", "market_agreement",
            "agreement_ci_low", "agreement_ci_high", "spearman_score_vs_reaction", "spearman_p"]
    print("\nINTC, all outlets (FinBERT):")
    print(intc_sources[cols].round(3).to_string(index=False))
    print("\nNasdaq vs Benzinga, INTC+MRK+SBUX+WFC:")
    print(pair_sources[cols].round(3).to_string(index=False))
    print("\nCross-source daily sentiment correlation:")
    print(cross.round(3).to_string(index=False))
    d = results["deltas"]
    print("\nDownstream (test 2020-01..2020-06-11), Δ macro-F1 vs price-only:")
    print(d[["ticker", "horizon", "source", "f1_delta", "f1_ci_low", "f1_ci_high", "f1_p_value",
             "f1_q_bh"]].round(3).to_string(index=False))
    print(f"\n[OK] Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_source_comparison()
