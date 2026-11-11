"""
Robustness check with exactly timestamped news (Alpaca News API, Benzinga feed).

FNSPID timestamps are date-only (99.8% are midnight UTC), so a headline dated D
cannot be split into before or after the 16:00 New York close of day D, and
after-close news can leak into day D's features. Alpaca's historical news API
gives each Benzinga article a `created_at` time, so this module can test the
ablation with news assigned by its real publication time:

  timed       published before 16:00 ET on trading day D -> day D's features;
              at/after 16:00, weekends and holidays -> the next trading day.
              Every headline used for day D was public before D's close, so the
              model only sees news a trader could have acted on.
  date_only   the same headlines with the time discarded (calendar date in New
              York -> that day, or the next trading day), mimicking how the
              FNSPID data had to be handled. After-close news can leak.

For each timing, XGBoost is trained with price features only and with price +
the six FinBERT sentiment features, with the same splits, hyperparameters,
bootstrap and Benjamini-Hochberg correction as src.ablation.

Needs a free Alpaca account. Put the keys in the project's .env file (it is
gitignored; never commit the keys):

    ALPACA_API_KEY_ID=...
    ALPACA_API_SECRET_KEY=...

Usage:
    python -m src.timestamped_news            # fetch (cached), score, run XGBoost
    python -m src.timestamped_news --fetch-only
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import (
    DEVICE,
    MARKET_CLOSE_HOUR,
    MARKET_TIMEZONE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    STUDY_END,
    STUDY_START,
    STUDY_TICKERS,
)
from src.ablation import benjamini_hochberg, run_ticker_horizon, summarise
from src.features import HORIZONS, PRICE_FEATURES, SENTIMENT_FEATURES, build_dataset
from src.news_sentiment import (
    _aggregate,
    add_decayed_sentiment,
    assign_trading_day,
    fetch_prices,
    mentions_company,
)

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
RAW_DIR = RAW_DATA_DIR / "alpaca_news"
SCORED_PARQUET = PROCESSED_DATA_DIR / "alpaca_news_scored.parquet"
OUTPUT_DIR = RESULTS_DIR / "timestamped_news"

FEATURE_SETS = {
    "price": PRICE_FEATURES,
    "price+sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _headers() -> dict:
    key, secret = os.getenv("ALPACA_API_KEY_ID", ""), os.getenv("ALPACA_API_SECRET_KEY", "")
    if not key or not secret:
        raise SystemExit(
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in the project's .env file."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(session, params: dict, headers: dict) -> dict:
    """One page, with retries on rate limits and transient errors."""
    for attempt in range(8):
        r = session.get(ALPACA_NEWS_URL, params=params, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(60, 2 ** attempt))
            continue
        raise SystemExit(f"Alpaca news API error {r.status_code}: {r.text[:300]}")
    raise SystemExit("Alpaca news API: too many retries")


def fetch_ticker_year(session, ticker: str, year: int, headers: dict) -> pd.DataFrame:
    """All Benzinga articles tagged with `ticker` in one calendar year (cached)."""
    path = RAW_DIR / f"{ticker}_{year}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    params = {
        "symbols": ticker,
        "start": f"{year}-01-01T00:00:00Z",
        "end": f"{year + 1}-01-01T00:00:00Z",
        "limit": 50,
        "sort": "asc",
        "include_content": "false",
    }
    rows, token = [], None
    while True:
        if token:
            params["page_token"] = token
        page = _get(session, params, headers)
        for a in page.get("news", []):
            rows.append(
                {
                    "id": a.get("id"),
                    "published_utc": a.get("created_at"),
                    "headline": a.get("headline", ""),
                    "source": a.get("source") or "benzinga",
                    "n_symbols": len(a.get("symbols") or []),
                    "url": a.get("url", ""),
                }
            )
        token = page.get("next_page_token")
        if not token:
            break
        time.sleep(0.32)  # stays under the free plan's 200 requests per minute

    df = pd.DataFrame(rows, columns=["id", "published_utc", "headline", "source", "n_symbols", "url"])
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    df["ticker"] = ticker
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  {ticker} {year}: {len(df):,} articles", flush=True)
    return df


def fetch_all(tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    import requests

    headers = _headers()
    years = range(pd.Timestamp(STUDY_START).year, pd.Timestamp(STUDY_END).year + 1)
    with requests.Session() as session:
        frames = [fetch_ticker_year(session, t, y, headers) for t in tickers for y in years]
    news = pd.concat(frames, ignore_index=True)
    news = news[news["headline"].str.strip().str.len() > 0]
    return news.drop_duplicates(subset=["ticker", "id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scoring and daily features
# ---------------------------------------------------------------------------


def score(news: pd.DataFrame) -> pd.DataFrame:
    if SCORED_PARQUET.exists():
        cached = pd.read_parquet(SCORED_PARQUET)
        if len(cached) == len(news):
            return cached

    from src.sentiment_analyzer import FinBERTSentimentAnalyzer

    # Score each distinct headline once (one article can be tagged to several study tickers)
    unique = news["headline"].astype(str).drop_duplicates().tolist()
    print(f"Scoring {len(unique):,} distinct headlines with FinBERT on {DEVICE} ...", flush=True)
    probs = FinBERTSentimentAnalyzer(device=DEVICE).predict_proba(unique)
    table = pd.DataFrame({"headline": unique, **{f"p_{k}": v for k, v in probs.items()}})

    scored = news.assign(headline=news["headline"].astype(str)).merge(table, on="headline", how="left")
    scored["finbert_label"] = scored[["p_positive", "p_negative", "p_neutral"]].idxmax(axis=1).str[2:]
    scored["sentiment_score"] = scored["p_positive"] - scored["p_negative"]
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORED_PARQUET, index=False)
    return scored


def assign_date_only(news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Discard the time: New York calendar date D -> trading day D (or the next one)."""
    local_date = news["published_utc"].dt.tz_convert(MARKET_TIMEZONE).dt.normalize().dt.tz_localize(None)
    days = trading_days.sort_values().values
    pos = np.searchsorted(days, local_date.values, side="left")
    valid = pos < len(days)
    out = news.loc[valid].copy()
    out["trading_day"] = days[pos[valid]]
    return out


def daily_features(scored: pd.DataFrame, timing: str, tickers: List[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        trading_days = pd.DatetimeIndex(pd.to_datetime(fetch_prices(t)["date"]))
        sub = scored[scored["ticker"] == t]
        aligned = (assign_trading_day(sub, trading_days) if timing == "timed"
                   else assign_date_only(sub, trading_days))
        aligned["mentions_company"] = mentions_company(aligned["headline"], t)
        daily = aligned.groupby("trading_day").apply(_aggregate, include_groups=False).reset_index()
        frames.append(add_decayed_sentiment(daily, trading_days).assign(ticker=t))
    return pd.concat(frames, ignore_index=True)


def timing_profile(scored: pd.DataFrame) -> pd.DataFrame:
    """Share of headlines published before the open, during the session, after the close."""
    local = scored["published_utc"].dt.tz_convert(MARKET_TIMEZONE)
    minutes = local.dt.hour * 60 + local.dt.minute
    weekend = local.dt.dayofweek >= 5
    bucket = np.select(
        [weekend, minutes < 9 * 60 + 30, minutes < MARKET_CLOSE_HOUR * 60],
        ["weekend", "before_open", "during_session"],
        default="after_close",
    )
    out = pd.crosstab(scored["ticker"], bucket, normalize="index").mul(100).round(1)
    out["headlines"] = scored.groupby("ticker").size()
    midnight = (scored["published_utc"].dt.hour == 0) & (scored["published_utc"].dt.minute == 0)
    out["pct_midnight_utc"] = midnight.groupby(scored["ticker"]).mean().mul(100).round(1)
    return out


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run(tickers: List[str] = STUDY_TICKERS, model_name: str = "xgboost") -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    news = fetch_all(tickers)
    scored = score(news)
    profile = timing_profile(scored)
    profile.to_csv(OUTPUT_DIR / "timing_profile.csv")
    print("\nWhen the headlines were published (% of each ticker's headlines, New York time):")
    print(profile.to_string())

    all_deltas, all_summary, corr_rows = [], [], []
    for timing in ("timed", "date_only"):
        daily = daily_features(scored, timing, tickers)
        dataset = pd.concat([build_dataset(t, daily) for t in tickers], ignore_index=True)

        # Spearman correlation of daily sentiment with the next day's return (close D -> D+1)
        news_days = dataset[dataset["news_count"] > 0]
        rho = news_days["sent_mean"].corr(news_days["fwd_ret_1"], method="spearman")
        corr_rows.append({"timing": timing, "news_days": len(news_days),
                          "spearman_sent_vs_next_day_return": rho})

        all_preds = []
        for t in tickers:
            df = dataset[dataset["ticker"] == t].sort_values("date").reset_index(drop=True)
            for h in HORIZONS:
                preds, _ = run_ticker_horizon(df, h, model_name, feature_sets=FEATURE_SETS)
                all_preds.append(preds.assign(ticker=t, horizon=h))
            print(f"[OK] {model_name} {timing} {t}", flush=True)
        preds = pd.concat(all_preds, ignore_index=True)
        preds["series"] = preds["ticker"]
        summary, _, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
        all_summary.append(summary.assign(timing=timing))
        all_deltas.append(deltas.assign(timing=timing))

    summary = pd.concat(all_summary, ignore_index=True)
    deltas = pd.concat(all_deltas, ignore_index=True)
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])
    corr = pd.DataFrame(corr_rows)

    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "timestamped_deltas.csv", index=False)
    pooled = deltas[deltas["ticker"] == "ALL"]
    pooled.to_csv(OUTPUT_DIR / "timestamped_pooled.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "next_day_correlation.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nMacro-F1, five stocks pooled:")
    print(summary[summary["ticker"] == "ALL"].pivot_table(
        index=["timing", "horizon"], columns="feature_set", values="f1_macro").round(3).to_string())
    print("\nSentiment vs price-only, pooled (BH-corrected across all rows):")
    print(pooled[["timing", "horizon", "f1_delta", "f1_ci_low", "f1_ci_high", "f1_p_value",
                  "f1_q_bh", "rmse_delta", "rmse_q_bh"]].round(3).to_string(index=False))
    print("\nNext-day correlation:")
    print(corr.round(3).to_string(index=False))
    print(f"\n[OK] Results written to {OUTPUT_DIR}")
    return deltas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    if args.fetch_only:
        n = fetch_all()
        print(f"[OK] {len(n):,} articles fetched to {RAW_DIR}")
    else:
        run()
