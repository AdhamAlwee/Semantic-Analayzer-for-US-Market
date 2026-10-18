"""
Turn FNSPID headlines into daily, per-ticker FinBERT sentiment features.

Steps (interim report, Sections 3.2-3.5):
  1. Score every headline with FinBERT -> P(positive), P(negative), P(neutral)
  2. Assign each headline to the trading day whose close it can first affect
     (news at/after 16:00 New York time counts toward the next trading day)
  3. Aggregate per ticker x trading day: mean sentiment, within-day sentiment
     volatility, headline counts, and the same for headlines that actually name
     the company (FNSPID ticker tags also cover peer/market news)
  4. Recency weighting ACROSS days (exponential decay, half-life in trading days).
     FNSPID timestamps are date-only (midnight UTC), so intraday recency
     weighting is not possible; a headline dated D only feeds moves after D's close.
  5. The same aggregation per news source, for the source-comparison axis

Usage:
    python -m src.news_sentiment
"""

import sys
from pathlib import Path
from typing import List, Optional

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
    STUDY_END,
    STUDY_START,
    STUDY_TICKERS,
)
from src.fnspid_loader import COVERAGE_DIR, load_news
from src.utils import normalize_yfinance_dataframe

SCORED_PARQUET = PROCESSED_DATA_DIR / "fnspid_news_scored.parquet"
DAILY_PARQUET = PROCESSED_DATA_DIR / "daily_sentiment.parquet"
DAILY_BY_SOURCE_PARQUET = PROCESSED_DATA_DIR / "daily_sentiment_by_source.parquet"
PRICES_DIR = RAW_DATA_DIR / "prices"

DECAY_HALF_LIFE_DAYS = 3.0

COMPANY_ALIASES = {
    "INTC": ["intel", "intc"],
    "SBUX": ["starbucks", "sbux"],
    "WFC": ["wells fargo", "wfc"],
    "CVX": ["chevron", "cvx"],
    "MRK": ["merck", "mrk"],
}


def mentions_company(headlines: pd.Series, ticker: str) -> pd.Series:
    """True when the headline names the company or its ticker."""
    aliases = COMPANY_ALIASES.get(ticker, [ticker.lower()])
    pattern = r"\b(?:" + "|".join(aliases) + r")\b"
    return headlines.str.lower().str.contains(pattern, regex=True)


def _suffixed(path: Path, sentiment_model: str) -> Path:
    """FinBERT keeps the default file names; other sentiment models get a suffix."""
    if sentiment_model == "finbert":
        return path
    return path.with_name(f"{path.stem}_{sentiment_model}{path.suffix}")


def daily_sentiment_path(sentiment_model: str = "finbert") -> Path:
    return _suffixed(DAILY_PARQUET, sentiment_model)


SCORING_CHUNK = 5000


def _score_in_chunks(
    texts: List[str], sentiment_model: str, max_chunks: Optional[int] = None
) -> np.ndarray:
    """
    Score with a comparison model, saving each chunk of probabilities to disk so
    an interrupted run (e.g. low memory) resumes where it stopped. Stops after
    max_chunks new chunks if given (raises ChunkLimitReached) for time-boxed runs.
    """
    from src.model_comparison import MODEL_FACTORIES

    chunk_dir = PROCESSED_DATA_DIR / f"_scoring_chunks_{sentiment_model}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = int(np.ceil(len(texts) / SCORING_CHUNK))
    todo = [i for i in range(n_chunks) if not (chunk_dir / f"{i:04d}.npy").exists()]

    if todo:
        clf = MODEL_FACTORIES[sentiment_model]()
        for done, i in enumerate(todo):
            if max_chunks is not None and done >= max_chunks:
                clf.close()
                raise ChunkLimitReached(f"{len(todo) - done} of {n_chunks} chunks left")
            part = texts[i * SCORING_CHUNK : (i + 1) * SCORING_CHUNK]
            np.save(chunk_dir / f"{i:04d}.npy", clf.predict_proba(part))
            print(f"  chunk {i + 1}/{n_chunks} saved", flush=True)
        clf.close()

    matrix = np.vstack([np.load(chunk_dir / f"{i:04d}.npy") for i in range(n_chunks)])
    for f in chunk_dir.glob("*.npy"):
        f.unlink()
    chunk_dir.rmdir()
    return matrix


class ChunkLimitReached(Exception):
    """Raised when a time-boxed scoring run stops early; rerun to continue."""


def score_headlines(
    news: pd.DataFrame, rebuild: bool = False, sentiment_model: str = "finbert"
) -> pd.DataFrame:
    """
    Add class probabilities and a signed score (P(pos) - P(neg)) to each headline.

    sentiment_model: "finbert" (project default) or any model name from
    src.model_comparison.MODEL_FACTORIES, for the downstream model comparison.
    The label column is called finbert_label for every model so the
    aggregation code is shared.
    """
    path = _suffixed(SCORED_PARQUET, sentiment_model)
    if path.exists() and not rebuild:
        scored = pd.read_parquet(path)
        if len(scored) == len(news):
            return scored

    texts = news["headline"].astype(str).tolist()
    print(f"Scoring {len(news):,} headlines with {sentiment_model} on {DEVICE} ...", flush=True)
    if sentiment_model == "finbert":
        from src.sentiment_analyzer import FinBERTSentimentAnalyzer

        probs = FinBERTSentimentAnalyzer(device=DEVICE).predict_proba(texts)
    else:
        from src.model_comparison import LABELS

        matrix = _score_in_chunks(texts, sentiment_model)
        probs = {label: matrix[:, i] for i, label in enumerate(LABELS)}

    scored = news.copy()
    for label in ("positive", "negative", "neutral"):
        scored[f"p_{label}"] = probs[label]
    scored["finbert_label"] = scored[["p_positive", "p_negative", "p_neutral"]].idxmax(axis=1).str[2:]
    scored["sentiment_score"] = scored["p_positive"] - scored["p_negative"]

    scored.to_parquet(path, index=False)
    print(f"[OK] Saved scored headlines to {path}")
    return scored


def fetch_prices(ticker: str, start: str = STUDY_START, end: str = STUDY_END) -> pd.DataFrame:
    """Daily OHLCV from yfinance (cached to data/raw/prices)."""
    import yfinance as yf

    path = PRICES_DIR / f"{ticker}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=True)
    prices, close_col, volume_col = normalize_yfinance_dataframe(raw)
    prices = prices.rename(columns={close_col: "close", volume_col: "volume"})
    for name in ("open", "high", "low"):
        match = next((c for c in prices.columns if c.startswith(name)), None)
        if match:
            prices = prices.rename(columns={match: name})
    prices = prices[["date", "open", "high", "low", "close", "volume"]]
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=False)
    return prices


def assign_trading_day(news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Map each headline to the first trading day whose close it can influence.

    Before 16:00 ET on a trading day -> that day; at/after 16:00, weekends and
    holidays -> the next trading day. Also records hours before that day's close.
    """
    local = news["published_utc"].dt.tz_convert(MARKET_TIMEZONE)
    effective_date = local.dt.normalize() + pd.to_timedelta(
        (local.dt.hour >= MARKET_CLOSE_HOUR).astype(int), unit="D"
    )
    effective_date = effective_date.dt.tz_localize(None)

    days = trading_days.sort_values().values
    pos = np.searchsorted(days, effective_date.values, side="left")
    valid = pos < len(days)

    out = news.loc[valid].copy()
    out["trading_day"] = days[pos[valid]]
    close_time = (
        pd.to_datetime(out["trading_day"]).dt.tz_localize(MARKET_TIMEZONE)
        + pd.Timedelta(hours=MARKET_CLOSE_HOUR)
    )
    out["hours_before_close"] = (
        (close_time - local[valid]).dt.total_seconds() / 3600
    ).clip(lower=0)
    return out


def _aggregate(group: pd.DataFrame) -> pd.Series:
    score = group["sentiment_score"]
    company = group[group["mentions_company"]]["sentiment_score"]
    return pd.Series(
        {
            "news_count": len(group),
            "sent_mean": score.mean(),
            "sent_std": score.std(ddof=0),
            "pos_ratio": (group["finbert_label"] == "positive").mean(),
            "neg_ratio": (group["finbert_label"] == "negative").mean(),
            "company_news_count": len(company),
            "company_sent_mean": company.mean() if len(company) else np.nan,
        }
    )


def add_decayed_sentiment(
    daily: pd.DataFrame, trading_days: pd.DatetimeIndex, half_life: float = DECAY_HALF_LIFE_DAYS
) -> pd.DataFrame:
    """
    Reindex one ticker's daily features to every trading day and add
    count-weighted sentiment that decays across days (no-news days contribute 0).
    """
    full = daily.set_index("trading_day").reindex(trading_days)
    full.index.name = "trading_day"
    full[["news_count", "company_news_count"]] = full[["news_count", "company_news_count"]].fillna(0)

    for prefix, count_col, mean_col in (
        ("sent", "news_count", "sent_mean"),
        ("company_sent", "company_news_count", "company_sent_mean"),
    ):
        total = (full[mean_col].fillna(0) * full[count_col]).ewm(halflife=half_life).mean()
        count = full[count_col].ewm(halflife=half_life).mean()
        full[f"{prefix}_decay"] = (total / count.replace(0, np.nan)).fillna(0)
    return full.reset_index()


def build_daily_sentiment(
    tickers: List[str] = STUDY_TICKERS,
    rebuild_scores: bool = False,
    sentiment_model: str = "finbert",
) -> pd.DataFrame:
    """Full step: load news -> score -> align to trading days -> daily features."""
    news = load_news()
    scored = score_headlines(news, rebuild=rebuild_scores, sentiment_model=sentiment_model)

    aligned_frames, coverage_rows = [], []
    for ticker in tickers:
        prices = fetch_prices(ticker)
        trading_days = pd.DatetimeIndex(pd.to_datetime(prices["date"]))
        aligned = assign_trading_day(scored[scored["ticker"] == ticker], trading_days)
        aligned["mentions_company"] = mentions_company(aligned["headline"], ticker)
        aligned_frames.append(aligned)

        days_with_news = aligned["trading_day"].nunique()
        coverage_rows.append(
            {
                "ticker": ticker,
                "headlines": len(aligned),
                "trading_days": len(trading_days),
                "days_with_news": days_with_news,
                "pct_days_with_news": round(100 * days_with_news / len(trading_days), 1),
                "median_headlines_per_news_day": aligned.groupby("trading_day").size().median(),
                "pct_headlines_naming_company": round(100 * aligned["mentions_company"].mean(), 1),
            }
        )

    aligned = pd.concat(aligned_frames, ignore_index=True)

    daily = (
        aligned.groupby(["ticker", "trading_day"]).apply(_aggregate, include_groups=False)
        .reset_index()
    )
    daily = pd.concat(
        [
            add_decayed_sentiment(
                daily[daily["ticker"] == t].drop(columns="ticker"),
                pd.DatetimeIndex(pd.to_datetime(fetch_prices(t)["date"])),
            ).assign(ticker=t)
            for t in tickers
        ],
        ignore_index=True,
    )
    daily_by_source = (
        aligned.groupby(["source", "ticker", "trading_day"]).apply(_aggregate, include_groups=False)
        .reset_index()
    )
    daily_path = daily_sentiment_path(sentiment_model)
    by_source_path = _suffixed(DAILY_BY_SOURCE_PARQUET, sentiment_model)
    daily.to_parquet(daily_path, index=False)
    daily_by_source.to_parquet(by_source_path, index=False)

    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(coverage_rows)
    label_mix = pd.crosstab(aligned["ticker"], aligned["finbert_label"], normalize="index").round(3)
    if sentiment_model == "finbert":
        coverage.to_csv(COVERAGE_DIR / "trading_day_coverage.csv", index=False)
        label_mix.to_csv(COVERAGE_DIR / "finbert_label_mix_by_ticker.csv")
        source_mix = pd.crosstab(aligned["source"], aligned["finbert_label"], margins=True)
        source_mix.to_csv(COVERAGE_DIR / "finbert_label_mix_by_source.csv")

    print(f"[OK] Daily sentiment ({sentiment_model}): {len(daily):,} ticker-days -> {daily_path}")
    print(f"[OK] Daily sentiment by source: {len(daily_by_source):,} rows -> {by_source_path}")
    print("\nTrading-day coverage:")
    print(coverage.to_string(index=False))
    print(f"\n{sentiment_model} label mix per ticker:")
    print(label_mix.to_string())
    return daily


if __name__ == "__main__":
    build_daily_sentiment()
