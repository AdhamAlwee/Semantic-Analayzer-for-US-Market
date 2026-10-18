"""
Model-ready dataset: price features + FinBERT sentiment features + forward targets.

Every feature on row t uses only information available at day t's close
(prices up to t, headlines dated up to t). Targets look forward h trading days.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import STUDY_TICKERS
from src.news_sentiment import DAILY_PARQUET, daily_sentiment_path, fetch_prices

HORIZONS = [1, 3, 5]  # trading days: 1-day, 3-day, 1-week

PRICE_FEATURES = [
    "ret_1",
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag4",
    "ret_lag5",
    "ret_5",
    "volatility_20",
    "ma_ratio_20",
    "ma_ratio_50",
    "rsi_14",
    "volume_ratio_20",
    "log_volume_change",
    "hl_range",
]

SENTIMENT_FEATURES = [
    "news_count_log",
    "sent_mean",
    "sent_std",
    "pos_ratio",
    "neg_ratio",
    "sent_decay",
]

COMPANY_SENTIMENT_FEATURES = [
    "company_news_count_log",
    "company_sent_mean",
    "company_sent_decay",
]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Technical features from daily OHLCV (stationary ratios, not raw price levels)."""
    df = prices.sort_values("date").reset_index(drop=True).copy()
    close, volume = df["close"], df["volume"].astype(float)

    df["ret_1"] = close.pct_change()
    for k in range(1, 6):
        df[f"ret_lag{k}"] = df["ret_1"].shift(k)
    df["ret_5"] = close.pct_change(5)
    df["volatility_20"] = df["ret_1"].rolling(20).std()
    df["ma_ratio_20"] = close / close.rolling(20).mean() - 1
    df["ma_ratio_50"] = close / close.rolling(50).mean() - 1
    df["rsi_14"] = rsi(close)
    df["volume_ratio_20"] = volume / volume.rolling(20).mean()
    df["log_volume_change"] = np.log(volume.replace(0, np.nan)).diff()
    df["hl_range"] = (df["high"] - df["low"]) / close
    return df


def add_targets(df: pd.DataFrame, horizons: List[int] = HORIZONS) -> pd.DataFrame:
    """Forward return over h trading days: close[t+h] / close[t] - 1 (NaN at the end)."""
    for h in horizons:
        df[f"fwd_ret_{h}"] = df["close"].shift(-h) / df["close"] - 1
    return df


def build_dataset(ticker: str, daily_sentiment: pd.DataFrame = None) -> pd.DataFrame:
    """One row per trading day for a ticker, with all features and targets."""
    if daily_sentiment is None:
        daily_sentiment = pd.read_parquet(DAILY_PARQUET)

    df = add_targets(price_features(fetch_prices(ticker)))
    sent = daily_sentiment[daily_sentiment["ticker"] == ticker].drop(columns="ticker")
    df = df.merge(sent, left_on="date", right_on="trading_day", how="left").drop(columns="trading_day")

    # Days without news: zero counts and neutral (0) sentiment; decay features carry over
    zero_fill = ["news_count", "company_news_count", "sent_mean", "sent_std", "pos_ratio",
                 "neg_ratio", "company_sent_mean", "sent_decay", "company_sent_decay"]
    df[zero_fill] = df[zero_fill].fillna(0)
    df["news_count_log"] = np.log1p(df["news_count"])
    df["company_news_count_log"] = np.log1p(df["company_news_count"])

    df = df.dropna(subset=PRICE_FEATURES).reset_index(drop=True)
    df["ticker"] = ticker
    return df


def build_all(tickers: List[str] = STUDY_TICKERS, sentiment_model: str = "finbert") -> pd.DataFrame:
    """All tickers; sentiment features come from the chosen sentiment model's daily file."""
    daily = pd.read_parquet(daily_sentiment_path(sentiment_model))
    return pd.concat([build_dataset(t, daily) for t in tickers], ignore_index=True)
