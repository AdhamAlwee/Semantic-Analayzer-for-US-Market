"""
How strongly is daily FinBERT sentiment related to returns before, on and after its day?

The ablation found no reliable gain from sentiment for predicting future movement,
while market agreement was much higher when the publication day was included.
This module measures that directly: the Spearman rank correlation between a
stock's daily mean sentiment on trading day t and its return over four windows:

  previous day   close t-2 -> close t-1   (news reacting to an earlier move)
  same day       close t-1 -> close t     (the day the headlines are dated)
  next day       close t   -> close t+1   (what a predictor would need)
  day after      close t+1 -> close t+2

Only days with at least one headline are used. Confidence intervals and p-values
come from a moving-block bootstrap (blocks of 10 trading days, resampled within
each stock), and Benjamini-Hochberg q-values are taken over all tests.

Usage:
    python -m src.same_day_association
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_TICKERS
from src.ablation import BOOTSTRAP_BLOCK, BOOTSTRAP_SAMPLES, RANDOM_SEED, benjamini_hochberg
from src.features import build_all

OUTPUT_DIR = RESULTS_DIR / "same_day_association"
WINDOWS = ["previous_day", "same_day", "next_day", "day_after"]


def add_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Returns over each window, relative to the sentiment day t (within each ticker)."""
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker")
    df["previous_day"] = df["ret_lag1"]
    df["same_day"] = df["ret_1"]
    df["next_day"] = df["fwd_ret_1"]
    df["day_after"] = g["fwd_ret_1"].shift(-1)
    return df


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _block_indices(n: int, rng: np.random.Generator, block: int) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 0) + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def bootstrap_spearman(groups, block: int = BOOTSTRAP_BLOCK,
                       n_samples: int = BOOTSTRAP_SAMPLES, seed: int = RANDOM_SEED) -> dict:
    """Spearman rho over the stacked groups, with blocks resampled within each group."""
    rng = np.random.default_rng(seed)
    x_all = np.concatenate([x for x, _ in groups])
    y_all = np.concatenate([y for _, y in groups])
    boots = np.empty(n_samples)
    for i in range(n_samples):
        xs, ys = [], []
        for x, y in groups:
            idx = _block_indices(len(x), rng, block)
            xs.append(x[idx])
            ys.append(y[idx])
        boots[i] = spearman(np.concatenate(xs), np.concatenate(ys))
    return {
        "n_days": len(x_all),
        "rho": spearman(x_all, y_all),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_value": float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))),
    }


def run_same_day_association(sentiment_model: str = "finbert") -> pd.DataFrame:
    df = add_windows(build_all(STUDY_TICKERS, sentiment_model))
    df = df[df["news_count"] > 0]

    rows = []
    for window in WINDOWS:
        per_ticker = []
        for ticker in STUDY_TICKERS:
            d = df[(df["ticker"] == ticker)].dropna(subset=["sent_mean", window])
            pair = (d["sent_mean"].to_numpy(), d[window].to_numpy())
            per_ticker.append(pair)
            rows.append({"window": window, "ticker": ticker, **bootstrap_spearman([pair])})
        rows.append({"window": window, "ticker": "ALL", **bootstrap_spearman(per_ticker)})

    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg(out["p_value"])
    out["significant"] = out["q_value"] < 0.05

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "sentiment_return_correlation.csv", index=False)
    return out


if __name__ == "__main__":
    result = run_same_day_association()
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(result.to_string(index=False))
