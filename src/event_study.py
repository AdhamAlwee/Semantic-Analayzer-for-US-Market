"""
What do news-sentiment days mean in returns, and could they be traded?

1. Event study. Within each stock, days with news are split into quintiles of
   daily mean FinBERT sentiment. For the most positive and most negative
   quintiles, the market-adjusted return (stock minus SPY) is averaged for each
   day from 5 trading days before to 5 days after the news day (day 0 is the
   close of t-1 to the close of t), and cumulated from day -5.

2. Trading test. A fixed, pre-specified rule on the 2021-2023 test years:
   on day t, a stock is held long if its sentiment is in the top quintile and
   short if it is in the bottom quintile, with quintile cut-offs taken from
   2018-2020 only. Each position is market-adjusted (stock minus SPY), held for
   one day and equally weighted across the stocks traded that day. The rule is
   evaluated for three holding days:
     day 0   close t-1 -> close t     (not tradable: the news is not yet known)
     day +1  close t   -> close t+1   (tradable only if the news came before the close)
     day +2  close t+1 -> close t+2   (tradable whatever the publication time)
   Results are reported before costs and after 10 basis points per side
   (20 bp per one-day round trip), against buy-and-hold of the five stocks and SPY.

Usage:
    python -m src.event_study
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_TICKERS
from src.ablation import BOOTSTRAP_BLOCK, BOOTSTRAP_SAMPLES, RANDOM_SEED
from src.features import build_all
from src.news_sentiment import fetch_prices

OUTPUT_DIR = RESULTS_DIR / "event_study"
WINDOW = 5
TEST_START = pd.Timestamp("2021-01-01")
COST_PER_SIDE = 0.0010
TRADING_DAYS = 252


def load_panel() -> pd.DataFrame:
    """Daily panel with market-adjusted returns relative to the sentiment day."""
    df = build_all(STUDY_TICKERS)[["date", "ticker", "ret_1", "sent_mean", "news_count"]]
    spy = fetch_prices("SPY").sort_values("date")
    spy["spy_ret"] = spy["close"].pct_change()
    df = df.merge(spy[["date", "spy_ret"]], on="date", how="left")
    df["ar"] = df["ret_1"] - df["spy_ret"]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker")["ar"]
    for k in range(-WINDOW, WINDOW + 1):
        df[f"ar_{k}"] = g.shift(-k)
    return df


def assign_quintiles(df: pd.DataFrame, cutoff_rows: pd.Series) -> pd.Series:
    """Quintile (1-5) of sent_mean on news days, with cut-offs from cutoff_rows within each ticker."""
    q = pd.Series(np.nan, index=df.index)
    for ticker, d in df.groupby("ticker"):
        ref = d.loc[cutoff_rows.loc[d.index] & (d["news_count"] > 0), "sent_mean"]
        edges = ref.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        news = d["news_count"] > 0
        q.loc[d.index[news]] = 1 + np.searchsorted(edges, d.loc[news, "sent_mean"], side="right")
    return q


def event_study(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quintile"] = assign_quintiles(df, pd.Series(True, index=df.index))
    cols = [f"ar_{k}" for k in range(-WINDOW, WINDOW + 1)]
    rows = []
    for name, qv in [("most positive", 5), ("most negative", 1)]:
        ev = df.loc[df["quintile"] == qv, cols].dropna()
        car = ev.cumsum(axis=1)
        for k, col in zip(range(-WINDOW, WINDOW + 1), cols):
            rows.append({
                "group": name, "day": k, "n_events": len(ev),
                "mean_ar": ev[col].mean(), "se_ar": ev[col].std(ddof=1) / np.sqrt(len(ev)),
                "car": car[col].mean(), "se_car": car[col].std(ddof=1) / np.sqrt(len(ev)),
            })
    out = pd.DataFrame(rows)

    # Difference positive minus negative on each day (Welch t)
    pos, neg = out[out.group == "most positive"].set_index("day"), out[out.group == "most negative"].set_index("day")
    diff = pd.DataFrame({
        "day": pos.index,
        "diff_ar": pos["mean_ar"] - neg["mean_ar"],
        "t_stat": (pos["mean_ar"] - neg["mean_ar"]) / np.sqrt(pos["se_ar"] ** 2 + neg["se_ar"] ** 2),
    }).reset_index(drop=True)
    return out, diff


def plot_event_study(es: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = {"most positive": ("#2F6FB0", "o"), "most negative": ("#D2691E", "s")}
    for name, (color, marker) in styles.items():
        d = es[es.group == name]
        n = int(d["n_events"].iloc[0])
        ax.plot(d["day"], 100 * d["car"], color=color, marker=marker, label=f"{name} sentiment quintile (n = {n:,})")
        ax.fill_between(d["day"], 100 * (d["car"] - 1.96 * d["se_car"]),
                        100 * (d["car"] + 1.96 * d["se_car"]), color=color, alpha=0.15)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(-WINDOW, WINDOW + 1))
    ax.set_xlabel("Trading day relative to the news day (day 0)")
    ax.set_ylabel("Cumulative market-adjusted return (%)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _block_bootstrap_mean(x: np.ndarray, block=BOOTSTRAP_BLOCK, n_samples=BOOTSTRAP_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_samples)
    for i in range(n_samples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        means[i] = x[(starts[:, None] + np.arange(block)).ravel()[:n]].mean()
    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    return np.percentile(means, 2.5), np.percentile(means, 97.5), min(1.0, p)


def _summary(name: str, daily: pd.Series, active: pd.Series = None) -> dict:
    x = daily.to_numpy()
    lo, hi, p = _block_bootstrap_mean(x)
    sd = x.std(ddof=1)
    return {
        "strategy": name,
        "days": len(x),
        "active_days": int(active.sum()) if active is not None else len(x),
        "ann_return_pct": 100 * x.mean() * TRADING_DAYS,
        "ann_ci_low_pct": 100 * lo * TRADING_DAYS,
        "ann_ci_high_pct": 100 * hi * TRADING_DAYS,
        "ann_vol_pct": 100 * sd * np.sqrt(TRADING_DAYS),
        "sharpe": (x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "p_value": p,
    }


def trading_test(df: pd.DataFrame):
    df = df.copy()
    df["quintile"] = assign_quintiles(df, df["date"] < TEST_START)
    df["signal"] = np.select([df["quintile"] == 5, df["quintile"] == 1], [1.0, -1.0], 0.0)
    test = df[df["date"] >= TEST_START]
    dates = np.sort(test["date"].unique())

    rows, curves = [], {}
    for label, col in [("Day 0 (not tradable)", "ar_0"), ("Day +1", "ar_1"), ("Day +2 (always tradable)", "ar_2")]:
        pos = test[(test["signal"] != 0) & test[col].notna()].copy()
        pos["gross"] = pos["signal"] * pos[col]
        pos["net"] = pos["gross"] - 2 * COST_PER_SIDE
        daily = pos.groupby("date")[["gross", "net"]].mean().reindex(dates, fill_value=0.0)
        active = pos.groupby("date").size().reindex(dates, fill_value=0) > 0
        rows.append(_summary(f"{label}, before costs", daily["gross"], active))
        rows.append(_summary(f"{label}, after costs", daily["net"], active))
        curves[f"{label}, before costs"] = daily["gross"]
        curves[f"{label}, after costs"] = daily["net"]

    # Benchmarks: equal-weight buy-and-hold of the five stocks, and SPY (raw returns)
    bh = test.groupby("date")["ret_1"].mean().reindex(dates)
    spy = test.groupby("date")["spy_ret"].first().reindex(dates)
    rows.append(_summary("Buy and hold, five stocks", bh))
    rows.append(_summary("Buy and hold, SPY", spy))
    return pd.DataFrame(rows), pd.DataFrame(curves, index=pd.DatetimeIndex(dates))


def plot_strategies(curves: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = {
        "Day 0 (not tradable), before costs": ("#9AA8BF", "--"),
        "Day +1, before costs": ("#2F6FB0", "-"),
        "Day +1, after costs": ("#2F6FB0", ":"),
        "Day +2 (always tradable), before costs": ("#D2691E", "-"),
        "Day +2 (always tradable), after costs": ("#D2691E", ":"),
    }
    for name, (color, ls) in styles.items():
        ax.plot(curves.index, 100 * curves[name].cumsum(), color=color, linestyle=ls, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative market-adjusted return (%)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_event_study() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    es, diff = event_study(df)
    es.to_csv(OUTPUT_DIR / "event_study_car.csv", index=False)
    diff.to_csv(OUTPUT_DIR / "event_study_difference.csv", index=False)
    plot_event_study(es, OUTPUT_DIR / "event_study.png")

    summary, curves = trading_test(df)
    summary.to_csv(OUTPUT_DIR / "trading_test.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "trading_test_daily.csv")
    plot_strategies(curves, OUTPUT_DIR / "trading_test.png")
    return es, diff, summary


if __name__ == "__main__":
    es, diff, summary = run_event_study()
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(es.to_string(index=False))
        print(diff.to_string(index=False))
        print(summary.to_string(index=False))
