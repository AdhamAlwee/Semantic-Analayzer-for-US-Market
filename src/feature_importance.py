"""
How much does XGBoost rely on the sentiment features?

Reads the gain importances saved by src.ablation for the price+sentiment
models, expresses each feature's gain as a share of the model's total gain,
and averages the shares over the five stocks, three horizons and three test
years.

Usage:
    python -m src.feature_importance
"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR
from src.features import SENTIMENT_FEATURES

IMPORTANCE_CSV = RESULTS_DIR / "ablation" / "xgboost" / "feature_importance.csv"
OUTPUT_DIR = RESULTS_DIR / "feature_importance"
FEATURE_SET = "price+sentiment"


def importance_shares() -> pd.DataFrame:
    imp = pd.read_csv(IMPORTANCE_CSV)
    imp = imp[imp["feature_set"] == FEATURE_SET].copy()
    imp["share"] = imp["importance_gain"] / imp.groupby(["ticker", "horizon"])["importance_gain"].transform("sum")
    by_h = imp.pivot_table(index="feature", columns="horizon", values="share", aggfunc="mean")
    by_h.columns = [f"share_{h}d" for h in by_h.columns]
    by_h["share_mean"] = by_h.mean(axis=1)
    by_h["group"] = ["sentiment" if f in SENTIMENT_FEATURES else "price" for f in by_h.index]
    return by_h.sort_values("share_mean", ascending=False).reset_index()


def plot(shares: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = shares.iloc[::-1]
    colors = ["#D2691E" if g == "sentiment" else "#2F6FB0" for g in d["group"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(d["feature"], 100 * d["share_mean"], color=colors)
    ax.set_xlabel("Share of total XGBoost gain importance (%), mean over stocks and horizons")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2F6FB0", label="price feature"),
                       Patch(color="#D2691E", label="sentiment feature")], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s = importance_shares()
    s.to_csv(OUTPUT_DIR / "importance_shares.csv", index=False)
    plot(s, OUTPUT_DIR / "feature_importance.png")
    print(s.to_string(index=False))
    print(s.groupby("group")["share_mean"].sum())
