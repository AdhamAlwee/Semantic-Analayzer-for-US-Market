"""
Downstream sentiment-model comparison: does the choice of sentiment model change
how much sentiment helps price-movement prediction?

Extends the text-level benchmark (src/model_comparison.py) to the prediction task.
Each non-agentic sentiment model scores the same FNSPID headlines; its daily
features replace FinBERT's in the XGBoost ablation (same splits, horizons,
hyperparameters). Every model is compared to the same price-only baseline,
so the only thing that changes is where the sentiment numbers come from.

Usage:
    python -m src.sentiment_downstream
    python -m src.sentiment_downstream --models finbert vader
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_TICKERS
from src.ablation import benjamini_hochberg, run_ticker_horizon, summarise
from src.features import HORIZONS, PRICE_FEATURES, SENTIMENT_FEATURES, build_all
from src.model_comparison import MODEL_FACTORIES
from src.news_sentiment import (
    SCORED_PARQUET,
    _suffixed,
    build_daily_sentiment,
    daily_sentiment_path,
)

OUTPUT_DIR = RESULTS_DIR / "sentiment_downstream"
DEFAULT_MODELS = list(MODEL_FACTORIES)  # finbert + the six comparison models


def ensure_daily_features(models: List[str]) -> None:
    """Score headlines and build daily features for any model that lacks them."""
    for model in models:
        if not daily_sentiment_path(model).exists():
            build_daily_sentiment(sentiment_model=model)


def label_agreement_on_news(models: List[str]) -> pd.DataFrame:
    """Cohen's kappa between models' labels on the real FNSPID headlines."""
    labels = {
        m: pd.read_parquet(_suffixed(SCORED_PARQUET, m), columns=["finbert_label"])["finbert_label"]
        for m in models
    }
    kappa = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            kappa.loc[a, b] = cohen_kappa_score(labels[a], labels[b])
    mix = pd.DataFrame({m: labels[m].value_counts(normalize=True) for m in models}).T
    return kappa.round(3), mix.round(3)


def run_downstream(models: List[str], tickers: List[str] = STUDY_TICKERS) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_daily_features(models)

    all_preds = []
    for i, model in enumerate(models):
        dataset = build_all(tickers, sentiment_model=model)
        # Price-only is identical for every sentiment model, so train it once
        feature_sets = {f"price+{model}": PRICE_FEATURES + SENTIMENT_FEATURES}
        if i == 0:
            feature_sets = {"price": PRICE_FEATURES, **feature_sets}
        for ticker in tickers:
            df = dataset[dataset["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            for h in HORIZONS:
                preds, _ = run_ticker_horizon(df, h, "xgboost", feature_sets=feature_sets)
                all_preds.append(preds.assign(ticker=ticker, horizon=h))
        print(f"[OK] XGBoost ablation done with {model} sentiment", flush=True)

    preds = pd.concat(all_preds, ignore_index=True)
    preds["series"] = preds["ticker"]
    summary, _, deltas = summarise(pd.concat([preds, preds.assign(ticker="ALL")]))
    deltas["sentiment_model"] = deltas["compared_to_price"].str.replace("price+", "", regex=False)
    deltas["f1_q_bh"] = benjamini_hochberg(deltas["f1_p_value"])
    deltas["rmse_q_bh"] = benjamini_hochberg(deltas["rmse_p_value"])

    preds.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "deltas_vs_price.csv", index=False)

    per_stock = deltas[deltas["ticker"] != "ALL"]
    table = (
        deltas[deltas["ticker"] == "ALL"]
        .pivot_table(index="sentiment_model", columns="horizon", values="f1_delta")
        .add_prefix("dF1_h")
        .join(per_stock.groupby("sentiment_model")["f1_delta"]
              .apply(lambda d: int((d > 0).sum())).rename("cells_improved_of_15"))
        .join(deltas[deltas["ticker"] == "ALL"].groupby("sentiment_model")["f1_q_bh"].min()
              .rename("min_q_pooled"))
        .sort_values("dF1_h5", ascending=False)
    )
    text_f1 = text_benchmark_f1()
    if text_f1 is not None:
        table = table.join(text_f1)
    table.round(4).to_csv(OUTPUT_DIR / "downstream_comparison.csv")

    kappa, mix = label_agreement_on_news(models)
    kappa.to_csv(OUTPUT_DIR / "label_agreement_kappa_fnspid.csv")
    mix.to_csv(OUTPUT_DIR / "label_mix_fnspid.csv")
    plot_downstream(deltas, models, OUTPUT_DIR / "downstream_delta_f1.png")

    pd.set_option("display.width", 200)
    print("\nPooled Δ macro-F1 vs price-only (XGBoost), by sentiment model:")
    print(table.round(4).to_string())
    print("\nLabel agreement on FNSPID headlines (Cohen's kappa):")
    print(kappa.to_string())
    print("\nLabel mix on FNSPID headlines:")
    print(mix.to_string())
    print(f"\n[OK] Results written to {OUTPUT_DIR}")
    return table


def text_benchmark_f1() -> pd.DataFrame:
    """Macro-F1 on the uncontaminated text benchmark, for 'better text model -> better prediction?'."""
    path = RESULTS_DIR / "model_comparison" / "summary.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path)
    s = s[s["dataset"] == "twitter_fin_news"].set_index("model")["macro_f1"]
    return s.rename("text_macro_f1_twitter").to_frame()


def plot_downstream(deltas: pd.DataFrame, models: List[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = deltas[deltas["ticker"] == "ALL"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(HORIZONS)
    x = np.arange(len(models))
    for i, h in enumerate(HORIZONS):
        d = pooled[pooled["horizon"] == h].set_index("sentiment_model").loc[models]
        err = np.vstack([d["f1_delta"] - d["f1_ci_low"], d["f1_ci_high"] - d["f1_delta"]])
        ax.bar(x + (i - 1) * width, d["f1_delta"], width, yerr=err, capsize=3, label=f"{h}-day")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("Δ macro-F1 vs price-only (95% CI)")
    ax.set_title("Sentiment source vs predictive gain (XGBoost, 5 stocks pooled, 2021-2023)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    args = parser.parse_args()
    run_downstream(args.models)
