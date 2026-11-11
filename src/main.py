"""
Run the full FYP II pipeline end to end.

  1. fnspid_loader        FNSPID headlines for the study tickers (+ coverage tables)
  2. news_sentiment       FinBERT scores -> daily per-ticker sentiment features
  3. ablation             price-only vs price+sentiment (XGBoost, LSTM, class-weighted LSTM)
  4. model_comparison     FinBERT vs other non-agentic models on labelled text
  5. sentiment_downstream the same models as sentiment sources for prediction
  6. source_comparison    FinBERT consistency and predictive value per news outlet
  7. lag_robustness       the main experiments with sentiment lagged by one trading day,
                          the headline count tested apart from the sentiment scores,
                          the correlation of sentiment with returns before, on and after its day,
                          and an event study, a trading test and XGBoost feature importance

Each step caches its outputs, so re-running skips work that is already done
(except the experiments themselves, which are cheap to repeat).

Usage:
    python -m src.main                         # everything
    python -m src.main --steps ablation        # selected steps
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

STEPS = ["data", "sentiment", "ablation", "model_comparison", "downstream", "sources", "robustness"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    args = parser.parse_args()

    if "data" in args.steps:
        from src.fnspid_loader import coverage_report, load_news

        coverage_report(load_news())
    if "sentiment" in args.steps:
        from src.news_sentiment import build_daily_sentiment

        build_daily_sentiment()
    if "ablation" in args.steps:
        from src.ablation import MODEL_RUNNERS, compare_architectures, run_ablation

        for model_name in MODEL_RUNNERS:
            run_ablation(model_name)
        compare_architectures()
    if "model_comparison" in args.steps:
        from src.model_comparison import DATASETS, MODEL_FACTORIES, run_benchmark

        run_benchmark(list(DATASETS), list(MODEL_FACTORIES))
    if "downstream" in args.steps:
        from src.sentiment_downstream import DEFAULT_MODELS, run_downstream

        run_downstream(DEFAULT_MODELS)
    if "sources" in args.steps:
        from src.source_comparison import run_source_comparison

        run_source_comparison()
    if "robustness" in args.steps:
        from src.lag_robustness import run_lagged_ablation, run_lagged_agreement, run_lagged_downstream

        run_lagged_agreement()
        run_lagged_ablation("xgboost")
        run_lagged_ablation("lstm_balanced")
        run_lagged_downstream()

        from src.count_ablation import run_count_ablation

        run_count_ablation()

        from src.relevance_ablation import run as run_relevance
        from src.volatility_target import run as run_volatility

        run_relevance("xgboost")
        run_relevance("lstm_balanced")
        run_volatility()

        import os

        if os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY"):
            from src.timestamped_news import run as run_timestamped

            run_timestamped()
        else:
            print("[SKIP] timestamped_news: set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .env")

        from src.same_day_association import run_same_day_association

        run_same_day_association()

        from src.event_study import run_event_study
        from src.feature_importance import OUTPUT_DIR as FI_DIR, importance_shares, plot as plot_importance

        run_event_study()
        FI_DIR.mkdir(parents=True, exist_ok=True)
        shares = importance_shares()
        shares.to_csv(FI_DIR / "importance_shares.csv", index=False)
        plot_importance(shares, FI_DIR / "feature_importance.png")


if __name__ == "__main__":
    main()
