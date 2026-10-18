"""
Run the full FYP II pipeline end to end.

  1. fnspid_loader        FNSPID headlines for the study tickers (+ coverage tables)
  2. news_sentiment       FinBERT scores -> daily per-ticker sentiment features
  3. ablation             price-only vs price+sentiment (XGBoost, LSTM, class-weighted LSTM)
  4. model_comparison     FinBERT vs other non-agentic models on labelled text
  5. sentiment_downstream the same models as sentiment sources for prediction
  6. source_comparison    FinBERT consistency and predictive value per news outlet

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

STEPS = ["data", "sentiment", "ablation", "model_comparison", "downstream", "sources"]


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


if __name__ == "__main__":
    main()
