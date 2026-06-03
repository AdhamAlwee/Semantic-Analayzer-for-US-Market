"""
Main execution script for the entire pipeline
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DEVICE, MODELS_DIR, RESULTS_DIR
from src.data_pipeline import FinancialDataPipeline
from src.prediction_model import EnsemblePredictor
from src.sentiment_analyzer import FinBERTSentimentAnalyzer
from src.utils import log_message, save_data


def main():
    """Main execution pipeline."""
    log_message("Starting Financial News Sentiment Analysis Pipeline", "INFO")
    start_time = datetime.now()

    TICKERS = ["AAPL", "MSFT", "GOOGL"]
    DAYS_BACK = 365

    log_message("Step 1: Collecting and processing data", "INFO")
    pipeline = FinancialDataPipeline(tickers=TICKERS, days_back=DAYS_BACK)
    processed_data = pipeline.run_pipeline(save_intermediate=True)

    if not processed_data:
        log_message("No data available, exiting", "ERROR")
        return

    log_message("Step 2: Training prediction models", "INFO")
    all_results = {}

    for ticker, data in processed_data.items():
        log_message(f"Training model for {ticker}", "INFO")
        ensemble = EnsemblePredictor()
        results = ensemble.train_all(data, test_size=0.2)
        model_dir = MODELS_DIR / ticker
        ensemble.save_all_models(str(model_dir))
        all_results[ticker] = results

    log_message("Step 3: Analyzing sentiment with FinBERT", "INFO")

    try:
        analyzer = FinBERTSentimentAnalyzer(device=DEVICE)
        sample_headlines = [
            "Strong earnings beat expectations",
            "Company announces new innovative product",
            "Missed earnings estimates",
            "Stock trading at historical average",
        ]

        sentiment_results = []
        for headline in sample_headlines:
            result = analyzer.analyze_sentiment(headline, return_scores=True)
            sentiment_results.append(result)
            log_message(f"'{headline}' -> {result['sentiment']}", "INFO")

        save_data(pd.DataFrame(sentiment_results), str(RESULTS_DIR / "sentiment_analysis.csv"))

    except Exception as e:
        log_message(f"Sentiment analysis error: {e}", "WARNING")

    log_message("Step 4: Generating results summary", "INFO")

    summary = {
        "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers_processed": TICKERS,
        "data_period_days": DAYS_BACK,
        "models_trained": list(all_results.keys()),
        "training_results": {},
    }

    for ticker, results in all_results.items():
        summary["training_results"][ticker] = {
            "random_forest": {
                "test_accuracy": results["random_forest"]["metrics"]["test"]["accuracy"],
                "test_f1": results["random_forest"]["metrics"]["test"]["f1"],
            },
            "gradient_boosting": {
                "test_accuracy": results["gradient_boosting"]["metrics"]["test"]["accuracy"],
                "test_f1": results["gradient_boosting"]["metrics"]["test"]["f1"],
            },
        }

    summary_path = RESULTS_DIR / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    log_message(f"Summary saved to {summary_path}", "INFO")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 80)
    log_message(f"Pipeline completed in {duration:.2f} seconds", "INFO")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    main()
