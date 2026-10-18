# Legacy prototype (FYP I) — not used for any reported result

This folder keeps the first prototype for reference only:

- `src/data_pipeline.py`, `src/main.py`: generated **synthetic** headlines with hard-coded sentiment labels and trained on them
- `src/prediction_model.py`: Random Forest / Gradient Boosting with a shuffled (non-chronological) train/test split
- `src/data_loader.py`: an earlier yfinance loader
- `notebooks/01_complete_pipeline_example.ipynb`: notebook walkthrough of that prototype
- `models/`: models trained on the synthetic data (AAPL, MSFT, GOOGL)

It was replaced because synthetic sentiment carries no real information, and a shuffled split leaks future data into training.

The FYP II pipeline lives in `src/` (see the main README). The files here import modules by their old paths, so they no longer run as-is.
