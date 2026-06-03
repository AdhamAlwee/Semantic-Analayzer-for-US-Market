"""
Configuration settings for the Financial News Sentiment Analysis project
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configuration
FINBERT_MODEL_NAME = "ProsusAI/finbert"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

# Synthetic news pipeline labels (create_synthetic_news_data in data_pipeline.py)
SYNTHETIC_SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# FinBERT (ProsusAI/finbert) uses id2label: 0=positive, 1=negative, 2=neutral — see sentiment_analyzer.py

# Stock Market Data
US_STOCK_INDICES = ["^GSPC", "^IXIC", "^DJI"]  # S&P 500, NASDAQ, Dow Jones
DEFAULT_TICKER = "AAPL"

# Data Sources
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Device Configuration (CUDA only when available)
def _resolve_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _resolve_device()
