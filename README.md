# Financial News Sentiment Analysis Using FinBERT for Stock Price Movement Prediction

A Final Year Project (FYP) for predicting US stock price movements by analyzing financial news sentiment using FinBERT.

**Student:** Mas Hairul Adham Bin Alwee (ID: 22005746)  
**Supervisor:** Ts. Dr. Nurul Aida Binti Osman  
**Institution:** Universiti Teknologi PETRONAS (UTP)

---

## Project Overview

This project aims to predict US stock price movements using financial news sentiment analysis. Financial news articles are analyzed using **FinBERT** (a financial-domain language model) to classify sentiments as positive, negative, or neutral. The sentiment results are combined with historical stock market data to develop a prediction model capable of forecasting stock price trends.

### Problem Statement

Stock prices are influenced by financial news and market sentiment. Traditional prediction methods mainly rely on historical stock data and often ignore the impact of news sentiment. Existing sentiment analysis approaches may not accurately understand financial language, reducing prediction accuracy.

### Objectives

1. To collect and preprocess financial news and historical stock market data
2. To implement FinBERT for financial news sentiment classification
3. To develop a stock price movement prediction model using sentiment data
4. To evaluate the model performance using Accuracy, Precision, Recall, F1-score, and RMSE

---

## Project Structure

```
Semantic-Analayzer-for-US-Market/
├── data/                      # Data directory
│   ├── raw/                   # Raw data (news, stock data)
│   └── processed/             # Preprocessed data
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── data_pipeline.py       # End-to-end data collection pipeline
│   ├── sentiment_analyzer.py  # FinBERT sentiment analysis
│   ├── prediction_model.py    # ML prediction models
│   ├── main.py                # Full pipeline entry point
│   └── utils.py               # Helper functions
├── notebooks/                 # Jupyter notebooks for development
├── models/                    # Trained models and checkpoints
├── results/                   # Results, metrics, and visualizations
├── config/                    # Configuration files
│   └── config.py              # Project settings
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Semantic-Analayzer-for-US-Market
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or install the project package (includes dependencies from `requirements.txt`):
   ```bash
   pip install .
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root (loaded automatically by `config/config.py`):
   ```
   NEWS_API_KEY=your_news_api_key
   ALPHA_VANTAGE_KEY=your_alpha_vantage_key
   ```

---

## Key Technologies

| Component | Technology |
|---|---|
| **Programming Language** | Python 3.8+ |
| **NLP Model** | FinBERT (ProsusAI/finbert) |
| **Deep Learning Framework** | PyTorch / Transformers |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Stock Data** | yfinance |
| **Development Environment** | VS Code / Google Colab |

---

## Core Modules

### `src/data_loader.py`
Handles loading and preprocessing of stock market data (yfinance) and financial news CSVs.

### `src/data_pipeline.py`
Collects stock data, generates synthetic news for demos, merges by date, and preprocesses features.

### `src/sentiment_analyzer.py`
FinBERT sentiment classification using label mappings from the model config (0=positive, 1=negative, 2=neutral).

### `src/prediction_model.py`
Random Forest / Gradient Boosting ensemble for next-day price direction (`Price_Up`).

### `config/config.py`
Paths, hyperparameters, API keys, `DEVICE` (auto-selects CUDA when available), and `SYNTHETIC_SENTIMENT_LABELS` for the demo news generator.

---

## Usage

### Run full pipeline
```bash
python -m src.main
```

### Build real-news sentiment features (FNSPID + FinBERT)
```bash
python -m src.fnspid_loader     # download FNSPID (~29 GB) and extract headlines for STUDY_TICKERS
python -m src.news_sentiment    # FinBERT-score headlines -> daily per-ticker features
```
Outputs `data/processed/daily_sentiment.parquet` (and `daily_sentiment_by_source.parquet`); coverage tables go to `results/data_coverage/`.

### Ablation: price-only vs price + sentiment (XGBoost and LSTM, walk-forward)
```bash
python -m src.ablation                 # both architectures + comparison table
python -m src.ablation --model lstm    # one architecture
```
Tests on 2021, 2022 and 2023 (each trained on all earlier years), at 1-, 3- and 5-day horizons, with down/stable/up and return-regression targets. Writes metrics, block-bootstrap confidence intervals, feature importances and `ablation_delta_f1.png` to `results/ablation/`.

### Compare FinBERT against other (non-agentic) models
```bash
python -m src.model_comparison                 # 7 models x 2 labelled datasets
python -m src.model_comparison --limit 200     # quick run
python -m src.model_comparison --extra-llm Qwen/Qwen2.5-7B-Instruct   # larger LLM (Colab GPU)
```
Writes accuracy/F1, per-class scores, confusion matrices and inter-model agreement (Cohen's kappa) to `results/model_comparison/`.

### Basic Sentiment Analysis
```python
from config.config import DEVICE
from src.sentiment_analyzer import FinBERTSentimentAnalyzer

analyzer = FinBERTSentimentAnalyzer(device=DEVICE)
result = analyzer.analyze_sentiment("Great earnings report!")
print(result)  # {'text': '...', 'sentiment': 'positive', ...}
```

### Loading Stock Data
```python
from src.data_loader import StockDataLoader

loader = StockDataLoader(ticker="AAPL", start_date="2024-01-01", end_date="2024-12-31")
data = loader.fetch_stock_data()
processed_data = loader.preprocess_stock_data(data)
```

---

## Evaluation Metrics

The model will be evaluated using:
- **Accuracy:** Overall correctness of predictions
- **Precision:** Ratio of true positives to predicted positives
- **Recall:** Ratio of true positives to actual positives
- **F1-Score:** Harmonic mean of precision and recall
- **RMSE:** Root mean squared error for price predictions

---

## Development Status

- [x] Project structure setup
- [x] Core modules created
- [x] Configuration framework
- [x] Data collection pipeline
- [x] FinBERT integration
- [x] Prediction model development
- [x] Model evaluation (classification metrics)
- [ ] Live news API integration
- [ ] Results and visualization (extended)

---

## Contributing

This is an academic FYP project. Contributions and feedback are welcome from supervisors and committee members.

---

## License

This project is part of the Final Year Project (FYP) at Universiti Teknologi PETRONAS.

---

## Contact

**Author:** Mas Hairul Adham Bin Alwee  
**Email:** mas_22005746@utp.edu.my  
**Student ID:** 22005746
