# Financial News Sentiment Analysis Using FinBERT for Stock Price Movement Prediction

A Final Year Project (FYP) for predicting US stock price movements by analyzing financial news sentiment using FinBERT.

**Student:** Mas Hairul Adham Bin Alwee (ID: 22005746)  
**Supervisor:** Ts. Dr. Nurul Aida Binti Osman  
**Institution:** Universiti Teknologi PETRONAS (UTP)

> A full account of the FYP II work, results and changes from the interim report is in [FYP2_WORK_SUMMARY.md](FYP2_WORK_SUMMARY.md).

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
├── config/config.py              # Paths, study tickers/window, FNSPID settings, DEVICE
├── src/
│   ├── main.py                   # Runs the whole FYP II pipeline
│   ├── fnspid_loader.py          # FNSPID download, headline extraction, coverage scans
│   ├── sentiment_analyzer.py     # FinBERT wrapper (labels + probabilities)
│   ├── news_sentiment.py         # Headline scoring -> trading-day alignment -> daily features
│   ├── features.py               # Price features, sentiment features, forward targets
│   ├── ablation.py               # Price-only vs price+sentiment, walk-forward, bootstrap tests
│   ├── lstm_model.py             # LSTM (and class-weighted LSTM) for the ablation
│   ├── model_comparison.py       # FinBERT vs 6 non-agentic models + a naive Bayes baseline on labelled text
│   ├── sentiment_downstream.py   # The same 8 models as sentiment sources for prediction
│   ├── source_comparison.py      # FinBERT consistency and predictive value per news outlet
│   ├── lag_robustness.py         # Experiments repeated with sentiment lagged by one trading day
│   ├── count_ablation.py         # Headline count vs sentiment scores
│   ├── relevance_ablation.py     # Sentiment from headlines naming the company only
│   ├── volatility_target.py      # Predicting the size of the move instead of its direction
│   ├── same_day_association.py   # Sentiment vs returns before, on and after the news day
│   ├── event_study.py            # Event study and long-short trading test with costs
│   ├── feature_importance.py     # XGBoost gain importance by feature
│   └── utils.py                  # yfinance/date helpers
├── app/demo.py                   # Streamlit demo: score a headline, explore a stock, key results
├── data/                         # raw/ (FNSPID ~29 GB, prices) and processed/ (git-ignored)
├── results/                      # Committed result tables and figures (see below)
├── legacy/                       # FYP I synthetic-data prototype, kept for reference only
├── requirements.txt
└── setup.py
```

---

## Installation

Python 3.12 is recommended (PyTorch wheels). A CUDA GPU is optional but makes FinBERT scoring and the LLM comparison much faster.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or the CPU build
pip install -r requirements.txt
```

---

## Study Design

| | |
|---|---|
| **Tickers** | INTC (tech), SBUX (consumer), WFC (banking), CVX (energy), MRK (healthcare): one per sector, picked for dense FNSPID coverage in every year (`results/data_coverage/fnspid_all_tickers_by_year.csv`) |
| **Window** | 2018-01-01 to 2023-12-31 (1,509 trading days per ticker) |
| **News** | 34,810 FNSPID headlines (Dong, Fan & Peng, 2024); outlet taken from the URL domain |
| **Sentiment** | `ProsusAI/finbert`, no fine-tuning; daily mean, within-day std, positive/negative ratios, counts, 3-day-half-life decayed sentiment, company-mention subset |
| **Targets** | down / stable / up and % return at 1, 3 and 5 trading days |
| **Models** | XGBoost, LSTM, class-weighted LSTM; identical settings with and without sentiment |
| **Evaluation** | Walk-forward: test 2021, 2022, 2023, each trained on earlier years with an h-day purge; accuracy, macro precision/recall/F1, RMSE; paired moving-block bootstrap CIs; Benjamini-Hochberg correction |

---

## Usage

```bash
python -m src.main                                   # everything, in order
python -m src.main --steps ablation downstream       # selected steps
```

Or step by step:

```bash
python -m src.fnspid_loader         # download FNSPID and extract headlines for STUDY_TICKERS
python -m src.news_sentiment        # FinBERT scores -> data/processed/daily_sentiment.parquet
python -m src.ablation              # XGBoost, LSTM, class-weighted LSTM (+ --model to pick one)
python -m src.model_comparison      # 8 sentiment models x 2 labelled datasets
python -m src.sentiment_downstream  # the 8 models as sentiment sources for prediction
python -m src.source_comparison     # news-outlet comparison (2018 to mid-2020)
python -m src.lag_robustness        # experiments repeated with sentiment lagged by one day
python -m src.count_ablation        # headline count vs sentiment scores
python -m src.relevance_ablation    # company-naming headlines only (XGBoost + class-weighted LSTM)
python -m src.volatility_target     # size of the move as the target (RMSE, large-move F1)
python -m src.same_day_association  # sentiment vs returns before, on and after the news day
python -m src.event_study           # event study and trading test with transaction costs
python -m src.feature_importance    # XGBoost gain importance by feature
```

Interactive demo (score a headline, explore a stock, key results):

```bash
streamlit run app/demo.py
```

A larger prompted LLM can be added to the text benchmark on a bigger GPU (e.g. Colab):
```bash
python -m src.model_comparison --extra-llm Qwen/Qwen2.5-7B-Instruct
```

---

## Results

| Folder | Contents |
|---|---|
| `results/data_coverage/` | FNSPID coverage per ticker, year and source; FinBERT label mix |
| `results/ablation/<model>/` | Metrics, per-year metrics, Δ vs price-only with CIs and q-values, predictions, `ablation_delta_f1.png` |
| `results/ablation/architecture_comparison.csv` | Pooled results for all architectures side by side |
| `results/model_comparison/` | Text-level benchmark: accuracy/F1, per-class scores, confusion matrices, agreement (Cohen's kappa) |
| `results/sentiment_downstream/` | Predictive gain from each sentiment model, label agreement on FNSPID headlines |
| `results/source_comparison/` | Market agreement per outlet, cross-outlet correlation, per-outlet ablation |
| `results/lag_robustness/` | Ablation, model comparison and source agreement with sentiment lagged by one day |
| `results/count_ablation/` | Gain from the headline count vs the sentiment scores |
| `results/relevance_ablation/<model>/` | Gain from company-naming headlines only vs all headlines |
| `results/volatility_target/` | Size-of-move prediction: RMSE and large-move F1 with and without sentiment |
| `results/same_day_association/` | Correlation of sentiment with returns before, on and after the news day |
| `results/event_study/` | Market-adjusted returns around extreme sentiment days; trading test with costs |

### Main finding

Hypothesis: adding FinBERT sentiment to a price-only model significantly improves movement prediction (macro-F1) at 1, 3 or 5 trading days. **Not supported.**

- No significant gain for XGBoost or the class-weighted LSTM, same-day or lagged, after Benjamini-Hochberg correction. The plain LSTM's significant gain is a class-imbalance artifact.
- Also null: headline count vs scores, company-naming headlines only, size of the move as the target, and all 8 sentiment models.
- Sentiment correlates with the same day's return (ρ = 0.133) but barely with the next day's (0.027); a sentiment trading rule loses money after costs. Consistent with the efficient market hypothesis.

---

## Development Status

- [x] FNSPID data collection and coverage analysis
- [x] FinBERT sentiment features (daily, decayed, company-specific, per source)
- [x] Leakage-free walk-forward ablation at 1/3/5-day horizons (XGBoost, LSTM)
- [x] FinBERT vs non-agentic models and a naive Bayes baseline: text benchmark and downstream prediction
- [x] News-source comparison
- [x] Robustness checks: lagged sentiment, headline count, company-only headlines, size-of-move target
- [x] Dissertation and technical paper

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
