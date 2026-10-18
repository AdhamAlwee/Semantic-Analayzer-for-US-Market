# FYP II Work Summary

This document records what was built and found in FYP II (2026), and how it maps to the commitments in the interim report. For installation and usage, see [README.md](README.md).

---

## 1. Starting point

At the end of FYP I the repository was a prototype:

- It used **synthetic news headlines** whose sentiment labels were fixed in advance.
- It trained Random Forest and Gradient Boosting models on AAPL, GOOGL and MSFT.
- It had no real news data, no controlled "with vs without sentiment" test, and no comparison with other AI models.

The interim report commits FYP II to:

- real FNSPID news scored by FinBERT
- an LSTM and XGBoost, each tested with and without sentiment (an ablation)
- predictions at 1, 3 and 5 days
- results compared across news sources

The supervisor also asked for a comparison against **other AI models that are not agentic**. All of this is now built and run. The FYP I code is kept in [`legacy/`](legacy/README.md).

---

## 2. Comparing FinBERT with other AI models on labelled text

**Code:** `src/model_comparison.py` → `results/model_comparison/`

FinBERT was tested against six other non-agentic models. Each one reads a headline and returns positive, negative or neutral, with no tools and no multi-step reasoning.

| Model | Type |
|---|---|
| ProsusAI/finbert | Finance-tuned BERT (the main model) |
| yiyanghkust/finbert-tone | A second FinBERT, trained on analyst reports |
| DistilRoBERTa-financial | Small transformer fine-tuned on finance text |
| cardiffnlp RoBERTa-Twitter | General social-media sentiment model |
| VADER | Rule-based word list (traditional baseline) |
| facebook/bart-large-mnli | Zero-shot classifier with no sentiment training |
| Qwen2.5-1.5B-Instruct | Small LLM, asked once per headline (not agentic) |

**Datasets:**

- **Financial PhraseBank.** FinBERT and DistilRoBERTa were trained on it, so their scores there are inflated. The results flag this.
- **Twitter Financial News** (`zeroshot/twitter-financial-news-sentiment`). None of the models was trained on it, so it gives the fair comparison.

**Outputs:**

- accuracy and macro-F1 for each model
- per-class scores
- confusion matrices
- Cohen's kappa (how often two models agree)

Every model also returns a probability for each label, so its scores can be used as numeric features later (Section 5).

A larger LLM can be added on a bigger GPU (e.g. Colab):

```bash
python -m src.model_comparison --extra-llm Qwen/Qwen2.5-7B-Instruct
```

---

## 3. Real-data pipeline

### 3.1 FNSPID loader (`src/fnspid_loader.py`)

- Downloads the 29 GB FNSPID dataset (Dong, Fan & Peng, 2024) from Hugging Face to local disk.
- Reads it in chunks so it fits in 14 GB of RAM.
- Pulls out the headlines for each ticker and removes duplicates (same URL, or same headline on the same day).
- Takes the news outlet from each article's web address.
- Saves coverage tables by ticker, year and outlet.

### 3.2 Choosing the stocks

- The tickers named in the interim report (AAPL, AMZN, JPM, XOM, JNJ) have patchy FNSPID coverage between 2018 and 2023.
- Every ticker was scanned, and one per sector was picked that has news in every year:
  - **INTC** (tech)
  - **SBUX** (consumer)
  - **WFC** (banking)
  - **CVX** (energy)
  - **MRK** (healthcare)
- The study covers **34,810 headlines** from 2018 to 2023, which is 1,509 trading days per stock.
- The scan results are in `results/data_coverage/`.

### 3.3 Daily sentiment features (`src/news_sentiment.py`)

- FinBERT scores every headline.
- Each headline is assigned to a trading day. Anything published at or after the 4 pm US close counts toward the next day, so the model never sees news before it is published.
- Features per stock per day:
  - average sentiment
  - how much sentiment varies within the day
  - share of positive and of negative headlines
  - number of headlines
  - a sentiment score that fades over several days (half-life of 3 days)
  - a version using only headlines that actually name the company (only about 40–50% of tagged headlines do)
- Scoring is saved in chunks and resumes after a crash, so slow models such as Qwen survive memory errors.

### 3.4 Price features and targets (`src/features.py`)

- **Price features:**
  - past returns (1 to 5 days, and the 5-day return)
  - 20-day volatility
  - 20- and 50-day moving-average ratios
  - RSI-14
  - volume ratio and log volume change
  - daily high–low range
- **Targets at 1, 3 and 5 trading days:**
  - **down / stable / up**. The "stable" band is the 33rd percentile of the absolute return, set from training data only.
  - the **% return**, so RMSE can be reported

---

## 4. The ablation: does sentiment help?

**Code:** `src/ablation.py`, `src/lstm_model.py` → `results/ablation/`

### Design

- **Feature sets:**
  - price only
  - price + sentiment
  - price + sentiment + company-specific sentiment
- **Models:**
  - XGBoost.
  - An LSTM: 20-day input window, a joint classification and regression loss, 3 random seeds averaged, and early stopping.
  - A class-weighted LSTM, added because the plain LSTM rarely predicted "down".
- **Walk-forward testing:**
  - Each model is tested on 2021, 2022 and 2023, trained only on the years before.
  - A gap of *h* days between training and test data stops the target leaking into training.
- **Statistics:**
  - A paired moving-block bootstrap (block length 10, 2,000 resamples) gives confidence intervals.
  - Benjamini-Hochberg correction allows for running many tests at once.

### Results

| Model | Sentiment gain (macro-F1) at 1d / 3d / 5d | Significant? |
|---|---|---|
| XGBoost | +0.005 / +0.013 / +0.014 | No (q ≥ 0.32) |
| LSTM | +0.025 to +0.033 | Yes, but an artifact |
| Class-weighted LSTM | 0 / +0.018 / +0.017 | No |

- **The plain LSTM's "significant" gain is an artifact.**
  - Without sentiment, it barely predicts "down".
  - When sentiment makes it predict "down", those calls are right only as often as chance (precision equals the base rate).
  - The class-weighted LSTM removes the artifact, and its result matches XGBoost.
- **Conclusion:** all three models agree. There is no gain at 1 day, and a small, non-significant gain at 3–5 days.
- **Every model is close to chance**, and the RMSE is worse than simply predicting zero return. This fits the weak-form efficient market hypothesis. It is a reportable finding, not a failure.

---

## 5. Which sentiment model helps prediction most?

**Code:** `src/sentiment_downstream.py` → `results/sentiment_downstream/`

Each of the 7 models scored all 34,810 headlines and went through the same XGBoost ablation against the same price-only baseline. Figures are the change in macro-F1, all five stocks pooled.

| Model | 1d | 3d | 5d | Stock–horizon combos improved | Text F1 |
|---|---|---|---|---|---|
| **FinBERT** | +0.005 | **+0.013** | **+0.015** | 10/15 | 0.668 |
| BART zero-shot | +0.004 | +0.006 | +0.009 | 11/15 | 0.458 |
| Qwen 1.5B | +0.002 | +0.012 | +0.002 | 9/15 | 0.569 |
| RoBERTa-Twitter | +0.010 | +0.010 | +0.000 | 11/15 | 0.613 |
| DistilRoBERTa-fin | +0.008 | +0.004 | −0.002 | 8/15 | **0.706** |
| FinBERT-tone | +0.002 | +0.003 | −0.003 | 8/15 | 0.661 |
| VADER | −0.002 | +0.004 | −0.002 | 7/15 | 0.447 |

**Findings:**

- **FinBERT gives the largest gain at 3 and 5 days**, which supports choosing it. None of the models is statistically significant (smallest corrected q = 0.65), so this is a ranking, not proof.
- **The best text classifier is not the best predictor.** DistilRoBERTa wins the labelled-text test but adds almost nothing to price prediction. Sentiment models should be judged on the task they are used for, not only on text benchmarks.
- **The models disagree a lot on real headlines.**
  - Agreement with FinBERT is κ = 0.60 for DistilRoBERTa and 0.25 for Qwen.
  - Qwen and BART label over 60% of headlines positive, against FinBERT's 21%.
- **VADER barely reacts to financial language.** When converted to labels, 98.5% of its outputs are neutral.
- The FinBERT row reproduces the Section 4 XGBoost result exactly, which confirms the pipeline is consistent.

---

## 6. Comparing news sources

**Code:** `src/source_comparison.py` → `results/source_comparison/`

FNSPID only has outlets other than Nasdaq.com up to June 2020, so this comparison uses 2018 to mid-2020.

### Market agreement

This measures how often FinBERT's sentiment matches the stock's direction over the 2 days around publication (close of day T−1 to close of day T+1).

- **On INTC (6 outlets):**

  | Outlet | Agreement |
  |---|---|
  | GuruFocus | 72% |
  | Benzinga | 62% |
  | Seeking Alpha | 61% |
  | IBD | 60% |
  | Nasdaq | 57% |
  | Zacks | 53% |

- **Across INTC, MRK, SBUX and WFC:** Benzinga 62% (95% CI 60–65%) against Nasdaq 55% (95% CI 53–57%).
- **Caveat:** the window includes the publication day, and some headlines describe a price move that has already happened. This inflates agreement.

### Cross-outlet correlation

On the same day, different outlets' sentiment for the same stock correlates at only ρ ≈ 0.13. The choice of source changes the sentiment signal a lot.

### Ablation by source

The XGBoost ablation was run separately with Nasdaq and with Benzinga sentiment, tested on 2020. No gain is significant, because the test window is only the first half of 2020.

---

## 7. Cleanup and engineering

- **FYP I prototype moved to `legacy/`:** the synthetic pipeline, notebook and old `.pkl` models, with a README explaining why.
- **`src/main.py` runs the whole pipeline:**
  - `python -m src.main` runs everything.
  - `--steps …` runs selected steps.
- **`README.md` rewritten:** project structure, installation, study design, usage, and where each result lives.
- **`requirements.txt`:** added xgboost, vaderSentiment, pyarrow and accelerate.
- **Python environment:** rebuilt on Python 3.12 with CUDA PyTorch (torch 2.x, cu126).
- **Hardware limits:** the 4 GB GPU and 14 GB RAM set some constraints.
  - Qwen can hit GPU memory errors. Chunked, resumable scoring handles them.
  - Long jobs may need to run chunk by chunk.

---

## 8. Mapping to the dissertation

| Dissertation section | Evidence |
|---|---|
| FinBERT validation | `results/model_comparison/` |
| Supervisor's "compare against other AI (non-agentic)" | Sections 2 and 5 (text benchmark and prediction benchmark) |
| Ablation by model and horizon | `results/ablation/` |
| Comparison by news source | `results/source_comparison/` |
| Discussion | Small, non-significant gains at 3–5 days; the LSTM artifact; text accuracy ≠ predictive value; source disagreement; consistency with market efficiency |

---

## 9. Changes from the interim report and open items

- **Ticker change:** INTC, SBUX, WFC, CVX and MRK replace AAPL, AMZN, JPM, XOM and JNJ. This was a data-coverage decision, and `results/data_coverage/` documents it.
- **Recency weighting:** FNSPID gives only the date of each article, not the time. So within-day recency weighting became a sentiment score that fades across days.
- **Not built yet:**
  - NewsAPI/RSS live news (a supplementary source in the interim report)
  - a larger LLM, e.g. Qwen 7B on Colab
- **Remaining:** the dissertation write-up.
