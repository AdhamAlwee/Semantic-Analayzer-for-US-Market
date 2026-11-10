"""
Interactive demo of the FYP II project.

  1. Score a headline   type any headline and see FinBERT's probabilities
  2. Explore a stock    price and daily FinBERT sentiment over time, with the headlines
  3. Key results        the main figures and tables from results/

Run from the project root:
    streamlit run app/demo.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import RESULTS_DIR, STUDY_TICKERS
from src.features import build_all

SCORED_NEWS = project_root / "data" / "processed" / "fnspid_news_scored.parquet"
COMPANY = {"INTC": "Intel", "SBUX": "Starbucks", "WFC": "Wells Fargo", "CVX": "Chevron", "MRK": "Merck"}

st.set_page_config(page_title="FinBERT Sentiment and US Stocks", layout="wide")


@st.cache_resource(show_spinner="Loading FinBERT...")
def load_finbert():
    from transformers import pipeline

    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


@st.cache_data(show_spinner="Loading prices and sentiment...")
def load_panel() -> pd.DataFrame:
    return build_all(STUDY_TICKERS)


@st.cache_data(show_spinner="Loading headlines...")
def load_news() -> pd.DataFrame:
    news = pd.read_parquet(SCORED_NEWS, columns=["published_utc", "ticker", "source", "headline",
                                                 "finbert_label", "sentiment_score"])
    news["date"] = news["published_utc"].dt.tz_convert(None).dt.normalize()
    return news.drop(columns="published_utc")


st.title("Financial News Sentiment with FinBERT: US Stock Movement")
st.caption("FYP II · Mas Hairul Adham Bin Alwee · Universiti Teknologi PETRONAS")

tab_score, tab_stock, tab_results = st.tabs(["Score a headline", "Explore a stock", "Key results"])

with tab_score:
    st.subheader("Score a headline with FinBERT")
    examples = [
        "Starbucks Corp shares are up 10.5% after the company beat Q4 earnings estimates.",
        "Intel to Cut Workforce as Part of Massive Cost Reduction Efforts",
        "Wells Fargo posts higher quarterly profit",
        "Merck COVID-19 pill success slams Moderna shares, shakes up healthcare sector",
    ]
    choice = st.selectbox("Try an example from the data, or type your own below", examples)
    text = st.text_input("Headline", value=choice)
    if text.strip():
        probs = {d["label"]: d["score"] for d in load_finbert()(text)[0]}
        score = probs.get("positive", 0) - probs.get("negative", 0)
        label = max(probs, key=probs.get)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Label", label)
        c2.metric("Score  P(pos) − P(neg)", f"{score:+.2f}")
        c3.metric("P(positive)", f"{probs.get('positive', 0):.2f}")
        c4.metric("P(negative)", f"{probs.get('negative', 0):.2f}")
        st.bar_chart(pd.Series(probs, name="probability"))
        st.info("FinBERT scores the tone of the words, not which company the news is about, "
                "and not how the market will read it. The examples above were all scored confidently, "
                "yet some of the stocks moved the other way on the day.")

with tab_stock:
    st.subheader("Price and daily FinBERT sentiment")
    panel, news = load_panel(), load_news()
    c1, c2 = st.columns([1, 2])
    ticker = c1.selectbox("Stock", STUDY_TICKERS, format_func=lambda t: f"{t} · {COMPANY[t]}")
    window = c1.slider("Sentiment smoothing (trading days)", 1, 60, 20)
    d = panel[panel["ticker"] == ticker].set_index("date").sort_index()
    years = c2.slider("Years", 2018, 2023, (2018, 2023))
    d = d[(d.index.year >= years[0]) & (d.index.year <= years[1])]
    chart = pd.DataFrame({
        "Close price (rebased to 100)": 100 * d["close"] / d["close"].iloc[0],
    })
    st.line_chart(chart)
    sent = d["sent_mean"].where(d["news_count"] > 0).rolling(window, min_periods=1).mean()
    st.line_chart(pd.DataFrame({f"Daily mean sentiment ({window}-day average)": sent}))

    st.markdown("**Headlines**")
    q = st.text_input("Filter headlines (word search)", "")
    n = news[(news["ticker"] == ticker) & (news["date"].dt.year.between(*years))]
    if q:
        n = n[n["headline"].str.contains(q, case=False, regex=False)]
    st.caption(f"{len(n):,} headlines")
    st.dataframe(n.sort_values("date", ascending=False)[["date", "source", "finbert_label", "sentiment_score", "headline"]]
                 .head(500), hide_index=True, width="stretch")

with tab_results:
    st.subheader("Key results")
    st.markdown("**1. Adding sentiment did not reliably improve XGBoost** (change in macro-F1, 95% CI, 2021–2023)")
    st.image(str(RESULTS_DIR / "ablation" / "xgboost" / "ablation_delta_f1.png"))
    st.markdown("**2. Sentiment correlates with the same day's return, not later ones** (Spearman ρ, 2018–2023)")
    corr = pd.read_csv(RESULTS_DIR / "same_day_association" / "sentiment_return_correlation.csv")
    st.dataframe(corr.pivot(index="ticker", columns="window", values="rho")
                 [["previous_day", "same_day", "next_day", "day_after"]].round(3), width="stretch")
    st.markdown("**3. Prices move on the news day, then go flat** (event study)")
    st.image(str(RESULTS_DIR / "event_study" / "event_study.png"))
    st.markdown("**4. A trading rule on the signal loses money after costs** (2021–2023)")
    st.dataframe(pd.read_csv(RESULTS_DIR / "event_study" / "trading_test.csv")
                 [["strategy", "ann_return_pct", "ann_ci_low_pct", "ann_ci_high_pct", "sharpe"]].round(2),
                 hide_index=True, width="stretch")
    st.image(str(RESULTS_DIR / "event_study" / "trading_test.png"))
