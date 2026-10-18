"""
Data pipeline for collecting and processing financial news and stock data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf

from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_TICKER,
    RANDOM_SEED,
)
from src.utils import normalize_yfinance_dataframe, ensure_date_column


class FinancialDataPipeline:
    """Main pipeline for collecting and processing financial data"""

    SENTIMENT_FILL_COLS = ["daily_sentiment_score", "news_count", "headlines"]

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        days_back: int = 365,
        end_date: Optional[datetime] = None,
    ):
        """
        Initialize the data pipeline

        Args:
            tickers: List of stock tickers to process (default: [DEFAULT_TICKER])
            days_back: Number of days of historical data to fetch
            end_date: End of data window (default: today)
        """
        self.tickers = tickers or [DEFAULT_TICKER]
        self.days_back = days_back
        self.end_date = end_date or datetime.now()
        self.start_date = self.end_date - timedelta(days=days_back)

        self.stock_data = {}
        self.news_data = {}

    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical stock data for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with OHLCV data and calculated features
        """
        try:
            print(f"Fetching stock data for {ticker}...")
            data = yf.download(
                ticker,
                start=self.start_date.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                progress=False,
            )

            if data.empty:
                print(f"No data found for {ticker}")
                return pd.DataFrame()

            data, close_col, volume_col = normalize_yfinance_dataframe(data)
            data = ensure_date_column(data)

            if close_col is None or volume_col is None:
                print(f"  Available columns: {list(data.columns)}")
                raise ValueError("Could not find close or volume columns")

            print(f"  Using columns: close='{close_col}', volume='{volume_col}'")

            data["Returns"] = data[close_col].pct_change()
            data["Log_Returns"] = np.log(data[close_col] / data[close_col].shift(1))
            data["Volatility"] = data["Returns"].rolling(window=20).std()
            data["MA_20"] = data[close_col].rolling(window=20).mean()
            data["MA_50"] = data[close_col].rolling(window=50).mean()
            data["Price_Up"] = (data[close_col].shift(-1) > data[close_col]).astype(int)
            data["Volume_MA"] = data[volume_col].rolling(window=20).mean()
            data["Volume_Ratio"] = data[volume_col] / (data["Volume_MA"] + 1e-6)
            data["Ticker"] = ticker

            self.stock_data[ticker] = data
            print(f"[OK] Fetched {len(data)} records for {ticker}")
            return data

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()

    def create_synthetic_news_data(self, ticker: str = None, num_samples: int = 100) -> pd.DataFrame:
        """
        Create synthetic financial news data for demonstration

        Args:
            ticker: Ticker to associate with news
            num_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic news articles
        """
        ticker = ticker or self.tickers[0]
        np.random.seed(RANDOM_SEED)

        positive_headlines = [
            "Strong earnings beat expectations",
            "Company announces new innovative product",
            "Analyst upgrades stock to buy",
            "Revenue growth exceeds forecasts",
            "Strategic partnership announced",
            "Market share gains in key segment",
            "CEO confident about future outlook",
            "Dividends increased by 25%",
            "Successful product launch drives growth",
            "Expansion into new markets announced",
        ]

        negative_headlines = [
            "Missed earnings estimates",
            "Product recall announced",
            "Analyst downgrades stock",
            "Revenue decline disappoints",
            "Executive departure announced",
            "Market share losses reported",
            "Guidance lowered for next quarter",
            "Failed product launch",
            "Regulatory investigation launched",
            "Supply chain disruptions",
        ]

        neutral_headlines = [
            "Company announces routine earnings",
            "Stock trading at historical average",
            "Market conditions remain stable",
            "Analyst maintains hold rating",
            "Routine dividend payment",
            "Analyst conference scheduled",
            "Insider stock transaction reported",
            "Board meeting scheduled",
            "Share buyback program active",
            "Standard business update",
        ]

        headline_sentiments = (
            [(h, 2) for h in positive_headlines]
            + [(h, 0) for h in negative_headlines]
            + [(h, 1) for h in neutral_headlines]
        )

        data = []
        base_date = self.start_date

        for i in range(num_samples):
            headline, sentiment = headline_sentiments[
                np.random.randint(len(headline_sentiments))
            ]
            date = base_date + timedelta(days=i % self.days_back)
            data.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "headline": headline,
                    "text": headline,
                    "sentiment": sentiment,
                    "source": "synthetic",
                }
            )

        df = pd.DataFrame(data)
        self.news_data[ticker] = df
        print(f"[OK] Created {len(df)} synthetic news records for {ticker}")
        return df

    def merge_news_and_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Merge news data with stock data by date

        Args:
            ticker: Stock ticker symbol

        Returns:
            Merged DataFrame
        """
        if ticker not in self.stock_data or ticker not in self.news_data:
            print(f"Data not available for {ticker}")
            return pd.DataFrame()

        stock_df = ensure_date_column(self.stock_data[ticker])

        news_df = ensure_date_column(self.news_data[ticker])

        daily_sentiment = news_df.groupby("date", as_index=False).agg(
            daily_sentiment_score=("sentiment", "mean"),
            news_count=("sentiment", "count"),
            headlines=("headline", lambda s: " | ".join(s.astype(str))),
        )

        merged = pd.merge(stock_df, daily_sentiment, on="date", how="left")

        fill_cols = [c for c in self.SENTIMENT_FILL_COLS if c in merged.columns]
        merged[fill_cols] = merged[fill_cols].ffill()
        merged = merged.dropna(subset=["daily_sentiment_score"])

        print(f"[OK] Merged data shape: {merged.shape}")
        return merged

    def preprocess_data(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess merged data

        Args:
            merged_df: Merged news and stock data

        Returns:
            Preprocessed DataFrame
        """
        df = merged_df.copy()

        score_min = df["daily_sentiment_score"].min()
        score_max = df["daily_sentiment_score"].max()
        score_range = score_max - score_min
        if score_range == 0:
            df["daily_sentiment_score"] = 0.0
        else:
            df["daily_sentiment_score"] = (
                df["daily_sentiment_score"] - score_min
            ) / score_range

        numeric_means = df.mean(numeric_only=True)
        df = df.fillna(numeric_means)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("")

        df = df.drop_duplicates(subset=["date", "Ticker"])
        print(f"[OK] Preprocessed data shape: {df.shape}")
        return df

    def save_data(self, data: pd.DataFrame, filename: str, data_type: str = "processed") -> None:
        """Save data to CSV."""
        filepath = RAW_DATA_DIR / filename if data_type == "raw" else PROCESSED_DATA_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"[OK] Data saved to {filepath}")

    def run_pipeline(self, save_intermediate: bool = True) -> Dict[str, pd.DataFrame]:
        """Run complete data pipeline."""
        results = {}

        for ticker in self.tickers:
            print(f"\n{'='*60}")
            print(f"Processing {ticker}...")
            print(f"{'='*60}")

            stock_data = self.fetch_stock_data(ticker)
            if stock_data.empty:
                continue

            if save_intermediate:
                self.save_data(stock_data, f"{ticker}_stock_data.csv", "raw")

            news_data = self.create_synthetic_news_data(ticker, num_samples=100)
            if save_intermediate:
                self.save_data(news_data, f"{ticker}_news_data.csv", "raw")

            merged_data = self.merge_news_and_stock_data(ticker)
            if merged_data.empty:
                continue

            processed_data = self.preprocess_data(merged_data)
            self.save_data(processed_data, f"{ticker}_processed_data.csv", "processed")
            results[ticker] = processed_data

        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    pipeline = FinancialDataPipeline(tickers=["AAPL", "MSFT", "GOOGL"], days_back=365)
    results = pipeline.run_pipeline(save_intermediate=True)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    for ticker, data in results.items():
        print(f"\n{ticker}: {len(data)} records")
        print(data.head())
