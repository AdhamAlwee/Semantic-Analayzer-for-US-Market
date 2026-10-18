"""
Data loading utilities for financial news and stock market data
"""

import pandas as pd
from pathlib import Path
import yfinance as yf
from typing import Optional
from datetime import datetime, timedelta

from src.utils import normalize_yfinance_dataframe


class StockDataLoader:
    """Load and preprocess stock market data"""

    def __init__(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Initialize stock data loader

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date for data retrieval (YYYY-MM-DD)
            end_date: End date for data retrieval (YYYY-MM-DD)
        """
        self.ticker = ticker
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.start_date = start_date or (
            datetime.now() - timedelta(days=365)
        ).strftime("%Y-%m-%d")

    def fetch_stock_data(self) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance

        Returns:
            DataFrame with stock OHLCV data and return features
        """
        try:
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
            )
            if data.empty:
                return pd.DataFrame()

            data, close_col, _volume_col = normalize_yfinance_dataframe(data)
            if close_col is None:
                raise ValueError(
                    f"Could not find close column. Available: {list(data.columns)}"
                )

            data["returns"] = data[close_col].pct_change()
            data["price_movement"] = (
                data[close_col].shift(-1) > data[close_col]
            ).astype(int)
            return data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()

    def preprocess_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess stock data

        Args:
            data: Raw stock data

        Returns:
            Preprocessed stock data
        """
        return data.dropna()


class NewsDataLoader:
    """Load and preprocess financial news data"""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize news data loader

        Args:
            data_path: Path to news data file
        """
        self.data_path = data_path

    def load_news_data(self) -> pd.DataFrame:
        """
        Load news data from CSV file

        Returns:
            DataFrame with news articles
        """
        if self.data_path and Path(self.data_path).exists():
            return pd.read_csv(self.data_path)
        return pd.DataFrame()

    def preprocess_news_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess news data

        Args:
            data: Raw news data

        Returns:
            Preprocessed news data
        """
        data = data.drop_duplicates(subset=["text"], keep="first")
        return data.dropna(subset=["text"])
