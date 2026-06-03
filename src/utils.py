"""
Utility functions for the project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from datetime import datetime


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a normalized 'date' column exists.

    yfinance + MultiIndex columns often yield names like ``index_`` or ``date_``
    after flattening, not plain ``date``.
    """
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df

    explicit_candidates = (
        "datetime",
        "timestamp",
        "index",
        "date_",
        "index_",
        "datetime_",
    )
    for candidate in explicit_candidates:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            return df

    for col in list(df.columns):
        base = str(col).lower().rstrip("_")
        if base in ("date", "datetime", "timestamp", "index"):
            df = df.rename(columns={col: "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            return df

    if not isinstance(df.index, pd.RangeIndex) and pd.api.types.is_datetime64_any_dtype(
        df.index
    ):
        df = df.reset_index()
        return ensure_date_column(df)

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.rename(columns={col: "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            return df

    raise KeyError(
        f"Could not find a date column. Available columns: {list(df.columns)}"
    )


def normalize_yfinance_dataframe(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """
    Normalize yfinance download output for consistent column access.

    Handles DatetimeIndex reset, MultiIndex columns, and lowercase names.

    Returns:
        (dataframe, close_column_name, volume_column_name)
    """
    if data.empty:
        return data, None, None

    if not isinstance(data.index, pd.RangeIndex):
        data = data.reset_index()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(str(c) for c in col).strip() for col in data.columns.values]

    data.columns = [
        col.lower() if isinstance(col, str) else str(col).lower() for col in data.columns
    ]

    data = ensure_date_column(data)

    close_col = None
    volume_col = None

    for col in data.columns:
        col_lower = str(col).lower()
        if "adj close" in col_lower or "adjusted close" in col_lower:
            close_col = col
        elif "close" in col_lower and close_col is None:
            close_col = col
        if "volume" in col_lower:
            volume_col = col

    return data, close_col, volume_col


def save_data(data: pd.DataFrame, filepath: str) -> None:
    """
    Save dataframe to CSV file

    Args:
        data: DataFrame to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file

    Args:
        filepath: Path to data file

    Returns:
        Loaded DataFrame
    """
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    raise FileNotFoundError(f"File not found: {filepath}")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets

    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def log_message(message: str, level: str = "INFO") -> None:
    """
    Log a message with timestamp

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
