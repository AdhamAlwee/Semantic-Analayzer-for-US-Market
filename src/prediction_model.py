"""
Stock price prediction model using sentiment features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime


class PricePredictionModel:
    """Stock price movement prediction model"""

    FEATURE_COLUMNS = [
        "Returns",
        "Log_Returns",
        "Volatility",
        "MA_20",
        "MA_50",
        "Volume_Ratio",
        "daily_sentiment_score",
        "news_count",
    ]

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize prediction model

        Args:
            model_type: "random_forest" or "gradient_boosting"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_results = None

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=42,
            )

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare features and target for modeling

        Args:
            data: Processed data with sentiment and stock features

        Returns:
            X (features), y (target), feature_names
        """
        available_features = [c for c in self.FEATURE_COLUMNS if c in data.columns]
        if "Price_Up" not in data.columns:
            raise ValueError("Target column 'Price_Up' not found in data")

        model_df = data[available_features + ["Price_Up"]].dropna()
        if model_df.empty:
            raise ValueError("No complete rows after dropping missing feature/target values")

        X = model_df[available_features].values
        y = model_df["Price_Up"].values.astype(int)
        self.feature_names = available_features

        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")

        return X, y, available_features

    def train(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Train the prediction model."""
        print(f"\nTraining {self.model_type} model...")

        X, y, _ = self.prepare_features(data)

        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        metrics = {
            "train": self._calculate_metrics(y_train, y_train_pred),
            "test": self._calculate_metrics(y_test, y_test_pred),
        }

        feature_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.training_results = {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_type": self.model_type,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_shape": X.shape,
            "test_size": test_size,
        }

        self._print_results(metrics, feature_importance)
        return self.training_results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        }

    def _print_results(self, metrics: Dict, feature_importance: pd.DataFrame) -> None:
        """Print training results."""
        print("\n" + "=" * 60)
        print("Training Results")
        print("=" * 60)

        print("\nTrain Metrics:")
        for metric, value in metrics["train"].items():
            print(f"  {metric}: {value:.4f}")

        print("\nTest Metrics:")
        for metric, value in metrics["test"].items():
            print(f"  {metric}: {value:.4f}")

        print("\nTop 5 Important Features:")
        print(feature_importance.head())

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)

    def save_model(self, filepath: str) -> None:
        """Save trained model and scaler."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, filepath.replace(".pkl", "_scaler.pkl"))
        print(f"[OK] Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model and scaler."""
        self.model = joblib.load(filepath)
        self.scaler = joblib.load(filepath.replace(".pkl", "_scaler.pkl"))
        print(f"[OK] Model loaded from {filepath}")


class EnsemblePredictor:
    """Ensemble of multiple models for improved predictions"""

    def __init__(self):
        self.models = {
            "random_forest": PricePredictionModel("random_forest"),
            "gradient_boosting": PricePredictionModel("gradient_boosting"),
        }
        self.training_results = {}

    def train_all(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Dict]:
        """Train all models in the ensemble."""
        results = {}
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            results[name] = model.train(data, test_size)
        self.training_results = results
        return results

    def predict_ensemble(self, X: np.ndarray, method: str = "average") -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions."""
        predictions = []
        probabilities = []

        for model in self.models.values():
            pred, probs = model.predict(X)
            predictions.append(pred)
            probabilities.append(probs)

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        if method == "voting":
            ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        else:
            ensemble_pred = np.round(np.mean(probabilities, axis=0)[:, 1]).astype(int)

        return ensemble_pred, np.mean(probabilities, axis=0)

    def save_all_models(self, dirpath: str) -> None:
        """Save all models in ensemble."""
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save_model(str(Path(dirpath) / f"{name}_model.pkl"))


if __name__ == "__main__":
    from src.data_pipeline import FinancialDataPipeline

    pipeline = FinancialDataPipeline(tickers=["AAPL"], days_back=365)
    results = pipeline.run_pipeline()

    if results:
        data = results["AAPL"]
        single_model = PricePredictionModel(model_type="random_forest")
        single_model.train(data)
        ensemble = EnsemblePredictor()
        ensemble.train_all(data)
