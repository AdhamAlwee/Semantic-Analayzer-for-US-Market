"""
Sentiment analysis using FinBERT model
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Optional

from config.config import DEVICE, FINBERT_MODEL_NAME


class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial text"""

    def __init__(
        self,
        model_name: str = FINBERT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        """
        Initialize FinBERT sentiment analyzer

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cpu' or 'cuda'); defaults to config.DEVICE
        """
        self.device = device if device is not None else DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Use official FinBERT label order from model config
        self.label_mapping = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

    def analyze_sentiment(self, text: str, return_scores: bool = False) -> Dict:
        """
        Analyze sentiment of a single text

        Args:
            text: Text to analyze
            return_scores: Whether to return confidence scores

        Returns:
            Dictionary with sentiment and scores
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()

        result = {
            "text": text[:100],
            "sentiment": self.label_mapping[predicted_class],
            "predicted_class": predicted_class,
        }

        if return_scores:
            result["scores"] = {
                self.label_mapping[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }

        return result

    def predict_proba(self, texts: List[str], batch_size: int = 64) -> Dict[str, List[float]]:
        """
        Class probabilities for many texts.

        Returns:
            {"positive": [...], "negative": [...], "neutral": [...]} aligned with texts
        """
        probs = {label: [] for label in self.label_mapping.values()}
        for i in range(0, len(texts), batch_size):
            batch_inputs = self.tokenizer(
                texts[i : i + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            batch_inputs = {key: val.to(self.device) for key, val in batch_inputs.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**batch_inputs).logits, dim=1).cpu()
            for idx, label in self.label_mapping.items():
                probs[label].extend(p[:, idx].tolist())
        return probs

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze sentiment for multiple texts

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            List of sentiment analysis results
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            batch_inputs = {key: val.to(self.device) for key, val in batch_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                logits = outputs.logits
                predicted_classes = torch.argmax(logits, dim=1)

            for j, pred_class in enumerate(predicted_classes):
                results.append(
                    {
                        "text": batch[j][:100],
                        "sentiment": self.label_mapping[pred_class.item()],
                        "predicted_class": pred_class.item(),
                    }
                )

        return results
