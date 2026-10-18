"""
Benchmark FinBERT against other (non-agentic) sentiment models on labelled
financial-sentiment datasets.

Every model is a single forward pass per text: no tools, no multi-step
reasoning, no external calls. All checkpoints are open and versioned, so the
benchmark is reproducible (see interim report, Section 2.4).

Usage:
    python -m src.model_comparison                      # all models, all datasets
    python -m src.model_comparison --limit 200          # quick smoke test
    python -m src.model_comparison --models finbert vader --datasets twitter
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DEVICE, RAW_DATA_DIR, RESULTS_DIR, RANDOM_SEED

LABELS = ["negative", "neutral", "positive"]
BENCHMARK_DIR = RAW_DATA_DIR / "benchmarks"
OUTPUT_DIR = RESULTS_DIR / "model_comparison"

_LABEL_SYNONYMS = {
    "negative": "negative",
    "neg": "negative",
    "bearish": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "positive": "positive",
    "pos": "positive",
    "bullish": "positive",
}


def normalize_label(raw: str) -> str:
    """Map a model- or dataset-specific label name onto negative/neutral/positive."""
    key = str(raw).strip().lower()
    if key not in _LABEL_SYNONYMS:
        raise ValueError(f"Unrecognised sentiment label: {raw!r}")
    return _LABEL_SYNONYMS[key]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def load_phrasebank(subset: str = "Sentences_AllAgree") -> pd.DataFrame:
    """
    Financial PhraseBank (Malo et al., 2014), read from the original v1.0 zip.

    NOTE: ProsusAI/finbert and the DistilRoBERTa financial model were fine-tuned
    on this dataset, so their scores here are optimistic (train/test overlap).
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
        cache_dir=BENCHMARK_DIR,
    )
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(f"FinancialPhraseBank-v1.0/{subset}.txt").decode("latin-1")

    rows = []
    for line in raw.splitlines():
        if "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        rows.append({"text": text.strip(), "label": normalize_label(label)})
    return pd.DataFrame(rows)


def load_twitter_financial_news(split: str = "valid") -> pd.DataFrame:
    """
    Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment).

    Finance-related tweets/headlines labelled Bearish/Bullish/Neutral. None of the
    benchmarked models were trained on it, so it is the uncontaminated test set.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="zeroshot/twitter-financial-news-sentiment",
        filename=f"sent_{split}.csv",
        repo_type="dataset",
        cache_dir=BENCHMARK_DIR,
    )
    df = pd.read_csv(path)
    id2label = {0: "negative", 1: "positive", 2: "neutral"}
    return pd.DataFrame({"text": df["text"], "label": df["label"].map(id2label)})


DATASETS: Dict[str, Callable[[], pd.DataFrame]] = {
    "phrasebank_allagree": load_phrasebank,
    "twitter_fin_news": load_twitter_financial_news,
}

# Which datasets each model saw during fine-tuning (contamination flags)
TRAINED_ON = {
    "finbert": {"phrasebank_allagree"},
    "distilroberta_fin": {"phrasebank_allagree"},
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SentimentClassifier:
    """
    Common interface:
      predict_proba(texts) -> (n, 3) array of probabilities in LABELS order
      predict(texts)       -> list of negative/neutral/positive
    """

    name: str = ""
    family: str = ""
    checkpoint: str = ""

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, texts: List[str]) -> List[str]:
        return [LABELS[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def num_parameters(self) -> Optional[int]:
        return None

    def close(self) -> None:
        """Release GPU memory before the next model is loaded."""
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


class VaderClassifier(SentimentClassifier):
    """Rule/lexicon baseline (Hutto & Gilbert, 2014). Not finance-specific."""

    name = "vader"
    family = "Lexicon (general)"
    checkpoint = "vaderSentiment"

    def __init__(self, threshold: float = 0.05):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        self.threshold = threshold

    def predict(self, texts: List[str]) -> List[str]:
        preds = []
        for text in texts:
            compound = self.analyzer.polarity_scores(text)["compound"]
            if compound >= self.threshold:
                preds.append("positive")
            elif compound <= -self.threshold:
                preds.append("negative")
            else:
                preds.append("neutral")
        return preds

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """VADER's neg/neu/pos proportions (they sum to 1) as pseudo-probabilities."""
        scores = [self.analyzer.polarity_scores(t) for t in texts]
        return np.array([[s["neg"], s["neu"], s["pos"]] for s in scores])


class HFSequenceClassifier(SentimentClassifier):
    """Any Hugging Face sequence-classification checkpoint with sentiment labels."""

    def __init__(
        self,
        name: str,
        family: str,
        checkpoint: str,
        device: str = DEVICE,
        batch_size: int = 32,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.family = family
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()
        self.id2label = {
            int(k): normalize_label(v) for k, v in self.model.config.id2label.items()
        }

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        import torch

        column = [LABELS.index(self.id2label[i]) for i in range(len(self.id2label))]
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[i : i + self.batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**batch).logits, dim=1).cpu().numpy()
            ordered = np.zeros_like(probs)
            ordered[:, column] = probs
            out.append(ordered)
        return np.vstack(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def close(self) -> None:
        del self.model
        super().close()


class ZeroShotNLIClassifier(SentimentClassifier):
    """
    Zero-shot classification via natural-language inference (Yin et al., 2019).
    The model was never trained on sentiment labels; it scores hypotheses.
    """

    family = "Zero-shot NLI (general)"

    def __init__(
        self,
        name: str = "bart_mnli_zeroshot",
        checkpoint: str = "facebook/bart-large-mnli",
        device: str = DEVICE,
        batch_size: int = 16,
    ):
        from transformers import pipeline

        self.name = name
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.pipe = pipeline(
            "zero-shot-classification",
            model=checkpoint,
            device=0 if device == "cuda" else -1,
        )
        self.template = "The sentiment of this financial news is {}."

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        outputs = self.pipe(
            texts,
            candidate_labels=LABELS,
            hypothesis_template=self.template,
            batch_size=self.batch_size,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return np.array(
            [[dict(zip(o["labels"], o["scores"]))[label] for label in LABELS] for o in outputs]
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.pipe.model.parameters())

    def close(self) -> None:
        del self.pipe
        super().close()


class PromptedLLMClassifier(SentimentClassifier):
    """
    General-purpose instruction-tuned LLM used zero-shot with a fixed prompt.

    Non-agentic: one prompt, one forward pass, no tools. Instead of free-text
    generation, the next-token logits of the three label words are compared,
    so every output is a valid label and the result is deterministic.
    """

    family = "General LLM, prompted zero-shot"

    SYSTEM_PROMPT = (
        "You are a financial sentiment classifier. Classify the sentiment of the "
        "given financial news text from an investor's point of view. Answer with "
        "exactly one word: positive, negative, or neutral."
    )

    def __init__(
        self,
        name: str = "qwen2.5_1.5b_instruct",
        checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = DEVICE,
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.model.to(device)
        self.model.eval()

        self.label_token_ids = []
        for label in LABELS:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            self.label_token_ids.append(ids[0])
        if len(set(self.label_token_ids)) != len(LABELS):
            raise ValueError("Label words share a first token; choose other label words")

    def _build_prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Text: {text}\nSentiment:"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Softmax over the three label words' next-token logits."""
        import torch

        out = []
        for i in range(0, len(texts), self.batch_size):
            prompts = [self._build_prompt(t) for t in texts[i : i + self.batch_size]]
            batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                next_token_logits = self.model(**batch).logits[:, -1, :]
            label_logits = next_token_logits[:, self.label_token_ids].float()
            out.append(torch.softmax(label_logits, dim=1).cpu().numpy())
        return np.vstack(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def close(self) -> None:
        del self.model
        super().close()


MODEL_FACTORIES: Dict[str, Callable[[], SentimentClassifier]] = {
    "finbert": lambda: HFSequenceClassifier(
        "finbert", "Domain-specific BERT (ours)", "ProsusAI/finbert"
    ),
    "finbert_tone": lambda: HFSequenceClassifier(
        "finbert_tone", "Domain-specific BERT", "yiyanghkust/finbert-tone"
    ),
    "distilroberta_fin": lambda: HFSequenceClassifier(
        "distilroberta_fin",
        "Compact financial transformer",
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    ),
    "roberta_twitter": lambda: HFSequenceClassifier(
        "roberta_twitter",
        "General-domain transformer",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ),
    "vader": VaderClassifier,
    "bart_mnli_zeroshot": ZeroShotNLIClassifier,
    "qwen2.5_1.5b_instruct": PromptedLLMClassifier,
}


def register_llm(checkpoint: str) -> str:
    """Add another prompted LLM (e.g. Qwen/Qwen2.5-7B-Instruct on Colab) by checkpoint."""
    name = checkpoint.split("/")[-1].lower()
    MODEL_FACTORIES[name] = lambda: PromptedLLMClassifier(name=name, checkpoint=checkpoint)
    return name


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    family: str
    checkpoint: str
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    ms_per_text: float
    params_millions: Optional[float]
    trained_on_this_dataset: bool


def evaluate(
    dataset: str, clf: SentimentClassifier, gold: List[str], preds: List[str], seconds: float
) -> BenchmarkResult:
    params = clf.num_parameters()
    return BenchmarkResult(
        dataset=dataset,
        model=clf.name,
        family=clf.family,
        checkpoint=clf.checkpoint,
        n=len(gold),
        accuracy=accuracy_score(gold, preds),
        macro_f1=f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0),
        weighted_f1=f1_score(gold, preds, labels=LABELS, average="weighted", zero_division=0),
        ms_per_text=1000 * seconds / max(len(gold), 1),
        params_millions=round(params / 1e6, 1) if params else None,
        trained_on_this_dataset=dataset in TRAINED_ON.get(clf.name, set()),
    )


def agreement_matrix(predictions: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
    """Pairwise Cohen's kappa between models' predictions."""
    kappa = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    for a in model_names:
        for b in model_names:
            kappa.loc[a, b] = cohen_kappa_score(predictions[a], predictions[b])
    return kappa.round(3)


def save_confusion_plots(predictions: pd.DataFrame, model_names: List[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(4, len(model_names))
    rows = int(np.ceil(len(model_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.flat, model_names):
        cm = confusion_matrix(predictions["label"], predictions[name], labels=LABELS)
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, int(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks(range(3), [l[:3] for l in LABELS])
        ax.set_yticks(range(3), [l[:3] for l in LABELS])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    for ax in list(axes.flat)[len(model_names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_benchmark(
    dataset_names: List[str],
    model_names: List[str],
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run every model on every dataset; write CSVs, plots and a summary to results/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for ds in dataset_names:
        df = DATASETS[ds]()
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=RANDOM_SEED)
        data[ds] = df.reset_index(drop=True)
        print(f"[OK] {ds}: {len(df)} texts, label counts {df['label'].value_counts().to_dict()}")

    predictions = {ds: df.copy() for ds, df in data.items()}
    results, per_class_rows = [], []

    # Load each model once, run it on all datasets, then free it (4 GB GPU)
    for name in model_names:
        print(f"\n{'='*60}\nModel: {name}\n{'='*60}")
        clf = MODEL_FACTORIES[name]()
        for ds, df in data.items():
            texts = df["text"].astype(str).tolist()
            start = time.perf_counter()
            preds = clf.predict(texts)
            seconds = time.perf_counter() - start

            predictions[ds][name] = preds
            result = evaluate(ds, clf, df["label"].tolist(), preds, seconds)
            results.append(result)

            report = classification_report(
                df["label"], preds, labels=LABELS, output_dict=True, zero_division=0
            )
            for label in LABELS:
                per_class_rows.append({"dataset": ds, "model": name, "class": label,
                                       **{k: report[label][k] for k in
                                          ("precision", "recall", "f1-score", "support")}})
            print(f"  {ds}: accuracy={result.accuracy:.4f} macro_f1={result.macro_f1:.4f} "
                  f"({result.ms_per_text:.1f} ms/text)")
        clf.close()

    summary = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        ["dataset", "macro_f1"], ascending=[True, False]
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUTPUT_DIR / "per_class.csv", index=False)

    for ds, pred_df in predictions.items():
        pred_df.to_csv(OUTPUT_DIR / f"predictions_{ds}.csv", index=False)
        agreement_matrix(pred_df, model_names).to_csv(OUTPUT_DIR / f"agreement_kappa_{ds}.csv")
        save_confusion_plots(pred_df, model_names, OUTPUT_DIR / f"confusion_{ds}.png")

    with open(OUTPUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"datasets": dataset_names, "models": model_names, "limit": limit,
                   "device": DEVICE,
                   "checkpoints": {r.model: r.checkpoint for r in results}}, f, indent=2)

    print(f"\n[OK] Results written to {OUTPUT_DIR}")
    print(summary[["dataset", "model", "accuracy", "macro_f1", "ms_per_text",
                   "trained_on_this_dataset"]].to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES),
                        choices=list(MODEL_FACTORIES))
    parser.add_argument("--limit", type=int, default=None,
                        help="Random sample size per dataset (for quick runs)")
    parser.add_argument("--extra-llm", nargs="*", default=[], metavar="CHECKPOINT",
                        help="Extra Hugging Face instruct LLMs to prompt, e.g. "
                             "Qwen/Qwen2.5-7B-Instruct (needs a larger GPU, e.g. Colab)")
    args = parser.parse_args()
    models = args.models + [register_llm(c) for c in args.extra_llm]
    run_benchmark(args.datasets, models, args.limit)


if __name__ == "__main__":
    main()
