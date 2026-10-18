"""
LSTM movement/return model for the ablation study (sequential deep-learning family).

One network with two heads, trained jointly:
  - classification head: down / stable / up (cross-entropy)
  - regression head: forward return, standardised (MSE)

Input: the last SEQ_LEN trading days of the chosen features, standardised with
training-window statistics only. Early stopping uses the most recent 15% of the
training window (chronological, purged), never the test year. Predictions are
averaged over several seeds to reduce run-to-run noise; the same seeds are used
for every feature set so the ablation stays matched.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DEVICE, RANDOM_SEED

SEQ_LEN = 20
HIDDEN_SIZE = 32
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8
VALIDATION_FRACTION = 0.15
REGRESSION_LOSS_WEIGHT = 0.5
SEEDS = [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2]


class LSTMMovementModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.cls_head = nn.Linear(HIDDEN_SIZE, 3)
        self.reg_head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.cls_head(h), self.reg_head(h).squeeze(-1)


def make_sequences(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Window ending at each row: X[row-SEQ_LEN+1 : row+1] -> (n, SEQ_LEN, n_features)."""
    offsets = np.arange(-SEQ_LEN + 1, 1)
    return X[rows[:, None] + offsets]


def _train_one(
    x_fit, y_fit, r_fit, x_val, y_val, r_val, n_features: int, seed: int,
    class_weights: Optional[torch.Tensor] = None,
) -> LSTMMovementModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMMovementModel(n_features).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce, mse = nn.CrossEntropyLoss(weight=class_weights), nn.MSELoss()

    def loss_fn(x, y, r):
        logits, ret = model(x)
        return ce(logits, y) + REGRESSION_LOSS_WEIGHT * mse(ret, r)

    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, waited = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(x_fit), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            loss_fn(x_fit[idx], y_fit[idx], r_fit[idx]).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(x_val, y_val, r_val).item()
        if val_loss < best_loss - 1e-4:
            best_loss, waited = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def lstm_fit_predict(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: List[str],
    target: str,
    y_train_cls: np.ndarray,
    horizon: int,
    class_weighted: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[float]]]:
    """
    Train on train_idx rows, predict test_idx rows. Returns (class, return %, None).

    class_weighted: weight cross-entropy by inverse class frequency in the fit set,
    so the network cannot score well by ignoring a class (prevents collapse).
    """
    X = data[features].to_numpy(dtype=np.float32)
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0) + 1e-8
    X = np.clip((X - mean) / std, -5, 5).astype(np.float32)

    returns = data[target].to_numpy(dtype=np.float32)
    r_scale = float(returns[train_idx].std()) + 1e-8

    # Rows with a full SEQ_LEN history; split off a purged, most-recent validation slice
    keep = train_idx >= SEQ_LEN - 1
    rows, labels = train_idx[keep], y_train_cls[keep]
    n_val = max(int(len(rows) * VALIDATION_FRACTION), 20)
    fit_rows, fit_labels = rows[: -(n_val + horizon)], labels[: -(n_val + horizon)]
    val_rows, val_labels = rows[-n_val:], labels[-n_val:]

    def tensors(r, y):
        return (
            torch.from_numpy(make_sequences(X, r)).to(DEVICE),
            torch.from_numpy(y.astype(np.int64)).to(DEVICE),
            torch.from_numpy(returns[r] / r_scale).to(DEVICE),
        )

    x_fit, y_fit, r_fit = tensors(fit_rows, fit_labels)
    x_val, y_val, r_val = tensors(val_rows, val_labels)
    x_test = torch.from_numpy(make_sequences(X, test_idx)).to(DEVICE)

    class_weights = None
    if class_weighted:
        counts = np.bincount(fit_labels, minlength=3).astype(np.float32)
        weights = counts.sum() / (3 * np.maximum(counts, 1))
        class_weights = torch.from_numpy(weights).to(DEVICE)

    probs, rets = [], []
    for seed in SEEDS:
        model = _train_one(x_fit, y_fit, r_fit, x_val, y_val, r_val, len(features), seed,
                           class_weights)
        with torch.no_grad():
            logits, ret = model(x_test)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        rets.append(ret.cpu().numpy() * r_scale * 100)  # back to percent

    return np.mean(probs, axis=0).argmax(axis=1), np.mean(rets, axis=0), None
