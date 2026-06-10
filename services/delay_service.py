"""
Delay / SLA-breach prediction (RF-3.2) — SUPERVISED deep learning.

Given an in-flight instance's runtime health vector, predict the *probability it will breach
its SLA*. This complements the autoencoder:
  - Autoencoder (unsupervised) answers "is this instance abnormal right now?"
  - Delay predictor (supervised) answers "what is the calibrated probability it ends late?"

Rigor: binary classifier with BCE loss, held-out test set, reported AUC + accuracy.
"""
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from models.schemas import ModelMetrics
from services import dataset_service as ds

SEED = 42
_DEVICE = torch.device("cpu")
_MODEL_INFO = "PyTorch MLP 7→16→8→1 (supervised, BCE) — SLA-breach probability"


class _DelayNet(nn.Module):
    """MLP binary classifier: 7 health features → P(SLA breach)."""

    def __init__(self, n_in: int = ds.N_RISK_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[y_true == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


class _DelayModel:
    def __init__(self):
        self.net = _DelayNet().to(_DEVICE)
        self.mean = np.zeros(ds.N_RISK_FEATURES, dtype=np.float32)
        self.std = np.ones(ds.N_RISK_FEATURES, dtype=np.float32)
        self.metrics: dict = {}
        self.n_train = 0
        self.n_test = 0

    def train(self, n: int = 1800, epochs: int = 400) -> None:
        torch.manual_seed(SEED)
        X, y = ds.generate_delay_dataset(n=n)
        Xtr, ytr, Xte, yte = ds.train_test_split(X, y, test_frac=0.2)
        self.n_train, self.n_test = len(Xtr), len(Xte)

        self.mean = Xtr.mean(axis=0)
        self.std = Xtr.std(axis=0) + 1e-6
        xtr = torch.from_numpy(((Xtr - self.mean) / self.std).astype(np.float32))
        ytr_t = torch.from_numpy(ytr.astype(np.float32))

        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(xtr), ytr_t)
            loss.backward()
            opt.step()

        # Evaluate on held-out test.
        self.net.eval()
        xte = torch.from_numpy(((Xte - self.mean) / self.std).astype(np.float32))
        with torch.no_grad():
            scores = torch.sigmoid(self.net(xte)).cpu().numpy()
        pred = (scores >= 0.5).astype(np.int64)
        acc = float((pred == yte).mean())
        self.metrics = {
            "auc": round(_auc(yte, scores), 4),
            "accuracy": round(acc, 4),
            "positiveRate": round(float(yte.mean()), 4),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (n, 7) raw health vectors → P(breach) per row."""
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        xn = ((X - self.mean) / self.std).astype(np.float32)
        x = torch.from_numpy(xn)
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(x)).cpu().numpy()


@lru_cache(maxsize=1)
def _model() -> _DelayModel:
    m = _DelayModel()
    m.train()
    return m


def warm_up() -> None:
    _model()


def predict_proba(X: np.ndarray) -> np.ndarray:
    return _model().predict_proba(X)


def metrics() -> ModelMetrics:
    m = _model()
    return ModelMetrics(
        name="Delay Predictor",
        task="supervised classification (SLA breach)",
        architecture="MLP 7→16→8→1, sigmoid, BCE",
        trainSamples=m.n_train,
        testSamples=m.n_test,
        metrics=m.metrics,
        notes="Calibrated probability an in-flight instance ends late. Reported AUC on test.",
    )
