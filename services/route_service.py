"""
Intelligent route prediction (RF-3.1) — SUPERVISED deep learning.

Given the decision-time context of a process instance, an MLP classifier predicts which
outgoing branch historically led to the best outcome (fastest, least rework). This is the
DL capability the deterministic SpEL engine lacked: it *predicts the best route*.

Critically, it is ADVISORY. The gateway shows the recommendation to the functionary/admin;
the rule-based engine still executes the actual transition. The model never hijacks the flow.

Rigor (not an MVP):
  - Proper model class for the task: multi-class classifier, not an autoencoder hack.
  - Held-out test set; reported accuracy, macro-F1 and confusion matrix.
  - Trained on a domain-grounded corpus with non-trivial feature interactions, so the
    learned boundary is genuine (a trivial threshold rule would not generalize).
"""
from functools import lru_cache
from typing import List

import numpy as np
import torch
from torch import nn

from models.schemas import (
    RoutePredictRequest, RoutePredictResponse, BranchScore, ModelMetrics,
)
from services import dataset_service as ds

SEED = 42
_DEVICE = torch.device("cpu")

# Above this top-branch probability we treat the recommendation as confident.
# (3 classes ⇒ chance = 0.33; 0.45 is a meaningful margin over chance.)
CONFIDENCE_THRESHOLD = 0.45

_MODEL_INFO = "PyTorch MLP 6→16→8→3 (supervised, cross-entropy) — route classifier"


class _RouteNet(nn.Module):
    """Small MLP classifier: 6 context features → 3 branch archetypes."""

    def __init__(self, n_in: int = ds.N_ROUTE_FEATURES, n_out: int = ds.N_BRANCHES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, n_out),
        )

    def forward(self, x):
        return self.net(x)  # logits


def _macro_f1(cm: np.ndarray) -> float:
    """Macro-averaged F1 from a confusion matrix (rows = true, cols = pred)."""
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


class _RouteModel:
    """Trained classifier + feature scaler + evaluation metrics."""

    def __init__(self):
        self.net = _RouteNet().to(_DEVICE)
        self.mean = np.zeros(ds.N_ROUTE_FEATURES, dtype=np.float32)
        self.std = np.ones(ds.N_ROUTE_FEATURES, dtype=np.float32)
        self.labels = list(ds.BRANCH_LABELS)
        self.metrics: dict = {}
        self.n_train = 0
        self.n_test = 0

    # ── training + evaluation ────────────────────────────────────────────────
    def train(self, n: int = 1500, epochs: int = 400) -> None:
        torch.manual_seed(SEED)
        data = ds.generate_route_dataset(n=n)
        Xtr, ytr, Xte, yte = ds.train_test_split(data.X, data.y, test_frac=0.2)
        self.n_train, self.n_test = len(Xtr), len(Xte)

        # Standardize on TRAIN statistics only (no leakage into the test set).
        self.mean = Xtr.mean(axis=0)
        self.std = Xtr.std(axis=0) + 1e-6
        xtr = torch.from_numpy(((Xtr - self.mean) / self.std).astype(np.float32))
        ytr_t = torch.from_numpy(ytr)

        # Inverse-frequency class weights: the corpus is imbalanced (most cases need
        # full review), which would bias the model toward the majority class. Weighting
        # the loss is the standard fix for imbalanced classification.
        counts = np.bincount(ytr, minlength=ds.N_BRANCHES).astype(np.float32)
        weights = (counts.sum() / (ds.N_BRANCHES * np.maximum(counts, 1.0)))
        wt = torch.from_numpy(weights.astype(np.float32))

        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=wt)
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(xtr), ytr_t)
            loss.backward()
            opt.step()

        # Evaluate on the held-out test set.
        self.net.eval()
        xte = torch.from_numpy(((Xte - self.mean) / self.std).astype(np.float32))
        with torch.no_grad():
            pred = self.net(xte).argmax(dim=1).cpu().numpy()
        acc = float((pred == yte).mean())
        cm = np.zeros((ds.N_BRANCHES, ds.N_BRANCHES), dtype=np.int64)
        for t, p in zip(yte, pred):
            cm[t, p] += 1
        self.metrics = {
            "accuracy": round(acc, 4),
            "macroF1": round(_macro_f1(cm), 4),
            "confusionMatrix": cm.tolist(),
            "classes": self.labels,
        }

    # ── inference ────────────────────────────────────────────────────────────
    def _probs(self, context_vec: np.ndarray) -> np.ndarray:
        xn = ((context_vec - self.mean) / self.std).astype(np.float32)
        x = torch.from_numpy(xn).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs


@lru_cache(maxsize=1)
def _model() -> _RouteModel:
    m = _RouteModel()
    m.train()
    return m


def warm_up() -> None:
    """Train the classifier once at startup so the first request is fast."""
    _model()


def metrics() -> ModelMetrics:
    m = _model()
    return ModelMetrics(
        name="Route Predictor",
        task="supervised classification (best branch)",
        architecture="MLP 6→16→8→3, softmax, cross-entropy",
        trainSamples=m.n_train,
        testSamples=m.n_test,
        metrics=m.metrics,
        notes="Advisory only — the deterministic engine controls the actual transition.",
    )


def _context_vector(ctx) -> np.ndarray:
    return np.array([getattr(ctx, name) for name in ds.ROUTE_FEATURES], dtype=np.float32)


def predict(request: RoutePredictRequest) -> RoutePredictResponse:
    """
    Predict the best branch for the given context.

    The model scores the 3 branch archetypes (Vía Rápida / Revisión Completa /
    Inspección de Campo). The gateway provides the policy's actual branches *in archetype
    order*; we map the archetype probabilities onto them and rank. If fewer than 3 branches
    are offered, probabilities are renormalized over those available.
    """
    m = _model()
    probs = m._probs(_context_vector(request.context))  # (3,)

    branches = request.branches
    if not branches:
        # No concrete branches: expose the archetype ranking directly.
        ranking = [
            BranchScore(branchId=str(i), label=m.labels[i], probability=round(float(probs[i]), 4))
            for i in range(ds.N_BRANCHES)
        ]
    else:
        # Map archetype prob onto each offered branch by position (gateway aligns them).
        raw = np.array([probs[i] if i < len(probs) else 0.0 for i in range(len(branches))],
                       dtype=np.float32)
        total = raw.sum()
        norm = raw / total if total > 0 else np.full(len(branches), 1.0 / len(branches))
        ranking = [
            BranchScore(branchId=b.branchId,
                        label=b.label or (m.labels[i] if i < len(m.labels) else b.branchId),
                        probability=round(float(norm[i]), 4))
            for i, b in enumerate(branches)
        ]

    ranking.sort(key=lambda r: r.probability, reverse=True)
    top = ranking[0]
    confident = top.probability >= CONFIDENCE_THRESHOLD
    rationale = (
        f"Según el histórico, '{top.label}' es la ruta con mejor desenlace para este "
        f"contexto ({int(round(top.probability * 100))}% de probabilidad)."
        if confident else
        "El contexto es ambiguo entre rutas; se sugiere decisión del funcionario."
    )
    return RoutePredictResponse(
        recommendedBranchId=top.branchId if confident else None,
        recommendedLabel=top.label if confident else None,
        confidence=round(float(top.probability), 4),
        confident=confident,
        ranking=ranking,
        rationale=rationale,
        modelInfo=_MODEL_INFO,
    )
