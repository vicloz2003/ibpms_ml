"""
Risk engine (RF-3): unsupervised anomaly detection on business-process instances
using a PyTorch autoencoder (Deep Learning, as required by the docente).

Why unsupervised: there is no labeled dataset of "risky" vs "healthy" processes, so a
supervised classifier is not viable. We instead train a small autoencoder on *healthy*
process feature vectors (synthetic, drawn from the distribution of a well-behaved
instance) so it learns to reconstruct "normal". At inference, the per-instance
reconstruction error is the anomaly score: a stuck/slow/looping process reconstructs
poorly → high error → high risk. This covers:
  - RF-3.2 risk of delays/bottlenecks (elapsed/queue/bottleneck features),
  - RF-3.3 anomaly detection (reconstruction error > learned threshold),
  - RF-3.4 resource-allocation priority (risk score bucketed HIGH/MEDIUM/LOW),
  - RF-3.1 a route/action hint derived from the top error-driving feature.
"""
from functools import lru_cache
from typing import List

import numpy as np
import torch
from torch import nn

from models.schemas import (
    RiskFeatures, RiskScoreRequest, RiskScoreResponse, RiskAssessment,
)

# Feature order is fixed — must match RiskFeatures and the gateway's vector builder.
FEATURE_NAMES = [
    "elapsedRatio", "progressRatio", "avgTaskRatio", "maxQueueRatio",
    "reassignmentRatio", "bottleneckPressure", "reworkRatio",
]
N_FEATURES = len(FEATURE_NAMES)

# Human-readable hints per feature, used to explain risk and suggest an action (RF-3.1/3.3).
FEATURE_HINTS = {
    "elapsedRatio":       "El trámite lleva mucho más tiempo del esperado para su política.",
    "progressRatio":      "Avance bajo respecto al tiempo transcurrido (posible estancamiento).",
    "avgTaskRatio":       "Las tareas se están resolviendo más lento que el promedio.",
    "maxQueueRatio":      "Una tarea quedó mucho tiempo sin reclamar; reasignar a otro funcionario.",
    "reassignmentRatio":  "Reasignaciones excesivas; revisar la ruta o el responsable.",
    "bottleneckPressure": "El nodo actual es un cuello de botella conocido; considerar ruta alterna.",
    "reworkRatio":        "Reproceso/bucle detectado; revisar las condiciones de decisión.",
}

SEED = 42
_DEVICE = torch.device("cpu")


class _AutoEncoder(nn.Module):
    """Tiny symmetric MLP autoencoder: 7 → 5 → 3 → 5 → 7."""

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 5), nn.ReLU(),
            nn.Linear(5, 3), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 5), nn.ReLU(),
            nn.Linear(5, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class _RiskModel:
    """Holds the trained autoencoder, the feature scaler, and the anomaly threshold."""

    def __init__(self):
        self.net = _AutoEncoder().to(_DEVICE)
        self.mean = np.zeros(N_FEATURES, dtype=np.float32)
        self.std = np.ones(N_FEATURES, dtype=np.float32)
        self.threshold = 1.0  # reconstruction-error cutoff (set during training)
        self.err_scale = 1.0  # normalizes error → [0,1] risk score

    # ── training ────────────────────────────────────────────────────────────
    def _synthetic_healthy(self, n: int) -> np.ndarray:
        """
        Generate healthy-process feature vectors. A healthy instance:
          - elapsedRatio ~ 1 (on schedule), progressRatio ~ 1 (advancing),
          - avgTaskRatio ~ 1, low queue/reassignment/bottleneck/rework.
        Gaussian noise (clipped ≥0) gives a realistic "normal" manifold.
        """
        rng = np.random.default_rng(SEED)
        cols = [
            rng.normal(1.0, 0.20, n),   # elapsedRatio
            rng.normal(0.9, 0.15, n),   # progressRatio
            rng.normal(1.0, 0.20, n),   # avgTaskRatio
            rng.normal(0.3, 0.15, n),   # maxQueueRatio
            rng.normal(0.1, 0.08, n),   # reassignmentRatio
            rng.normal(0.2, 0.12, n),   # bottleneckPressure
            rng.normal(0.05, 0.05, n),  # reworkRatio
        ]
        data = np.clip(np.stack(cols, axis=1), 0.0, None).astype(np.float32)
        return data

    def train(self, n_samples: int = 4000, epochs: int = 250) -> None:
        torch.manual_seed(SEED)
        raw = self._synthetic_healthy(n_samples)
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0) + 1e-6
        x = torch.from_numpy((raw - self.mean) / self.std).to(_DEVICE)

        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = self.net(x)
            loss = loss_fn(out, x)
            loss.backward()
            opt.step()

        # Per-sample errors on the healthy set define the anomaly threshold.
        self.net.eval()
        with torch.no_grad():
            errs = ((self.net(x) - x) ** 2).mean(dim=1).cpu().numpy()
        # 97.5th percentile of healthy errors → anything above is anomalous.
        self.threshold = float(np.percentile(errs, 97.5)) or 1e-3
        self.err_scale = self.threshold  # risk curve is scaled by the threshold

    # ── inference ───────────────────────────────────────────────────────────
    def _vector(self, f: RiskFeatures) -> np.ndarray:
        return np.array([getattr(f, name) for name in FEATURE_NAMES], dtype=np.float32)

    def score(self, features: List[RiskFeatures]) -> List[RiskAssessment]:
        if not features:
            return []
        raw = np.stack([self._vector(f) for f in features], axis=0)
        xn = (raw - self.mean) / self.std
        x = torch.from_numpy(xn.astype(np.float32)).to(_DEVICE)
        self.net.eval()
        with torch.no_grad():
            recon = self.net(x).cpu().numpy()
        sq_err = (recon - xn) ** 2                 # per-feature squared error
        per_sample = sq_err.mean(axis=1)           # reconstruction error

        out: List[RiskAssessment] = []
        thr = self.threshold if self.threshold > 0 else 1e-3
        for i, f in enumerate(features):
            err = float(per_sample[i])
            # Risk score in [0,1], scaled by the learned threshold so it keeps gradation:
            #  - err ≤ threshold → linearly mapped into the LOW band [0, 0.33),
            #  - err > threshold → soft-saturating curve into [0.33, 1.0] (mild < severe).
            if err <= thr:
                risk = 0.33 * (err / thr)
            else:
                risk = 0.33 + 0.67 * (1.0 - float(np.exp(-(err - thr) / thr)))
            risk = float(min(1.0, max(0.0, risk)))
            anomaly = err > thr
            priority = "HIGH" if risk >= 0.66 else "MEDIUM" if risk >= 0.33 else "LOW"

            # Drivers / recommendation only when the instance is actually at risk.
            if anomaly:
                order = np.argsort(sq_err[i])[::-1]
                drivers = [FEATURE_NAMES[j] for j in order[:3] if sq_err[i][j] > 0.10]
                recommendation = FEATURE_HINTS.get(
                    drivers[0], "Trámite con desviaciones; revisar el avance.") if drivers \
                    else "Trámite con desviaciones; revisar el avance."
            else:
                drivers = []
                recommendation = "Trámite dentro de parámetros normales."
            out.append(RiskAssessment(
                processInstanceId=f.processInstanceId,
                riskScore=round(risk, 4),
                anomaly=bool(anomaly),
                priority=priority,
                reconstructionError=round(err, 6),
                drivers=drivers,
                recommendation=recommendation,
            ))
        return out


@lru_cache(maxsize=1)
def _model() -> _RiskModel:
    m = _RiskModel()
    m.train()
    return m


def warm_up() -> None:
    """Train the autoencoder once at startup so the first request is fast."""
    _model()


def score(request: RiskScoreRequest) -> RiskScoreResponse:
    m = _model()
    assessments = m.score(request.instances)
    return RiskScoreResponse(
        assessments=assessments,
        threshold=round(m.threshold, 6),
        modelInfo="PyTorch autoencoder (unsupervised) — 7→3→7, MSE reconstruction error",
    )
