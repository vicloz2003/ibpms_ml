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
    RiskFeatures, RiskScoreRequest, RiskScoreResponse, RiskAssessment, ModelMetrics,
)
from services import dataset_service as ds
from services import delay_service

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
        self.metrics: dict = {}
        self.n_train = 0
        self.n_test = 0

    # ── training + validation ─────────────────────────────────────────────────
    def train(self, n_samples: int = 1500, epochs: int = 250) -> None:
        torch.manual_seed(SEED)
        # Healthy corpus from the shared dataset module (domain-grounded normal manifold).
        raw = ds.generate_healthy_risk(n=n_samples)
        self.n_train = len(raw)
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

        # Validate against held-out INJECTED anomalies → reportable detection metrics.
        anom = ds.generate_anomalous_risk(n=300)
        self.n_test = len(anom)
        xa = torch.from_numpy(((anom - self.mean) / self.std).astype(np.float32)).to(_DEVICE)
        with torch.no_grad():
            anom_err = ((self.net(xa) - xa) ** 2).mean(dim=1).cpu().numpy()
        detection_rate = float((anom_err > self.threshold).mean())   # true positives
        false_positive = float((errs > self.threshold).mean())       # on healthy set
        self.metrics = {
            "detectionRate": round(detection_rate, 4),
            "falsePositiveRate": round(false_positive, 4),
            "threshold": round(self.threshold, 6),
            "thresholdPercentile": 97.5,
        }

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

        # RF-3.2: supervised P(SLA breach) for the same batch (delay forecast).
        delay_probs = delay_service.predict_proba(raw)

        out: List[RiskAssessment] = []
        thr = self.threshold if self.threshold > 0 else 1e-3
        for i, f in enumerate(features):
            err = float(per_sample[i])
            # Anomaly-based risk in [0,1], scaled by the learned threshold so it keeps gradation:
            #  - err ≤ threshold → linearly mapped into the LOW band [0, 0.33),
            #  - err > threshold → soft-saturating curve into [0.33, 1.0] (mild < severe).
            if err <= thr:
                risk = 0.33 * (err / thr)
            else:
                risk = 0.33 + 0.67 * (1.0 - float(np.exp(-(err - thr) / thr)))
            risk = float(min(1.0, max(0.0, risk)))
            anomaly = err > thr
            delay_p = float(delay_probs[i]) if len(delay_probs) else 0.0

            # RF-3.4: explainable priority = a transparent blend of the DL signals plus the
            # business importance. Not a black box — each term is auditable.
            #   anomaly risk (0.45) + delay forecast (0.40) + business weight (0.15)
            priority_score = float(min(1.0, 0.45 * risk + 0.40 * delay_p + 0.15 * f.businessWeight))
            priority = ("HIGH" if priority_score >= 0.60
                        else "MEDIUM" if priority_score >= 0.33 else "LOW")

            # Drivers / recommendation only when the instance is actually at risk.
            if anomaly or delay_p >= 0.5:
                order = np.argsort(sq_err[i])[::-1]
                drivers = [FEATURE_NAMES[j] for j in order[:3] if sq_err[i][j] > 0.10]
                recommendation = FEATURE_HINTS.get(
                    drivers[0], "Trámite con desviaciones; revisar el avance.") if drivers \
                    else "Alta probabilidad de demora; priorizar la atención."
            else:
                drivers = []
                recommendation = "Trámite dentro de parámetros normales."
            out.append(RiskAssessment(
                processInstanceId=f.processInstanceId,
                riskScore=round(risk, 4),
                delayProbability=round(delay_p, 4),
                anomaly=bool(anomaly),
                priority=priority,
                priorityScore=round(priority_score, 4),
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


def metrics() -> ModelMetrics:
    m = _model()
    return ModelMetrics(
        name="Risk Autoencoder",
        task="unsupervised anomaly detection",
        architecture="PyTorch autoencoder 7→5→3→5→7, MSE reconstruction error",
        trainSamples=m.n_train,
        testSamples=m.n_test,
        metrics=m.metrics,
        notes="Threshold = 97.5th percentile of healthy reconstruction error. "
              "detectionRate measured on held-out injected anomalies.",
    )


def score(request: RiskScoreRequest) -> RiskScoreResponse:
    m = _model()
    assessments = m.score(request.instances)
    return RiskScoreResponse(
        assessments=assessments,
        threshold=round(m.threshold, 6),
        modelInfo="PyTorch autoencoder (unsupervised) — 7→3→7, MSE reconstruction error",
    )
