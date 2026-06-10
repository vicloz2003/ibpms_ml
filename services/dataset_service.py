"""
Realistic synthetic dataset generator for the intelligent routing & risk engine.

Why synthetic (and why it's legitimate): the system has no production history yet
(cold-start). Training on a domain-grounded synthetic corpus is standard practice
(simulation / data augmentation) — what matters is that (a) the data has *non-trivial
conditional structure* so the models learn genuine relationships rather than a laundered
rule, and (b) the same pipeline retrains on real instances once they accumulate.

This module is the single source of truth for the training corpus. It does NOT touch
MongoDB — the operational data (DemoDataSeeder) stays untouched.

Two datasets are produced:
  - Route corpus  : (decision-time context) -> best branch        [supervised classification]
  - Risk corpus   : healthy + injected-anomaly health vectors      [unsupervised + validation]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

SEED = 42

# ── Route context feature order (must match RouteContext + the gateway) ──────────
ROUTE_FEATURES = [
    "amount", "clientSegment", "complexity", "docCompleteness", "zoneRural", "priorIssues",
]
N_ROUTE_FEATURES = len(ROUTE_FEATURES)

# Three canonical branches every multi-way decision in the ELECSUR policies maps to.
# The model is generic: the gateway maps a policy's real branches onto these archetypes.
BRANCH_LABELS = ["Vía Rápida", "Revisión Completa", "Inspección de Campo"]
N_BRANCHES = len(BRANCH_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth outcome model (hidden from the learner).
#
# Each branch has an expected COST = processing time + expected rework. The "best"
# branch is the lowest-cost one for a given context. Costs depend on the context with
# genuine INTERACTIONS (complexity×rural, amount, docs), so the classifier must learn
# a non-linear decision boundary — not a single threshold rule.
# ─────────────────────────────────────────────────────────────────────────────

def _branch_costs(c: np.ndarray) -> np.ndarray:
    """
    c: route context vector [amount, segment, complexity, docs, rural, priorIssues].
    Returns the expected cost of each of the 3 branches (lower = better).
    """
    amount, segment, complexity, docs, rural, prior = c

    # Branch 0 — Vía Rápida: cheap baseline, but bounces back (rework) when the request
    # actually needed review/inspection. Penalized hard by missing prerequisites.
    express = (
        1.0
        + 3.0 * (1.0 - docs)                 # incomplete docs → rejected, redo
        + 2.5 * amount                       # high amount skipped review → later rework
        + 1.5 * segment                      # commercial/industrial rarely fit the fast lane
        + 2.0 * complexity * rural           # technical + rural can't be solved remotely
    )

    # Branch 1 — Revisión Completa: thorough, medium baseline. Absorbs amount/doc risk
    # (no rework there) but cannot resolve a field problem on its own.
    review = (
        2.3
        + 0.6 * complexity                   # still some load for complex cases
        + 2.2 * complexity * rural           # a desk review can't fix a rural field issue
        + 0.3 * (1.0 - docs)                 # minor: chases missing docs but handles it
    )

    # Branch 2 — Inspección de Campo: slow + dispatch cost, but the ONLY branch that
    # resolves complex rural cases without rework. Wasteful when not needed.
    field = (
        3.2
        + 1.8 * (1.0 - complexity)           # overkill for simple cases
        + 1.2 * (1.0 - rural)                # dispatching to urban/easy access is wasteful
        - 2.6 * complexity * rural           # its reason to exist: resolves the hard case
    )

    return np.array([express, review, field], dtype=np.float32)


@dataclass
class RouteDataset:
    X: np.ndarray          # (n, N_ROUTE_FEATURES) float32
    y: np.ndarray          # (n,) int64 best-branch index
    feature_names: list
    branch_labels: list


def _sample_context(rng: np.random.Generator) -> np.ndarray:
    """Draw one realistic decision-time context. Distributions reflect a utility's mix."""
    amount      = float(np.clip(rng.beta(2.0, 3.0), 0.0, 1.0))     # skewed toward smaller
    segment     = float(rng.choice([0.0, 0.5, 1.0], p=[0.7, 0.2, 0.1]))  # mostly residential
    complexity  = float(np.clip(rng.beta(2.0, 4.0), 0.0, 1.0))     # mostly low complexity
    docs        = float(np.clip(rng.beta(5.0, 2.0), 0.0, 1.0))     # usually fairly complete
    rural       = float(rng.choice([0.0, 1.0], p=[0.65, 0.35]))    # 35% rural
    prior       = float(np.clip(rng.beta(1.5, 6.0), 0.0, 1.0))     # most clients clean history
    return np.array([amount, segment, complexity, docs, rural, prior], dtype=np.float32)


def generate_route_dataset(n: int = 1500, noise: float = 0.35,
                           seed: int = SEED) -> RouteDataset:
    """
    Build a labeled route corpus. Gaussian noise is added to the hidden costs so the
    label isn't a perfectly separable rule — the model must generalize, and the test
    accuracy is meaningful (a trivial rule would hit ~100%, a real boundary won't).
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, N_ROUTE_FEATURES), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i in range(n):
        c = _sample_context(rng)
        costs = _branch_costs(c) + rng.normal(0.0, noise, size=N_BRANCHES).astype(np.float32)
        X[i] = c
        y[i] = int(np.argmin(costs))     # best = lowest-cost branch
    return RouteDataset(X=X, y=y, feature_names=list(ROUTE_FEATURES),
                        branch_labels=list(BRANCH_LABELS))


# ─────────────────────────────────────────────────────────────────────────────
# Risk corpus: healthy health-vectors (for the autoencoder) + injected anomalies
# (held out, used to MEASURE detection rate — turns the autoencoder from a claim
# into something with a reported metric).
# ─────────────────────────────────────────────────────────────────────────────

RISK_FEATURES = [
    "elapsedRatio", "progressRatio", "avgTaskRatio", "maxQueueRatio",
    "reassignmentRatio", "bottleneckPressure", "reworkRatio",
]
N_RISK_FEATURES = len(RISK_FEATURES)


def generate_healthy_risk(n: int = 1500, seed: int = SEED) -> np.ndarray:
    """Healthy instances: on schedule, advancing, low queue/rework. The 'normal' manifold."""
    rng = np.random.default_rng(seed)
    cols = [
        rng.normal(1.0, 0.20, n),   # elapsedRatio
        rng.normal(0.9, 0.15, n),   # progressRatio
        rng.normal(1.0, 0.20, n),   # avgTaskRatio
        rng.normal(0.3, 0.15, n),   # maxQueueRatio
        rng.normal(0.1, 0.08, n),   # reassignmentRatio
        rng.normal(0.2, 0.12, n),   # bottleneckPressure
        rng.normal(0.05, 0.05, n),  # reworkRatio
    ]
    return np.clip(np.stack(cols, axis=1), 0.0, None).astype(np.float32)


def generate_anomalous_risk(n: int = 300, seed: int = SEED + 1) -> np.ndarray:
    """
    Injected anomalies: stuck (high elapsed, low progress), neglected (high queue),
    looping (high rework), churned (high reassignment). Used ONLY to validate that the
    autoencoder's threshold actually catches them (reported as detectionRate).
    """
    rng = np.random.default_rng(seed)
    cols = [
        rng.normal(2.4, 0.5, n),    # elapsedRatio — way over time
        rng.normal(0.3, 0.15, n),   # progressRatio — barely advancing
        rng.normal(1.9, 0.5, n),    # avgTaskRatio — slow tasks
        rng.normal(1.6, 0.5, n),    # maxQueueRatio — neglected in queue
        rng.normal(0.6, 0.2, n),    # reassignmentRatio — churned
        rng.normal(0.8, 0.2, n),    # bottleneckPressure — stuck at bottleneck
        rng.normal(0.5, 0.2, n),    # reworkRatio — looping
    ]
    return np.clip(np.stack(cols, axis=1), 0.0, None).astype(np.float32)


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2,
                     seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic shuffle + split. Keeps the evaluation honest (held-out test set)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]
