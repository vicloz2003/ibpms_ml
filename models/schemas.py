"""Pydantic request/response schemas for the ibpms_ml service."""
from typing import List, Optional
from pydantic import BaseModel, Field


class PolicyInput(BaseModel):
    """A business policy as provided by the Spring Boot gateway for classification."""
    id: str
    name: str
    description: Optional[str] = ""
    tags: List[str] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    """Client's free-text request + the catalog of active policies to choose from."""
    text: str = Field(..., min_length=1)
    policies: List[PolicyInput] = Field(default_factory=list)


class PolicyMatch(BaseModel):
    policyId: str
    policyName: str
    score: float


class ClassifyResponse(BaseModel):
    """Top recommendation + ranked alternatives + a confidence flag."""
    policyId: Optional[str]
    policyName: Optional[str]
    confidence: float
    confident: bool
    alternatives: List[PolicyMatch]
    message: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float


# ─────────────────────────────────────────────────────────────────────────────
# Risk engine (RF-3): unsupervised anomaly detection + risk/priority scoring (DL).
# ─────────────────────────────────────────────────────────────────────────────

class RiskFeatures(BaseModel):
    """
    Health features of a single ACTIVE process instance, computed by the Spring Boot
    gateway from MongoDB. All are ratios/counts so the model is policy-agnostic.
    """
    processInstanceId: str
    policyId: Optional[str] = ""
    # elapsed time / expected (policy avg) duration. >1 ⇒ slower than usual.
    elapsedRatio: float = Field(0.0, ge=0.0)
    # tasksCompleted / tasksTotal in [0,1]. Low progress + high elapsed ⇒ stuck.
    progressRatio: float = Field(0.0, ge=0.0)
    # avg task work-time so far / global avg for the same nodes. >1 ⇒ slow tasks.
    avgTaskRatio: float = Field(0.0, ge=0.0)
    # longest time a task sat unclaimed in the queue / expected. >1 ⇒ neglected.
    maxQueueRatio: float = Field(0.0, ge=0.0)
    # number of task reassignments (normalized 0..1 by the gateway).
    reassignmentRatio: float = Field(0.0, ge=0.0)
    # pressure of the current node being a known bottleneck, in [0,1].
    bottleneckPressure: float = Field(0.0, ge=0.0)
    # repeated/looped nodes ratio (rework). >0 ⇒ loops.
    reworkRatio: float = Field(0.0, ge=0.0)


class RiskScoreRequest(BaseModel):
    instances: List[RiskFeatures] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    processInstanceId: str
    riskScore: float            # [0,1] — higher = riskier
    anomaly: bool               # reconstruction error above the learned threshold
    priority: str               # HIGH | MEDIUM | LOW (resource allocation, RF-3.4)
    reconstructionError: float
    drivers: List[str]          # top features explaining the risk (RF-3.3)
    recommendation: str         # suggested action / alternate route hint (RF-3.1)


class RiskScoreResponse(BaseModel):
    assessments: List[RiskAssessment]
    threshold: float            # the learned anomaly threshold (for transparency)
    modelInfo: str


# ─────────────────────────────────────────────────────────────────────────────
# Intelligent routing engine (RF-3.1): supervised "best route" prediction (DL).
#
# Unlike the risk autoencoder (unsupervised), route prediction is a *supervised*
# classification problem: given the decision-time context of an instance, predict
# which outgoing branch historically led to the best outcome (fast + successful).
# The model is advisory — the deterministic SpEL engine still controls the flow.
# ─────────────────────────────────────────────────────────────────────────────

class RouteContext(BaseModel):
    """
    Context features available at a decision node, computed by the gateway from the
    instance + its form data. Policy-agnostic ratios/encodings so one model generalizes.
    """
    processInstanceId: str = ""
    nodeId: str = ""
    # Monetary/size magnitude of the request, normalized to ~[0,1] by the gateway.
    amount: float = Field(0.0, ge=0.0)
    # Client segment: 0=residential, 0.5=commercial, 1=industrial.
    clientSegment: float = Field(0.0, ge=0.0, le=1.0)
    # Technical complexity of the request, [0,1].
    complexity: float = Field(0.0, ge=0.0, le=1.0)
    # How complete the submitted documents are, [0,1] (1 = all present & valid).
    docCompleteness: float = Field(0.0, ge=0.0, le=1.0)
    # 1 if the service location is rural (affects field-ops route), else 0.
    zoneRural: float = Field(0.0, ge=0.0, le=1.0)
    # Client's history of prior issues/claims, [0,1].
    priorIssues: float = Field(0.0, ge=0.0, le=1.0)


class RouteBranch(BaseModel):
    """An outgoing branch the engine could take, as offered by the gateway."""
    branchId: str               # the ControlFlow / target node id
    label: str = ""             # human-readable name for the UI


class RoutePredictRequest(BaseModel):
    context: RouteContext
    branches: List[RouteBranch] = Field(default_factory=list)


class BranchScore(BaseModel):
    branchId: str
    label: str
    probability: float          # model's predicted probability this is the best branch


class RoutePredictResponse(BaseModel):
    recommendedBranchId: Optional[str]
    recommendedLabel: Optional[str]
    confidence: float           # probability of the top branch, [0,1]
    confident: bool             # confidence above the decision threshold
    ranking: List[BranchScore]  # all branches ranked by probability
    rationale: str              # short human explanation of the recommendation
    modelInfo: str


# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation metrics — exposed so the rigor (train/test, accuracy, AUC…) is
# transparent and demonstrable, not a black box.
# ─────────────────────────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    name: str
    task: str                   # e.g. "supervised classification", "unsupervised anomaly"
    architecture: str
    trainSamples: int
    testSamples: int
    metrics: dict               # task-specific: accuracy, f1, auc, mae, detectionRate…
    notes: str = ""


class MetricsResponse(BaseModel):
    models: List[ModelMetrics]
