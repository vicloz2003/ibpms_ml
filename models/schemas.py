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
