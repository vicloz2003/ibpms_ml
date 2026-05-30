"""Risk engine endpoints (RF-3): unsupervised anomaly/risk scoring (PyTorch autoencoder)."""
from fastapi import APIRouter

from models.schemas import RiskScoreRequest, RiskScoreResponse
from services import risk_service

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/score", response_model=RiskScoreResponse)
def score(request: RiskScoreRequest):
    """Score a batch of active process instances for delay risk, anomalies and priority."""
    return risk_service.score(request)
