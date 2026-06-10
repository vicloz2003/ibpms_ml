"""
Intelligent routing endpoints (RF-3.1): supervised best-route prediction (PyTorch MLP).

Advisory only — the recommendation is shown to the functionary; the deterministic
SpEL engine in Spring Boot still executes the actual transition.
"""
from fastapi import APIRouter

from models.schemas import (
    RoutePredictRequest, RoutePredictResponse, MetricsResponse,
)
from services import route_service, risk_service

router = APIRouter(prefix="/route", tags=["route"])


@router.post("/predict", response_model=RoutePredictResponse)
def predict(request: RoutePredictRequest):
    """Predict the best outgoing branch for an instance's decision-time context."""
    return route_service.predict(request)


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    """
    Evaluation metrics for the DL models (route classifier + risk autoencoder).
    Exposed so the rigor — train/test split, accuracy, detection rate — is transparent.
    """
    return MetricsResponse(models=[
        route_service.metrics(),
        risk_service.metrics(),
    ])
