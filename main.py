"""
iBPMS ML Service (:8001)

Heavy ML/DL microservice (PyTorch + Whisper). Separated from ibpms_ia (:8000, the
lightweight Gemini proxy) by resource profile. Hosts:
  - Agent: policy classification via multilingual sentence embeddings.
  - Speech-to-text: Whisper.
  - (future) Risk engine: anomaly detection autoencoder.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers.agent_router import router as agent_router
from routers.risk_router import router as risk_router

load_dotenv()

app = FastAPI(
    title="iBPMS ML Service",
    description="Microservicio de ML/DL: agente NLP, speech-to-text y análisis de riesgos",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:3000",
        "http://localhost:*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)
app.include_router(risk_router)


@app.on_event("startup")
def _startup():
    # Warm up models unless explicitly skipped (e.g. in CI). Comment-friendly via env.
    if os.getenv("IBPMS_ML_SKIP_WARMUP", "false").lower() != "true":
        try:
            from services import embedding_service
            embedding_service.warm_up()
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Embedding warm-up failed: {e}")
        try:
            from services import risk_service
            risk_service.warm_up()
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Risk autoencoder warm-up failed: {e}")


@app.get("/")
def root():
    return {"message": "iBPMS ML Service running", "port": 8001}


@app.get("/health")
def health():
    return {"status": "ok"}
