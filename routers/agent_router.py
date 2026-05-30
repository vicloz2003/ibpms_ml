"""Agent endpoints: policy classification (NLP/DL) and speech-to-text (Whisper)."""
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.schemas import ClassifyRequest, ClassifyResponse, TranscribeResponse
from services import embedding_service, transcribe_service

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """Classify the client's free-text request into the most suitable business policy."""
    return embedding_service.classify(request.text, request.policies)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    """Convert an uploaded audio file to Spanish text (Whisper)."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "wav")
    result = transcribe_service.transcribe(audio_bytes, suffix=suffix)
    return TranscribeResponse(**result)
