"""
Speech-to-text via faster-whisper (Deep Learning).

The professor requires audio processing to be based on deep learning / NLP. Whisper is
a transformer-based ASR model; faster-whisper is its CTranslate2 reimplementation,
which runs efficiently on CPU. The device only records audio and sends it here.
"""
import tempfile
from functools import lru_cache

from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # good Spanish accuracy, CPU-friendly (~150MB)


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    # int8 quantization → fast on CPU, low memory.
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def warm_up() -> None:
    # Model weights download/load on first construction.
    _model()


def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> dict:
    """Transcribe raw audio bytes to Spanish text. Returns text + language + duration."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    segments, info = _model().transcribe(tmp_path, language="es", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()

    return {
        "text": text,
        "language": info.language,
        "duration": round(info.duration, 2),
    }
