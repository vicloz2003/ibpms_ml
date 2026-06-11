"""
Speech-to-text via faster-whisper (Deep Learning).

The professor requires audio processing to be based on deep learning / NLP. Whisper is
a transformer-based ASR model; faster-whisper is its CTranslate2 reimplementation,
which runs efficiently on CPU. The device only records audio and sends it here.
"""
import os
import subprocess
import tempfile
from functools import lru_cache

from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # good Spanish accuracy, CPU-friendly (~150MB)


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    # int8 quantization → fast on CPU, low memory.
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def warm_up() -> None:
    _model()


def _to_wav16k(src_path: str) -> str:
    """Convert any audio format to 16kHz mono PCM WAV — the format Whisper expects.
    Robust against whatever container/codec the client sends (Android M4A/AAC,
    web WebM/Opus). Requires ffmpeg on the host.
    """
    dst = src_path + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", dst],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return dst


def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> dict:
    """Transcribe raw audio bytes to Spanish text. Returns text + language + duration."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        src_path = tmp.name

    wav_path = _to_wav16k(src_path)
    try:
        segments, info = _model().transcribe(wav_path, language="es", beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(wav_path)
        os.unlink(src_path)

    return {
        "text": text,
        "language": info.language,
        "duration": round(info.duration, 2),
    }
