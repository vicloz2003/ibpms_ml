# ibpms_ml — Servicio ML/DL (:8001)

Microservicio Python (FastAPI) para las funcionalidades de Machine Learning / Deep
Learning del iBPMS. Separado de `ibpms_ia` (:8000, proxy liviano a Gemini) por perfil
de recursos: aquí viven los modelos pesados de PyTorch/Whisper.

## Capacidades
- **Agente (NLP/DL):** clasifica el problema del cliente en la política de negocio
  adecuada usando embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Speech-to-text:** transcribe audio a texto con Whisper (`faster-whisper`).
- *(futuro)* Motor de riesgos: detección de anomalías con autoencoder.

## Requisitos
- Python 3.10 (probado con 3.10.7)

## Setup (Windows)
```bash
cd ibpms_ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```
La primera ejecución descarga los modelos (~620MB total) y los cachea.

## Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` , `/health` | Healthcheck |
| POST | `/agent/classify` | `{text, policies[]}` → política recomendada + confianza + alternativas |
| POST | `/agent/transcribe` | multipart `file` (audio) → `{text, language, duration}` |

### Ejemplo classify
```json
POST /agent/classify
{
  "text": "necesito un préstamo de 5000 dólares",
  "policies": [
    {"id":"p1","name":"Solicitud de Credito Personal","description":"...","tags":["credito","prestamo"]},
    {"id":"p2","name":"Reclamo de Garantia","description":"...","tags":["garantia","reclamo"]}
  ]
}
```
