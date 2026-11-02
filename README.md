# iBPMS — Servicio de Machine Learning y Deep Learning

Microservicio de **iBPMS** con los modelos locales de aprendizaje automático: clasificación del pedido del cliente, transcripción de voz a texto, predicción de la mejor ruta de un trámite, probabilidad de demora y detección de anomalías.

La descripción completa del sistema, su arquitectura y el resto de los componentes están en el repositorio del backend: **[p1-sw1-backend](https://github.com/vicloz2003/p1-sw1-backend)**.

Está separado de [ibpms_ia](https://github.com/vicloz2003/ibpms_ia) (proxy liviano hacia Gemini) por perfil de recursos: aquí viven los modelos que se cargan en memoria. No tiene estado ni acceso a la base de datos; el backend le envía los datos que necesita, por ejemplo las políticas activas con sus etiquetas.

## Modelos

| Capacidad | Modelo | Tipo |
|---|---|---|
| Clasificar el pedido del cliente en una política | `paraphrase-multilingual-MiniLM-L12-v2` (embeddings) + similitud coseno contra la descripción y las etiquetas de cada política | NLP |
| Voz a texto | Whisper `base` con `faster-whisper` (CPU, int8) | Deep learning |
| Mejor rama en una decisión | Red neuronal (MLP) clasificadora en PyTorch | Supervisado |
| Probabilidad de incumplir el plazo | Red neuronal (MLP) binaria en PyTorch | Supervisado |
| Anomalías y nivel de riesgo | Autoencoder en PyTorch: el error de reconstrucción es el puntaje de anomalía | No supervisado |

**Sobre los datos de entrenamiento.** El sistema no tiene historial de producción todavía, así que los modelos de ruta, demora y riesgo se entrenan con un conjunto **sintético** generado con reglas del dominio y relaciones no triviales entre variables. Cada modelo se evalúa con un conjunto de prueba separado, y sus métricas (exactitud, F1, AUC, tasa de detección) se publican en `/route/metrics`. El mismo proceso puede reentrenarse con trámites reales cuando existan.

**Las predicciones son consultivas.** La recomendación de ruta se muestra al funcionario; la transición la sigue decidiendo el motor de reglas del backend.

## Endpoints

| Método | Ruta | Entrada → salida |
|---|---|---|
| GET | `/`, `/health` | Estado del servicio |
| POST | `/agent/classify` | Texto del cliente y políticas → política recomendada, confianza y alternativas |
| POST | `/agent/transcribe` | Archivo de audio (multipart) → texto, idioma y duración |
| POST | `/route/predict` | Contexto del trámite en una decisión → rama recomendada, confianza, ranking y explicación |
| GET | `/route/metrics` | Métricas de evaluación de los tres modelos de PyTorch |
| POST | `/risk/score` | Lote de trámites activos → por cada uno: puntaje de riesgo, probabilidad de demora, anomalía y prioridad (alta, media o baja) |

La documentación interactiva está en `http://localhost:8001/docs` (Swagger, generada por FastAPI).

### Ejemplo: clasificar un pedido

```json
POST /agent/classify
{
  "text": "necesito un préstamo de 5000 dólares",
  "policies": [
    {"id": "p1", "name": "Solicitud de Crédito Personal", "description": "...", "tags": ["credito", "prestamo"]},
    {"id": "p2", "name": "Reclamo de Garantía", "description": "...", "tags": ["garantia", "reclamo"]}
  ]
}
```

## Stack

FastAPI · PyTorch · sentence-transformers · faster-whisper · NumPy · Pydantic

## Instalación y ejecución

Probado con Python 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

La primera ejecución descarga los modelos (unos 620 MB). Al arrancar, el servicio los carga y entrena en memoria para que la primera petición responda rápido.

| Variable | Descripción |
|---|---|
| `IBPMS_ML_SKIP_WARMUP` | `true` para no cargar los modelos al arrancar (útil en CI) |

## Estructura

```
├── main.py          # Aplicación FastAPI, CORS, registro de routers y precarga de modelos
├── routers/         # agent_router, route_router, risk_router
├── services/        # Un servicio por modelo + generador del conjunto de entrenamiento
└── models/          # Esquemas Pydantic de request y response
```
