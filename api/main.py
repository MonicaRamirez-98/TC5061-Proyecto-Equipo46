"""API REST de inferencia en tiempo real para el modelo P1 de MiBici.

Usa el servicio de Axel (mibici.serving.model_loader.MiBiciModelService),
que recupera el modelo del Model Registry de MLflow una sola vez al
arrancar la aplicación.

Ejecución local:
    uvicorn api.main:app --reload --port 8000
Documentación interactiva: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import (
    BatchInput,
    BatchOutput,
    HealthOutput,
    ModelInfoOutput,
    PredictionInput,
    PredictionOutput,
)
from mibici.serving.contract import FEATURE_COLUMNS
from mibici.serving.model_loader import MiBiciModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se carga una sola vez; no se llama a MLflow en cada solicitud.
    try:
        app.state.service = MiBiciModelService()
        logger.info(
            f"Modelo listo: {app.state.service.model_name} "
            f"v{app.state.service.version} ({app.state.service.model_uri})"
        )
    except Exception as exc:  # noqa: BLE001
        # La API arranca igual para que /health reporte el problema.
        app.state.service = None
        logger.error(f"No se pudo cargar el modelo: {type(exc).__name__}: {exc}")
    yield


app = FastAPI(
    title="MiBici · API de predicción de demanda (Equipo 46)",
    description=(
        "Predice cuántos viajes se iniciarán en una estación MiBici "
        "durante la hora siguiente (pregunta P1, regresión)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_service(request: Request) -> MiBiciModelService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Revisa la conexión con MLflow.",
        )
    return service


def _run_prediction(service, records: list[PredictionInput]) -> list[PredictionOutput]:
    payload = [record.model_dump() for record in records]
    try:
        values = service.predict(payload)
    except ValueError as exc:
        # Validación adicional del contrato de Axel.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo interno al predecir")
        raise HTTPException(
            status_code=500, detail="Error interno al generar la predicción."
        ) from exc

    return [
        PredictionOutput(
            station_id=record.station_id,
            predicted_departures_next_hour=round(float(value), 4),
            model_name=service.model_name,
            model_version=str(service.version),
        )
        for record, value in zip(records, values)
    ]


@app.get("/", include_in_schema=False)
def root():
    return {"message": "API MiBici Equipo 46. Documentación en /docs"}


@app.get("/health", response_model=HealthOutput)
def health(request: Request):
    loaded = getattr(request.app.state, "service", None) is not None
    body = HealthOutput(status="ok" if loaded else "degraded", model_loaded=loaded)
    return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())


@app.get("/model-info", response_model=ModelInfoOutput)
def model_info(request: Request):
    service = _get_service(request)
    return ModelInfoOutput(
        model_name=service.model_name,
        model_version=str(service.version),
        model_uri=service.model_uri,
        feature_columns=list(FEATURE_COLUMNS),
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request):
    return _run_prediction(_get_service(request), [payload])[0]


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(payload: BatchInput, request: Request):
    return BatchOutput(predictions=_run_prediction(_get_service(request), payload.records))
