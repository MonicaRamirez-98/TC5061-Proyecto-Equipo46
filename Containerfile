# Imagen de la API de predicción MiBici (P1) — Equipo 46
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero, para aprovechar la caché de construcción
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código del proyecto y de la API
COPY mibici/ ./mibici/
COPY api/ ./api/

# Ubicación del Model Registry y modelo a cargar (se pueden cambiar con -e)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MIBICI_MODEL_NAME=mibici-equipo46-p1-demanda \
    MIBICI_MODEL_ALIAS=champion

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]