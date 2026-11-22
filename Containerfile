# Basado en el Containerfile real de la Semana 8 del Equipo 46.
# Se conservan Python 3.11, libgomp1, el usuario no privilegiado y la API.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MIBICI_MODEL_NAME=mibici-equipo46-p1-demanda \
    MIBICI_MODEL_ALIAS=champion \
    HOME=/home/app

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-service.txt ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    -r requirements.txt -r requirements-service.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app \
    && mkdir -p /mlflow \
    && chown -R 10001:10001 /mlflow /home/app

COPY --chown=10001:10001 api/ ./api/
COPY --chown=10001:10001 mibici/ ./mibici/
COPY --chown=10001:10001 ops/ ./ops/

USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
