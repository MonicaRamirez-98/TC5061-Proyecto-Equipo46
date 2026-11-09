FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://host.containers.internal:5000 \
    MIBICI_MODEL_NAME=mibici-equipo46-p1-demanda \
    MIBICI_MODEL_ALIAS=champion

WORKDIR /app

# scikit-learn's Linux wheels need the OpenMP runtime at execution time.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

COPY --chown=10001:10001 api/ ./api/
COPY --chown=10001:10001 mibici/ ./mibici/

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
