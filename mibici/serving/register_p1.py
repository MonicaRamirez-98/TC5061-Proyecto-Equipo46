"""Registra únicamente P1 de MiBici en MLflow Model Registry.

Uso, desde raíz del repositorio con MLFLOW_TRACKING_URI configurada:
    python -m mibici.serving.register_p1

Selecciona el menor validation_mae de P1-HGB-01/02/03 para la
versión de datos del manifiesto local, comprueba la firma e instala
el alias champion sin requerir experimentos P2 ni P3.
"""

import json
import math
import os

import mlflow
from mlflow import MlflowClient

from mibici.config import MANIFEST_PATH, MLFLOW_EXPERIMENT_NAME
from mibici.serving.contract import FEATURE_COLUMNS

REGISTERED_NAME = "mibici-equipo46-p1-demanda"
CANDIDATE_NAMES = {"P1-HGB-01", "P1-HGB-02", "P1-HGB-03"}


def main():
    if not MANIFEST_PATH.is_file():
        raise SystemExit("Falta data/manifest/data_manifest.json; realiza la ingesta primero.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise SystemExit("La ingesta está incompleta; no se registrará el modelo.")
    data_version = manifest["data_version"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"No existe el experimento {MLFLOW_EXPERIMENT_NAME} en {tracking_uri}.")

    candidates = []
    for run in client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=5000,
        order_by=["attributes.start_time DESC"],
    ):
        name = run.data.tags.get("mlflow.runName")
        version = run.data.tags.get("data_version") or run.data.params.get("data_version")
        source_uri = run.data.tags.get("logged_model_uri")
        metric = run.data.metrics.get("validation_mae")
        if (
            name in CANDIDATE_NAMES
            and version == data_version
            and source_uri
            and metric is not None
            and math.isfinite(metric)
        ):
            candidates.append((float(metric), run.info.start_time or 0, run, source_uri))

    if not candidates:
        raise SystemExit(
            "No hay candidatos P1 válidos con la versión de datos "
            f"{data_version}. Ejecuta antes: mlflow run . -e search_p1 "
            "--env-manager local --experiment-name mibici-equipo46"
        )

    mae, _, run, source_uri = min(candidates, key=lambda row: (row[0], -row[1]))
    name = run.data.tags.get("mlflow.runName")
    print(f"Seleccionado: {name} | MAE validación={mae:.6f} | Run ID={run.info.run_id}")
    print(f"Origen: {source_uri}")
    print(f"Versión de datos: {data_version}")

    # Antes de registrar, comprobar que el artefacto existe y su firma es compatible.
    loaded = mlflow.pyfunc.load_model(source_uri)
    signature = loaded.metadata.signature if loaded.metadata else None
    if signature is None or signature.inputs is None:
        raise RuntimeError("El candidato no contiene firma de entrada MLflow.")
    signature_columns = [item.name for item in signature.inputs.inputs]
    if signature_columns != list(FEATURE_COLUMNS):
        raise RuntimeError(
            "La firma P1 no coincide con las ocho variables acordadas: "
            f"{signature_columns}"
        )

    # No duplicar una versión ya registrada desde la misma corrida.
    try:
        existing_versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    except Exception as exc:
        # Un modelo aún no creado también puede provocar una excepción, pero
        # no ocultamos fallos no relacionados con ausencia del modelo.
        from mlflow.exceptions import MlflowException
        if not isinstance(exc, MlflowException):
            raise
        existing_versions = []

    existing = next(
        (
            v for v in existing_versions
            if (v.tags or {}).get("source_run_id") == run.info.run_id
            or (v.run_id == run.info.run_id and v.source == source_uri)
        ),
        None,
    )
    if existing is None:
        registered = mlflow.register_model(model_uri=source_uri, name=REGISTERED_NAME)
    else:
        registered = existing

    number = str(registered.version)
    client.update_registered_model(
        name=REGISTERED_NAME,
        description="P1 MiBici: predicción de demanda de viajes por estación en la siguiente hora.",
    )
    for key, value in {
        "team": "46",
        "question": "P1",
        "data_version": data_version,
        "source_run_id": run.info.run_id,
        "source_model_uri": source_uri,
        "selected_run_name": name,
        "validation_mae": f"{mae:.6f}",
        "validation_status": "approved",
        "environment": "academic",
    }.items():
        client.set_model_version_tag(REGISTERED_NAME, number, key, str(value))
    client.set_registered_model_alias(REGISTERED_NAME, "champion", number)

    # Comprobar el alias y la recuperación, ahora desde Model Registry.
    resolved = client.get_model_version_by_alias(REGISTERED_NAME, "champion")
    if str(resolved.version) != number:
        raise RuntimeError("El alias champion no apunta a la versión recién seleccionada.")
    mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@champion")
    print(f"OK: {REGISTERED_NAME} | versión={number} | alias=champion")
    print("Modelo P1 registrado y recuperable desde MLflow Model Registry.")


if __name__ == "__main__":
    main()
