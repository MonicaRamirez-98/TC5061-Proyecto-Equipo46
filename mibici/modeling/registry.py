
"""Registro reproducible de modelos MiBici - Equipo 46.

Selecciona modelos existentes en MLflow Tracking y los registra
en Model Registry sin depender de Run IDs o Model IDs locales.

P1: menor MAE de validación.
P2: mayor F1 de validación; Recall y Balanced Accuracy desempatan.
P3: mayor Silhouette entre KMeans K=3,4,5,6.

No entrena modelos ni sobrescribe archivos de datos.
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mibici.config import MLFLOW_EXPERIMENT_NAME

DATA_VERSION = "d038fe16493d"

MODELS = [
    {
        "question": "P1",
        "name": "mibici-equipo46-p1-demanda",
        "candidates": {"P1-HGB-01", "P1-HGB-02", "P1-HGB-03"},
        "metrics": ["validation_mae"],
        "maximize": False,
        "description": (
            "Predicción de demanda por estación para la hora "
            "siguiente. Selección por MAE de validación."
        ),
    },
    {
        "question": "P2",
        "name": "mibici-equipo46-p2-riesgo",
        "candidates": {"P2-HGB-01", "P2-HGB-02", "P2-HGB-03"},
        "metrics": [
            "validation_f1",
            "validation_recall",
            "validation_balanced_accuracy",
        ],
        "maximize": True,
        "description": (
            "Predicción del indicador indirecto de presión "
            "operacional. Selección por F1 de validación."
        ),
    },
    {
        "question": "P3",
        "name": "mibici-equipo46-p3-grupos",
        "candidates": {
            f"P3_kmeans_{k}_station_usage_profiles"
            for k in [3, 4, 5, 6]
        },
        "metrics": ["silhouette", "calinski_harabasz"],
        "maximize": True,
        "description": (
            "Agrupación de estaciones según patrones de uso "
            "mediante StandardScaler y KMeans."
        ),
    },
]


def find_selected_run(client, experiment_id, config):
    """Identifica la mejor corrida válida para una pregunta."""

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=5000,
        order_by=["attributes.start_time DESC"],
    )

    candidates = []

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")

        if run_name not in config["candidates"]:
            continue

        version = (
            run.data.tags.get("data_version")
            or run.data.params.get("data_version")
        )

        if version != DATA_VERSION:
            continue

        uri = run.data.tags.get("logged_model_uri")

        if not uri:
            continue

        if not all(
            key in run.data.metrics
            for key in config["metrics"]
        ):
            continue

        candidates.append((run, uri))

    if not candidates:
        raise RuntimeError(
            f"No hay modelos válidos para {config['question']}. "
            "Ejecuta primero su entrenamiento con artefactos."
        )

    def ranking(item):
        run = item[0]
        return tuple(
            run.data.metrics[key]
            for key in config["metrics"]
        )

    # La lista está ordenada por fecha descendente.
    # En empates exactos preferimos la ejecución más reciente.
    candidates.sort(
        key=ranking,
        reverse=config["maximize"],
    )

    selected, uri = candidates[0]

    return selected, uri


def existing_version(client, name, run_id, model_uri):
    """Evita registrar otra vez el mismo origen."""

    try:
        versions = client.search_model_versions(
            f"name='{name}'"
        )
    except MlflowException:
        return None

    for version in versions:
        tags = version.tags or {}

        if tags.get("source_run_id") == run_id:
            return version

        if (
            version.run_id == run_id
            and version.source == model_uri
        ):
            return version

    return None


def register_models():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"No existe el experimento {MLFLOW_EXPERIMENT_NAME}."
        )

    # Primera fase: validar TODOS los orígenes antes de
    # realizar modificaciones al Model Registry.
    selections = []

    for config in MODELS:
        run, uri = find_selected_run(
            client,
            experiment.experiment_id,
            config,
        )

        # Confirma que el artefacto es recuperable.
        mlflow.sklearn.load_model(uri)

        selections.append((config, run, uri))

        print(
            f"Seleccionado {config['question']}: "
            f"{run.data.tags.get('mlflow.runName')} "
            f"| Run={run.info.run_id}"
        )

    # Segunda fase: crear o reutilizar versiones.
    for config, run, uri in selections:
        name = config["name"]

        version = existing_version(
            client,
            name,
            run.info.run_id,
            uri,
        )

        if version is None:
            version = mlflow.register_model(
                model_uri=uri,
                name=name,
            )

        number = str(version.version)

        client.update_registered_model(
            name=name,
            description=config["description"],
        )

        tags = {
            "team": "46",
            "question": config["question"],
            "data_version": DATA_VERSION,
            "source_run_id": run.info.run_id,
            "source_model_uri": uri,
            "selected_run_name": run.data.tags.get(
                "mlflow.runName", ""
            ),
            "validation_status": "approved",
            "environment": "academic",
        }

        for key, value in tags.items():
            client.set_model_version_tag(
                name=name,
                version=number,
                key=key,
                value=str(value),
            )

        client.update_model_version(
            name=name,
            version=number,
            description=(
                f"Modelo seleccionado de {config['question']}. "
                f"Run: {run.info.run_id}. "
                f"Datos: {DATA_VERSION}."
            ),
        )

        client.set_registered_model_alias(
            name=name,
            alias="champion",
            version=number,
        )

        alias_uri = f"models:/{name}@champion"

        # Comprueba la resolución del alias y la carga.
        resolved = client.get_model_version_by_alias(
            name=name,
            alias="champion",
        )

        if str(resolved.version) != number:
            raise RuntimeError(
                f"El alias de {name} no apunta a {number}."
            )

        loaded = mlflow.sklearn.load_model(alias_uri)

        print(
            f"OK: {name} | version={number} "
            f"| alias=champion "
            f"| clase={type(loaded).__name__}"
        )

    print("\nModel Registry verificado correctamente.")


if __name__ == "__main__":
    register_models()
