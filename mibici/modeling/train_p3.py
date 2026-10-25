
"""Agrupa estaciones por patrón de uso y registra P3 en MLflow.

Incluye:
- Ingesta de los archivos Parquet versionados.
- Entrenamiento de KMeans con StandardScaler.
- Métricas de evaluación y distribución de clusters.
- Registro de modelos con firma y ejemplo de entrada.
- Artefactos de datos, configuración y visualización.
- Compatibilidad con corridas individuales y nested runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    PROJECT_TRAIN_DEFAULTS,
    PROJ_ROOT,
    RANDOM_SEED,
)
from mibici.features import build_station_profiles


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Carga viajes, estaciones y manifiesto de datos."""

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    if not manifest.get("complete", False):
        raise ValueError(
            "La ingesta está incompleta. "
            "Ejecuta primero mlflow run . -e ingest."
        )

    trip_files = [
        item
        for item in manifest.get("files", [])
        if item.get("name", "").startswith("trips_")
    ]

    if not trip_files:
        raise ValueError(
            "El manifiesto no contiene archivos mensuales de viajes."
        )

    trip_paths = [
        PROCESSED_DATA_DIR / item["name"]
        for item in trip_files
    ]

    stations_path = PROCESSED_DATA_DIR / "stations.parquet"

    missing = [
        str(path)
        for path in [*trip_paths, stations_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos Parquet:\n"
            + "\n".join(missing)
        )

    trips = pd.concat(
        [pd.read_parquet(path) for path in trip_paths],
        ignore_index=True,
    )

    stations = pd.read_parquet(stations_path)

    return trips, stations, manifest


def git_commit() -> str:
    """Obtiene el commit actual para trazabilidad."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(
    model_name: str | None = None,
    hyperparameters: dict | None = None,
) -> None:
    """Entrena y registra un modelo P3 de agrupación de estaciones."""

    # ------------------------------------------------------
    # 1. Configuración de entrenamiento
    # ------------------------------------------------------

    defaults = PROJECT_TRAIN_DEFAULTS["p3"]

    model_name = model_name or defaults["model"]

    resolved_hyperparameters = {
        **defaults["hyperparameters"],
        **(hyperparameters or {}),
    }

    if model_name != defaults["model"]:
        raise ValueError(
            f"Modelo no admitido para P3: {model_name}"
        )

    n_clusters = int(
        resolved_hyperparameters["n_clusters"]
    )
    n_init = int(
        resolved_hyperparameters["n_init"]
    )
    max_iter = int(
        resolved_hyperparameters["max_iter"]
    )

    if n_clusters < 2:
        raise ValueError(
            "n_clusters debe ser al menos 2."
        )

    if n_init < 1:
        raise ValueError(
            "n_init debe ser al menos 1."
        )

    if max_iter < 1:
        raise ValueError(
            "max_iter debe ser al menos 1."
        )

    # ------------------------------------------------------
    # 2. Configuración del servidor MLflow
    # ------------------------------------------------------

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        (PROJ_ROOT / "mlruns").as_uri(),
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ------------------------------------------------------
    # 3. Carga de datos y construcción de perfiles
    # ------------------------------------------------------

    trips, stations, manifest = load_inputs()

    profiles, metadata = build_station_profiles(
        trips,
        stations,
    )

    if len(profiles) <= n_clusters:
        raise ValueError(
            "Se requieren más estaciones que grupos solicitados."
        )

    # Conservamos el mismo orden de características
    # y tipo numérico utilizados en el baseline.
    raw_values = profiles.to_numpy(
        dtype="float32"
    )

    # ------------------------------------------------------
    # 4. Escalamiento y entrenamiento del modelo
    # ------------------------------------------------------

    scaler = StandardScaler()

    values = scaler.fit_transform(raw_values)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=n_init,
        max_iter=max_iter,
    )

    labels = model.fit_predict(values)

    # Se integran los dos componentes YA entrenados.
    # No se ejecuta fit() nuevamente.
    pipeline = Pipeline(
        steps=[
            ("scaler", scaler),
            ("kmeans", model),
        ]
    )

    # ------------------------------------------------------
    # 5. Firma del modelo y ejemplo de entrada
    # ------------------------------------------------------

    input_example = raw_values[:5].copy()

    example_predictions = pipeline.predict(
        input_example
    )

    signature = infer_signature(
        input_example,
        example_predictions,
    )

    # Validación de consistencia del pipeline.
    pipeline_labels = pipeline.predict(
        raw_values
    )

    if not np.array_equal(
        labels,
        pipeline_labels,
    ):
        raise RuntimeError(
            "El pipeline produce etiquetas diferentes "
            "al modelo original."
        )

    # ------------------------------------------------------
    # 6. Evaluación de agrupaciones
    # ------------------------------------------------------

    sample_size = min(
        10_000,
        len(values),
    )

    metrics = {
        "silhouette": float(
            silhouette_score(
                values,
                labels,
                sample_size=sample_size,
                random_state=RANDOM_SEED,
            )
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(
                values,
                labels,
            )
        ),
        "inertia": float(
            model.inertia_
        ),
        "station_count": float(
            len(profiles)
        ),
    }

    for cluster_id in range(n_clusters):
        metrics[f"cluster_{cluster_id}_stations"] = float(
            np.sum(labels == cluster_id)
        )

    # ------------------------------------------------------
    # 7. Parámetros y trazabilidad
    # ------------------------------------------------------

    params = {
        "question": "P3",
        "baseline": f"KMeans_{n_clusters}_clusters",
        "n_clusters": n_clusters,
        "n_init": n_init,
        "max_iter": max_iter,
        "scaler": "StandardScaler",
        "random_seed": RANDOM_SEED,
        "station_count": len(profiles),
        "profile_dimension": profiles.shape[1],
        "profile": "arrivals_and_departures_by_weekday_and_hour",
        "start_month": str(manifest["start_month"]),
        "end_month": str(manifest["end_month"]),
        "data_version": str(manifest["data_version"]),
    }

    # ------------------------------------------------------
    # 8. Información geográfica para visualización
    # ------------------------------------------------------

    map_data = metadata.copy()

    map_data["cluster"] = labels

    # ------------------------------------------------------
    # 9. Registro del experimento en MLflow
    # ------------------------------------------------------

    with mlflow.start_run(
        run_name=f"P3_kmeans_{n_clusters}_station_usage_profiles",
        nested=mlflow.active_run() is not None,
    ):

        mlflow.set_tags(
            {
                "team": "46",
                "question": "P3",
                "model_family": "KMeans",
                "stage_of_work": "candidate",
                "git_commit": git_commit(),
                "data_version": str(
                    manifest["data_version"]
                ),
            }
        )

        mlflow.log_params(params)

        mlflow.log_metrics(metrics)

        # --------------------------------------------------
        # 10. Registro del modelo entrenado
        # --------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        # El modelo contiene el escalador y KMeans.
        # MLflow conserva sus dependencias y firma.

        mlflow.set_tag(
            "logged_model_uri",
            model_info.model_uri,
        )

        # --------------------------------------------------
        # 11. Registro de artefactos de datos
        # --------------------------------------------------

        mlflow.log_artifact(
            str(MANIFEST_PATH),
            artifact_path="data",
        )

        # Guardamos la estructura de las características
        # para reconstruir correctamente la entrada.
        mlflow.log_dict(
            {
                "feature_names": list(
                    profiles.columns
                ),
                "feature_count": int(
                    profiles.shape[1]
                ),
                "input_dtype": "float32",
                "feature_order_required": True,
            },
            "data/feature_schema.json",
        )

        # --------------------------------------------------
        # 12. Configuración de entrenamiento
        # --------------------------------------------------

        mlflow.log_dict(
            {
                "question": "P3",
                "model": "kmeans",
                "hyperparameters": {
                    "n_clusters": n_clusters,
                    "n_init": n_init,
                    "max_iter": max_iter,
                    "random_state": RANDOM_SEED,
                },
                "preprocessing": "StandardScaler",
                "data_version": str(
                    manifest["data_version"]
                ),
                "git_commit": git_commit(),
            },
            "config/training_config.json",
        )

        # --------------------------------------------------
        # 13. Gráfica de estaciones por cluster
        # --------------------------------------------------

        with tempfile.TemporaryDirectory() as temp_dir:

            fig, ax = plt.subplots(
                figsize=(8, 7)
            )

            for cluster_id, group in map_data.groupby(
                "cluster"
            ):

                ax.scatter(
                    group["lon"],
                    group["lat"],
                    s=24,
                    alpha=0.8,
                    label=f"Grupo {cluster_id}",
                )

            ax.set(
                title=(
                    "P3: grupos de estaciones "
                    "por patrón de uso"
                ),
                xlabel="Longitud",
                ylabel="Latitud",
                aspect="equal",
            )

            ax.legend()

            fig.tight_layout()

            artifact_path = (
                Path(temp_dir)
                / "p3_station_clusters_map.png"
            )

            fig.savefig(
                artifact_path,
                dpi=160,
            )

            plt.close(fig)

            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="plots",
            )

        # --------------------------------------------------
        # 14. Resumen final del entrenamiento
        # --------------------------------------------------

        mlflow.log_dict(
            {
                "model": "KMeans",
                "n_clusters": n_clusters,
                "station_count": len(profiles),
                "silhouette": metrics["silhouette"],
                "calinski_harabasz": metrics[
                    "calinski_harabasz"
                ],
                "inertia": metrics["inertia"],
                "cluster_distribution": {
                    str(cluster_id): int(
                        np.sum(labels == cluster_id)
                    )
                    for cluster_id in range(n_clusters)
                },
                "model_uri": model_info.model_uri,
            },
            "reports/p3_training_summary.json",
        )

    logger.success(
        "Run P3 registrado. "
        f"K={n_clusters}, "
        f"Silhouette={metrics['silhouette']:.4f}, "
        f"Calinski-Harabasz="
        f"{metrics['calinski_harabasz']:.4f}"
    )


if __name__ == "__main__":
    main()
