"""Agrupa estaciones por patrón de uso y registra P3 en MLflow."""

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
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RANDOM_SEED,
)
from mibici.features import build_station_profiles


N_CLUSTERS = 3


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Carga viajes, estaciones y metadatos de la versión del manifiesto."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if not manifest.get("complete", False):
        raise ValueError(
            "La ingesta está incompleta. Ejecuta primero "
            "python -m mibici.dataset."
        )

    trip_files = [
        item for item in manifest.get("files", [])
        if item.get("name", "").startswith("trips_")
    ]
    if not trip_files:
        raise ValueError("El manifiesto no contiene archivos mensuales de viajes.")

    trip_paths = [PROCESSED_DATA_DIR / item["name"] for item in trip_files]
    stations_path = PROCESSED_DATA_DIR / "stations.parquet"
    missing = [str(path) for path in trip_paths if not path.exists()]

    if missing or not stations_path.exists():
        raise FileNotFoundError(
            "Faltan Parquet de viajes o estaciones. "
            "Ejecuta python -m mibici.dataset."
        )

    trips = pd.concat(
        [pd.read_parquet(path) for path in trip_paths],
        ignore_index=True,
    )
    stations = pd.read_parquet(stations_path)
    return trips, stations, manifest


def git_commit() -> str:
    """Obtiene el commit con el que se ejecuta el run."""
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


def main() -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        (PROJ_ROOT / "mlruns").as_uri(),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    trips, stations, manifest = load_inputs()
    profiles, metadata = build_station_profiles(trips, stations)

    if len(profiles) <= N_CLUSTERS:
        raise ValueError("Se requieren más estaciones que grupos.")

    scaler = StandardScaler()
    values = scaler.fit_transform(profiles.to_numpy(dtype="float32"))

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=10,
    )
    labels = model.fit_predict(values)

    sample_size = min(10_000, len(values))
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
            calinski_harabasz_score(values, labels)
        ),
        "inertia": float(model.inertia_),
        "station_count": float(len(profiles)),
    }

    for cluster_id in range(N_CLUSTERS):
        metrics[f"cluster_{cluster_id}_stations"] = float(
            np.sum(labels == cluster_id)
        )

    params = {
        "question": "P3",
        "baseline": "KMeans_station_usage_profiles",
        "profile": "arrivals_and_departures_by_weekday_and_hour",
        "n_clusters": N_CLUSTERS,
        "scaler": "StandardScaler",
        "random_seed": RANDOM_SEED,
        "station_count": len(profiles),
        "profile_dimension": profiles.shape[1],
        "start_month": str(manifest["start_month"]),
        "end_month": str(manifest["end_month"]),
        "data_version": str(manifest["data_version"]),
    }

    map_data = metadata.copy()
    map_data["cluster"] = labels

    with mlflow.start_run(run_name="P3_kmeans_station_usage_profiles"):
        mlflow.set_tags({
            "team": "46",
            "question": "P3",
            "git_commit": git_commit(),
            "data_version": str(manifest["data_version"]),
        })
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            fig, ax = plt.subplots(figsize=(8, 7))

            for cluster_id, group in map_data.groupby("cluster"):
                ax.scatter(
                    group["lon"],
                    group["lat"],
                    s=24,
                    alpha=0.8,
                    label=f"Grupo {cluster_id}",
                )

            ax.set(
                title="P3: grupos de estaciones por patrón de uso",
                xlabel="Longitud",
                ylabel="Latitud",
                aspect="equal",
            )
            ax.legend()
            fig.tight_layout()

            artifact_path = Path(temp_dir) / "p3_station_clusters_map.png"
            fig.savefig(artifact_path, dpi=160)
            plt.close(fig)
            mlflow.log_artifact(str(artifact_path), artifact_path="evidence")

    logger.success(
        f"Run P3 registrado. Silhouette: {metrics['silhouette']:.4f}"
    )


if __name__ == "__main__":
    main()