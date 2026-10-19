"""Entrena y registra el baseline de P2 en MLflow."""

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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from mibici.config import (
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RANDOM_SEED,
)
from mibici.features import make_hourly_panel


RISK_FRACTION = 0.10


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


def chronological_split(panel: pd.DataFrame):
    """Divide las observaciones por hora objetivo en orden cronológico."""
    data = panel.dropna(subset=["target_risk_next_hour"]).copy()
    times = pd.DatetimeIndex(data["target_time"].drop_duplicates().sort_values())

    if len(times) < 10:
        raise ValueError("No hay suficientes horas para la partición temporal.")

    train_end = times[int(len(times) * 0.70)]
    validation_end = times[int(len(times) * 0.85)]

    train = data[data["target_time"] < train_end]
    validation = data[
        (data["target_time"] >= train_end)
        & (data["target_time"] < validation_end)
    ]
    test = data[data["target_time"] >= validation_end]

    if train.empty or validation.empty or test.empty:
        raise ValueError("Una de las particiones quedó vacía.")

    bounds = {
        "train_target_start": train["target_time"].min().isoformat(),
        "train_target_end": train["target_time"].max().isoformat(),
        "validation_target_start": validation["target_time"].min().isoformat(),
        "validation_target_end": validation["target_time"].max().isoformat(),
        "test_target_start": test["target_time"].min().isoformat(),
        "test_target_end": test["target_time"].max().isoformat(),
    }
    return train, validation, test, bounds


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


def evaluate(model, frame: pd.DataFrame, split_name: str) -> dict:
    """Calcula métricas de clasificación para una partición."""
    y_true = frame["target_risk_next_hour"].astype("int8").to_numpy()
    y_pred = model.predict(np.zeros((len(frame), 1)))

    return {
        f"{split_name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{split_name}_balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        f"{split_name}_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        f"{split_name}_recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        f"{split_name}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        (PROJ_ROOT / "mlruns").as_uri(),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    trips, stations, manifest = load_inputs()
    panel = make_hourly_panel(
        trips,
        stations=stations,
        risk_fraction=RISK_FRACTION,
    )
    train, validation, test, bounds = chronological_split(panel)

    y_train = train["target_risk_next_hour"].astype("int8").to_numpy()
    model = DummyClassifier(
        strategy="most_frequent",
        random_state=RANDOM_SEED,
    )
    model.fit(np.zeros((len(y_train), 1)), y_train)

    metrics = {}
    for split_name, frame in [
        ("validation", validation),
        ("test", test),
    ]:
        metrics.update(evaluate(model, frame, split_name))

    y_test = test["target_risk_next_hour"].astype("int8").to_numpy()
    y_test_pred = model.predict(np.zeros((len(test), 1)))
    matrix = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    params = {
        "question": "P2",
        "baseline": "DummyClassifier_most_frequent",
        "granularity": "station_hour",
        "horizon": "next_hour",
        "target": "absolute_net_flow_at_least_10pct_capacity",
        "risk_fraction_of_capacity": RISK_FRACTION,
        "split_strategy": "chronological_70_15_15_by_target_time",
        "start_month": str(manifest["start_month"]),
        "end_month": str(manifest["end_month"]),
        "data_version": str(manifest["data_version"]),
        "random_seed": RANDOM_SEED,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "training_risk_rate": float(np.mean(y_train)),
        **bounds,
    }

    with mlflow.start_run(run_name="P2_dummy_majority_capacity_pressure"):
        mlflow.set_tags({
            "team": "46",
            "question": "P2",
            "git_commit": git_commit(),
            "data_version": str(manifest["data_version"]),
        })
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            fig, ax = plt.subplots(figsize=(5, 4))
            image = ax.imshow(matrix, cmap="Blues")
            ax.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=["Sin presión", "Presión"],
                yticklabels=["Sin presión", "Presión"],
                xlabel="Predicción",
                ylabel="Observado",
                title="P2: matriz de confusión en prueba",
            )

            for (row, column), value in np.ndenumerate(matrix):
                ax.text(column, row, f"{value:,}", ha="center", va="center")

            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            artifact_path = Path(temp_dir) / "p2_test_confusion_matrix.png"
            fig.savefig(artifact_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(artifact_path), artifact_path="evidence")

    logger.success(
        f"Run P2 registrado. F1 de prueba: {metrics['test_f1']:.4f}"
    )


if __name__ == "__main__":
    main()