"""Entrena y registra el baseline de P1 en MLflow."""

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
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mibici.config import (
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    PROJECT_TRAIN_DEFAULTS,
    PROJ_ROOT,
    RANDOM_SEED,
)
from mibici.features import make_hourly_panel


def load_inputs() -> tuple[pd.DataFrame, dict]:
    """Carga los viajes de la versión registrada en el manifiesto."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if not manifest.get("complete", False):
        raise ValueError(
            "La ingesta está incompleta. "
            "Ejecuta primero python -m mibici.dataset."
        )

    trip_files = [
        item
        for item in manifest.get("files", [])
        if item.get("name", "").startswith("trips_")
    ]
    if not trip_files:
        raise ValueError("El manifiesto no contiene archivos mensuales de viajes.")

    paths = [PROCESSED_DATA_DIR / item["name"] for item in trip_files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos Parquet: {missing}")

    trips = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    return trips, manifest


def chronological_split(panel: pd.DataFrame):
    """Divide por hora objetivo en entrenamiento, validación y prueba."""
    data = panel.dropna(subset=["target_departures_next_hour"]).copy()
    times = pd.DatetimeIndex(
        data["target_time"].drop_duplicates().sort_values()
    )

    if len(times) < 10:
        raise ValueError("No hay suficientes horas para dividir los datos.")

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
    """Obtiene el commit usado para ejecutar el experimento."""
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


def evaluate(model, frame: pd.DataFrame, split_name: str):
    """Calcula métricas de regresión para una partición."""
    y_true = frame["target_departures_next_hour"].astype(float).to_numpy()
    y_pred = model.predict(np.zeros((len(frame), 1)))

    metrics = {
        f"{split_name}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{split_name}_rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        f"{split_name}_r2": float(r2_score(y_true, y_pred)),
    }
    return metrics, y_true, y_pred


def main(
    model_name: str | None = None,
    hyperparameters: dict | None = None,
) -> None:
    """Entrena P1 usando el modelo e hiperparámetros indicados."""
    defaults = PROJECT_TRAIN_DEFAULTS["p1"]
    model_name = model_name or defaults["model"]
    resolved_hyperparameters = {
        **defaults["hyperparameters"],
        **(hyperparameters or {}),
    }

    if model_name != defaults["model"]:
        raise ValueError(f"Modelo no admitido para P1: {model_name}")

    strategy = resolved_hyperparameters["strategy"]
    if strategy not in {"mean", "median", "quantile"}:
        raise ValueError(
            "La estrategia P1 debe ser mean, median o quantile."
        )

    if strategy == "quantile":
        quantile = float(resolved_hyperparameters.get("quantile", 0.5))
        if not 0 <= quantile <= 1:
            raise ValueError("quantile debe estar entre 0 y 1.")
    elif "quantile" in resolved_hyperparameters:
        raise ValueError(
            "quantile solo aplica cuando strategy=quantile."
        )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        (PROJ_ROOT / "mlruns").as_uri(),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    trips, manifest = load_inputs()
    panel = make_hourly_panel(trips)
    train, validation, test, bounds = chronological_split(panel)

    y_train = train["target_departures_next_hour"].astype(float).to_numpy()

    model_kwargs = {"strategy": strategy}
    if strategy == "quantile":
        model_kwargs["quantile"] = quantile

    model = DummyRegressor(**model_kwargs)
    model.fit(np.zeros((len(y_train), 1)), y_train)

    metrics = {}
    predictions = {}
    for split_name, frame in [
        ("validation", validation),
        ("test", test),
    ]:
        split_metrics, y_true, y_pred = evaluate(model, frame, split_name)
        metrics.update(split_metrics)
        if split_name == "test":
            predictions = (y_true, y_pred)

    params = {
        "question": "P1",
        "baseline": f"DummyRegressor_{strategy}",
        "strategy": strategy,
        "granularity": "station_hour",
        "horizon": "next_hour",
        "target": "outbound_trip_count_next_hour",
        "split_strategy": "chronological_70_15_15_by_target_time",
        "start_month": str(manifest["start_month"]),
        "end_month": str(manifest["end_month"]),
        "data_version": str(manifest["data_version"]),
        "random_seed": RANDOM_SEED,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        **bounds,
    }
    if strategy == "quantile":
        params["quantile"] = quantile

    with mlflow.start_run(run_name=f"P1_dummy_{strategy}_hourly_demand"):
        mlflow.set_tags(
            {
                "team": "46",
                "question": "P1",
                "git_commit": git_commit(),
                "data_version": str(manifest["data_version"]),
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            y_true, y_pred = predictions
            sample = np.linspace(
                0,
                len(y_true) - 1,
                min(3000, len(y_true)),
                dtype=int,
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_true[sample], y_pred[sample], s=8, alpha=0.25)
            ax.set(
                title="P1: baseline de demanda en prueba",
                xlabel="Salidas observadas en la hora siguiente",
                ylabel="Predicción del baseline",
            )
            fig.tight_layout()

            artifact_path = (
                Path(temp_dir) / "p1_test_actual_vs_predicted.png"
            )
            fig.savefig(artifact_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(artifact_path), artifact_path="evidence")

    logger.success(
        f"Run P1 registrado. MAE de prueba: {metrics['test_mae']:.4f}"
    )


if __name__ == "__main__":
    main()