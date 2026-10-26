
"""Experimentación P1: predicción de demanda con MLflow.

Ejecuta tres configuraciones de HistGradientBoostingRegressor,
conserva la partición cronológica del baseline y registra
modelos, métricas, firmas, artefactos y nested runs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from loguru import logger
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mibici.config import (
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    PROJ_ROOT,
    RANDOM_SEED,
)
from mibici.features import make_hourly_panel
from mibici.modeling.train import load_inputs, chronological_split


FEATURE_COLUMNS = [
    "station_id",
    "departures_this_hour",
    "arrivals_this_hour",
    "departures_mean_last_24h",
    "departures_mean_last_168h",
    "target_hour",
    "target_dayofweek",
    "target_is_weekend",
]

TARGET_COLUMN = "target_departures_next_hour"

CONFIGURATIONS = [
    {
        "candidate": "P1-HGB-01",
        "max_iter": 60,
        "learning_rate": 0.10,
        "max_leaf_nodes": 15,
    },
    {
        "candidate": "P1-HGB-02",
        "max_iter": 100,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
    },
    {
        "candidate": "P1-HGB-03",
        "max_iter": 150,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
    },
]


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_xy(frame):
    """Prepara las mismas columnas para todas las corridas."""

    X = frame[FEATURE_COLUMNS].astype("float32")
    y = frame[TARGET_COLUMN].astype("float32").to_numpy()

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Las características contienen NaN o infinitos.")

    if not np.isfinite(y).all():
        raise ValueError("El objetivo contiene NaN o infinitos.")

    return X, y


def evaluate(model, X, y, prefix):
    predictions = np.maximum(
        model.predict(X),
        0.0,
    )

    metrics = {
        f"{prefix}_mae": float(
            mean_absolute_error(y, predictions)
        ),
        f"{prefix}_rmse": float(
            np.sqrt(mean_squared_error(y, predictions))
        ),
        f"{prefix}_r2": float(
            r2_score(y, predictions)
        ),
    }

    return metrics, predictions


def main():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Cargando datos P1...")
    trips, manifest = load_inputs()

    panel = make_hourly_panel(trips)

    train, validation, test, bounds = chronological_split(panel)

    # Liberamos objetos grandes que ya no necesitamos.
    del panel, trips

    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(validation)
    X_test, y_test = prepare_xy(test)

    del train, validation, test

    data_version = str(manifest["data_version"])
    commit = get_git_commit()

    results = []

    logger.info("Iniciando búsqueda de hiperparámetros P1")

    # Si MLflow Project abrió una corrida, anidamos la búsqueda.
    with mlflow.start_run(
        run_name="P1_busqueda_HistGradientBoosting",
        nested=mlflow.active_run() is not None,
    ) as parent_run:

        mlflow.set_tags({
            "team": "46",
            "question": "P1",
            "search_type": "grid_search",
            "stage_of_work": "candidate",
            "data_version": data_version,
            "git_commit": commit,
        })

        mlflow.log_params({
            "question": "P1",
            "search_algorithm": "HistGradientBoostingRegressor",
            "candidate_count": len(CONFIGURATIONS),
            "data_version": data_version,
            "split_strategy": "chronological_70_15_15_by_target_time",
            "selection_metric": "validation_mae",
        })

        mlflow.log_dict(
            {
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "split_bounds": bounds,
                "data_version": data_version,
            },
            "config/search_definition.json",
        )

        mlflow.log_artifact(
            str(MANIFEST_PATH),
            artifact_path="data",
        )

        for config in CONFIGURATIONS:
            candidate = config["candidate"]

            logger.info(f"Entrenando {candidate}")

            with mlflow.start_run(
                run_name=candidate,
                nested=True,
            ) as child_run:

                model = HistGradientBoostingRegressor(
                    max_iter=config["max_iter"],
                    learning_rate=config["learning_rate"],
                    max_leaf_nodes=config["max_leaf_nodes"],
                    random_state=RANDOM_SEED,
                    early_stopping=False,
                )

                model.fit(X_train, y_train)

                val_metrics, val_pred = evaluate(
                    model, X_val, y_val, "validation"
                )

                test_metrics, test_pred = evaluate(
                    model, X_test, y_test, "test"
                )

                metrics = {
                    **val_metrics,
                    **test_metrics,
                }

                mlflow.set_tags({
                    "team": "46",
                    "question": "P1",
                    "candidate": candidate,
                    "model_family": "HistGradientBoostingRegressor",
                    "data_version": data_version,
                    "git_commit": commit,
                })

                mlflow.log_params({
                    **config,
                    "question": "P1",
                    "model": "hist_gradient_boosting",
                    "random_seed": RANDOM_SEED,
                    "training_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "test_rows": len(X_test),
                    "data_version": data_version,
                    **bounds,
                })

                mlflow.log_metrics(metrics)

                input_example = X_train.head(5).copy()

                signature = infer_signature(
                    input_example,
                    model.predict(input_example),
                )

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                )

                mlflow.set_tag(
                    "logged_model_uri",
                    model_info.model_uri,
                )

                mlflow.log_artifact(
                    str(MANIFEST_PATH),
                    artifact_path="data",
                )

                mlflow.log_dict(
                    {
                        "candidate": candidate,
                        "algorithm": "HistGradientBoostingRegressor",
                        "hyperparameters": config,
                        "features": FEATURE_COLUMNS,
                        "target": TARGET_COLUMN,
                        "data_version": data_version,
                        "selection_metric": "validation_mae",
                        "split_bounds": bounds,
                    },
                    "config/training_config.json",
                )

                # Gráfica con una muestra reproducible.
                sample_indices = np.linspace(
                    0,
                    len(y_test) - 1,
                    min(3000, len(y_test)),
                    dtype=int,
                )

                fig, ax = plt.subplots(figsize=(6, 5))

                ax.scatter(
                    y_test[sample_indices],
                    test_pred[sample_indices],
                    s=8,
                    alpha=0.25,
                )

                ax.set(
                    title=f"P1: {candidate}",
                    xlabel="Demanda observada",
                    ylabel="Demanda predicha",
                )

                fig.tight_layout()

                mlflow.log_figure(
                    fig,
                    "plots/actual_vs_predicted.png",
                )

                plt.close(fig)

                results.append({
                    "candidate": candidate,
                    "run_id": child_run.info.run_id,
                    "model_uri": model_info.model_uri,
                    **metrics,
                })

                logger.success(
                    f"{candidate}: "
                    f"MAE validación={metrics['validation_mae']:.4f}"
                )

        # Seleccionamos mediante VALIDACIÓN, no mediante prueba.
        best = min(
            results,
            key=lambda item: item["validation_mae"],
        )

        mlflow.log_metrics({
            "best_validation_mae": best["validation_mae"],
            "best_validation_rmse": best["validation_rmse"],
            "best_test_mae": best["test_mae"],
        })

        mlflow.set_tags({
            "best_candidate": best["candidate"],
            "best_run_id": best["run_id"],
        })

        mlflow.log_dict(
            {
                "selection_metric": "validation_mae",
                "best_candidate": best,
                "all_results": results,
            },
            "reports/p1_comparison.json",
        )

        logger.success(
            f"Mejor candidato: {best['candidate']} "
            f"MAE validación={best['validation_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
