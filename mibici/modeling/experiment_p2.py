
"""Experimentación avanzada P2 - Equipo 46.

Predice presión operacional en estaciones MiBici mediante
HistGradientBoostingClassifier.

Incluye:
- Tres modelos candidatos.
- Partición cronológica idéntica al baseline.
- Búsqueda organizada mediante nested runs.
- Métricas de clasificación.
- Matrices de confusión y curvas Precision-Recall.
- Modelos MLflow con firma y ejemplo de entrada.
- Manifiesto, configuración y trazabilidad.
"""

from __future__ import annotations

import os
import subprocess

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from loguru import logger
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)

from mibici.config import (
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    PROJ_ROOT,
    RANDOM_SEED,
)

from mibici.features import make_hourly_panel
from mibici.modeling.train_p2 import (
    load_inputs,
    chronological_split,
)


# ---------------------------------------------------------
# 1. Variables predictoras y objetivo
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "station_id",
    "capacity",
    "departures_this_hour",
    "arrivals_this_hour",
    "departures_mean_last_24h",
    "departures_mean_last_168h",
    "target_hour",
    "target_dayofweek",
    "target_is_weekend",
]

TARGET_COLUMN = "target_risk_next_hour"

RISK_FRACTION = 0.10


# ---------------------------------------------------------
# 2. Configuraciones de los candidatos
# ---------------------------------------------------------

CONFIGURATIONS = [
    {
        "candidate": "P2-HGB-01",
        "max_iter": 60,
        "learning_rate": 0.10,
        "max_leaf_nodes": 15,
    },
    {
        "candidate": "P2-HGB-02",
        "max_iter": 100,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
    },
    {
        "candidate": "P2-HGB-03",
        "max_iter": 150,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
    },
]


# ---------------------------------------------------------
# 3. Funciones auxiliares
# ---------------------------------------------------------

def git_commit() -> str:
    """Recupera el commit actual de Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            text=True,
        ).strip()

    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_xy(frame: pd.DataFrame):
    """Construye X e y sin utilizar objetivos futuros."""

    missing = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias: {missing}"
        )

    X = frame[FEATURE_COLUMNS].astype("float32")

    y = frame[TARGET_COLUMN].astype(
        "int8"
    ).to_numpy()

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Las características contienen valores inválidos."
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            "El objetivo debe contener únicamente 0 y 1."
        )

    return X, y


def evaluate(model, X, y, prefix):
    """Calcula métricas de clasificación binaria."""

    y_pred = model.predict(X)

    metrics = {
        f"{prefix}_accuracy": float(
            accuracy_score(y, y_pred)
        ),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(y, y_pred)
        ),
        f"{prefix}_precision": float(
            precision_score(y, y_pred, zero_division=0)
        ),
        f"{prefix}_recall": float(
            recall_score(y, y_pred, zero_division=0)
        ),
        f"{prefix}_f1": float(
            f1_score(y, y_pred, zero_division=0)
        ),
    }

    return metrics, y_pred


def log_plots(model, X_test, y_test, y_pred):
    """Registra matriz de confusión y curva Precision-Recall."""

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Sin presión", "Presión"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )

    ax.set_title(
        "P2 - Matriz de confusión en prueba"
    )

    fig.tight_layout()

    mlflow.log_figure(
        fig,
        "plots/confusion_matrix.png",
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
    )

    ax.set_title(
        "P2 - Curva Precision-Recall en prueba"
    )

    fig.tight_layout()

    mlflow.log_figure(
        fig,
        "plots/precision_recall_curve.png",
    )

    plt.close(fig)


# ---------------------------------------------------------
# 4. Entrenamiento principal
# ---------------------------------------------------------

def main():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    # -----------------------------------------------------
    # 5. Carga de datos y partición
    # -----------------------------------------------------

    logger.info("Cargando información de P2")

    trips, stations, manifest = load_inputs()

    panel = make_hourly_panel(
        trips,
        stations=stations,
        risk_fraction=RISK_FRACTION,
    )

    train, validation, test, bounds = (
        chronological_split(panel)
    )

    del trips, stations, panel

    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(validation)
    X_test, y_test = prepare_xy(test)

    del train, validation, test

    data_version = str(
        manifest["data_version"]
    )

    commit = git_commit()

    risk_rates = {
        "training_risk_rate": float(np.mean(y_train)),
        "validation_risk_rate": float(np.mean(y_val)),
        "test_risk_rate": float(np.mean(y_test)),
    }

    logger.info(
        f"Distribución de riesgo: {risk_rates}"
    )

    if np.unique(y_train).size < 2:
        raise ValueError(
            "El conjunto de entrenamiento no contiene ambas clases."
        )

    if np.unique(y_val).size < 2:
        raise ValueError(
            "La validación no contiene ambas clases."
        )

    if np.unique(y_test).size < 2:
        raise ValueError(
            "La prueba no contiene ambas clases."
        )

    results = []

    # -----------------------------------------------------
    # 6. Corrida padre
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name="P2_busqueda_HistGradientBoosting",
        nested=mlflow.active_run() is not None,
    ):

        mlflow.set_tags({
            "team": "46",
            "question": "P2",
            "search_type": "grid_search",
            "stage_of_work": "candidate",
            "data_version": data_version,
            "git_commit": commit,
        })

        mlflow.log_params({
            "question": "P2",
            "search_algorithm": "HistGradientBoostingClassifier",
            "candidate_count": len(CONFIGURATIONS),
            "risk_fraction": RISK_FRACTION,
            "selection_metric": "validation_f1",
            "split_strategy": "chronological_70_15_15_by_target_time",
            "data_version": data_version,
        })

        mlflow.log_metrics(risk_rates)

        mlflow.log_artifact(
            str(MANIFEST_PATH),
            artifact_path="data",
        )

        mlflow.log_dict(
            {
                "features": FEATURE_COLUMNS,
                "target": TARGET_COLUMN,
                "risk_fraction": RISK_FRACTION,
                "split_bounds": bounds,
                "configurations": CONFIGURATIONS,
                "data_version": data_version,
            },
            "config/search_definition.json",
        )

        # -------------------------------------------------
        # 7. Corridas hijas con entrenamiento real
        # -------------------------------------------------

        for config in CONFIGURATIONS:

            candidate = config["candidate"]

            logger.info(
                f"Entrenando candidato {candidate}"
            )

            with mlflow.start_run(
                run_name=candidate,
                nested=True,
            ) as child_run:

                model = HistGradientBoostingClassifier(
                    max_iter=config["max_iter"],
                    learning_rate=config["learning_rate"],
                    max_leaf_nodes=config["max_leaf_nodes"],
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    early_stopping=False,
                )

                model.fit(
                    X_train,
                    y_train,
                )

                val_metrics, _ = evaluate(
                    model,
                    X_val,
                    y_val,
                    "validation",
                )

                test_metrics, test_pred = evaluate(
                    model,
                    X_test,
                    y_test,
                    "test",
                )

                metrics = {
                    **val_metrics,
                    **test_metrics,
                }

                mlflow.set_tags({
                    "team": "46",
                    "question": "P2",
                    "candidate": candidate,
                    "model_family": "HistGradientBoostingClassifier",
                    "stage_of_work": "candidate",
                    "data_version": data_version,
                    "git_commit": commit,
                })

                mlflow.log_params({
                    **config,
                    "question": "P2",
                    "model": "hist_gradient_boosting_classifier",
                    "class_weight": "balanced",
                    "risk_fraction": RISK_FRACTION,
                    "random_seed": RANDOM_SEED,
                    "training_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "test_rows": len(X_test),
                    "data_version": data_version,
                    **bounds,
                })

                mlflow.log_metrics(metrics)

                mlflow.log_artifact(
                    str(MANIFEST_PATH),
                    artifact_path="data",
                )

                # -----------------------------------------
                # 8. Registro de modelo con firma
                # -----------------------------------------

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

                # -----------------------------------------
                # 9. Configuración y gráficas
                # -----------------------------------------

                mlflow.log_dict(
                    {
                        "candidate": candidate,
                        "algorithm": "HistGradientBoostingClassifier",
                        "hyperparameters": config,
                        "class_weight": "balanced",
                        "risk_fraction": RISK_FRACTION,
                        "feature_columns": FEATURE_COLUMNS,
                        "target_column": TARGET_COLUMN,
                        "data_version": data_version,
                        "split_bounds": bounds,
                        "risk_rates": risk_rates,
                    },
                    "config/training_config.json",
                )

                log_plots(
                    model,
                    X_test,
                    y_test,
                    test_pred,
                )

                results.append({
                    "candidate": candidate,
                    "run_id": child_run.info.run_id,
                    "model_uri": model_info.model_uri,
                    **metrics,
                })

                logger.success(
                    f"{candidate}: "
                    f"F1 validación="
                    f"{metrics['validation_f1']:.4f}, "
                    f"Recall validación="
                    f"{metrics['validation_recall']:.4f}"
                )

        # -------------------------------------------------
        # 10. Selección del mejor candidato
        # -------------------------------------------------

        best = max(
            results,
            key=lambda result: (
                result["validation_f1"],
                result["validation_recall"],
                result["validation_balanced_accuracy"],
            ),
        )

        mlflow.log_metrics({
            "best_validation_f1": best["validation_f1"],
            "best_validation_recall": best["validation_recall"],
            "best_validation_balanced_accuracy": best[
                "validation_balanced_accuracy"
            ],
        })

        mlflow.set_tags({
            "best_candidate": best["candidate"],
            "best_run_id": best["run_id"],
        })

        mlflow.log_dict(
            {
                "selection_metric": "validation_f1",
                "best_candidate": best,
                "all_results": results,
                "risk_fraction": RISK_FRACTION,
            },
            "reports/p2_comparison.json",
        )

        logger.success(
            f"Mejor candidato P2: {best['candidate']} "
            f"F1 validación={best['validation_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
