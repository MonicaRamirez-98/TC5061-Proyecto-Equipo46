"""Entrena una versión alternativa de P1 y la registra como challenger.

Uso, desde la raíz del repositorio con MLFLOW_TRACKING_URI configurada:
    python -m mibici.modeling.retrain_p1

Entrena un HistGradientBoostingRegressor con hiperparámetros distintos a
los de P1-HGB-01/02/03, con los mismos datos, variables y partición
cronológica que el modelo champion, para que ambas versiones sean
comparables. Registra el modelo como una nueva versión de
mibici-equipo46-p1-demanda con el alias challenger. El alias champion
no se modifica.
"""

from __future__ import annotations

import argparse
import os

import mlflow
import mlflow.sklearn
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici.config import MANIFEST_PATH, MLFLOW_EXPERIMENT_NAME, RANDOM_SEED
from mibici.features import make_hourly_panel
from mibici.modeling.experiment_p1 import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate,
    get_git_commit,
    prepare_xy,
)
from mibici.modeling.train import chronological_split, load_inputs

REGISTERED_NAME = "mibici-equipo46-p1-demanda"
CHAMPION_ALIAS = "champion"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="P1-HGB-04")
    parser.add_argument("--alias", default="challenger")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    hyperparameters = {
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_samples_leaf": args.min_samples_leaf,
        "l2_regularization": args.l2_regularization,
    }

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    # El champion debe existir antes de crear un challenger.
    champion = client.get_model_version_by_alias(REGISTERED_NAME, CHAMPION_ALIAS)
    logger.info(f"Champion actual: {REGISTERED_NAME} versión {champion.version}")

    logger.info("Cargando datos P1...")
    trips, manifest = load_inputs()
    data_version = str(manifest["data_version"])
    champion_data_version = (champion.tags or {}).get("data_version")
    if champion_data_version and champion_data_version != data_version:
        raise SystemExit(
            f"La versión de datos local ({data_version}) no coincide con la del "
            f"champion ({champion_data_version}). Ejecuta antes la ingesta para "
            "entrenar con los mismos datos."
        )

    panel = make_hourly_panel(trips)
    train, validation, test, bounds = chronological_split(panel)
    del panel, trips

    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(validation)
    X_test, y_test = prepare_xy(test)
    del train, validation, test

    commit = get_git_commit()

    with mlflow.start_run(run_name=args.candidate) as run:
        logger.info(f"Entrenando {args.candidate} con {hyperparameters}")
        model = HistGradientBoostingRegressor(
            **hyperparameters,
            random_state=RANDOM_SEED,
            early_stopping=False,
        )
        model.fit(X_train, y_train)

        val_metrics, _ = evaluate(model, X_val, y_val, "validation")
        test_metrics, _ = evaluate(model, X_test, y_test, "test")
        metrics = {**val_metrics, **test_metrics}

        mlflow.set_tags({
            "team": "46",
            "question": "P1",
            "candidate": args.candidate,
            "model_family": "HistGradientBoostingRegressor",
            "stage_of_work": "challenger",
            "data_version": data_version,
            "git_commit": commit,
            "compared_with_version": str(champion.version),
        })
        mlflow.log_params({
            **hyperparameters,
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
        signature = infer_signature(input_example, model.predict(input_example))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        mlflow.set_tag("logged_model_uri", model_info.model_uri)
        mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")
        mlflow.log_dict(
            {
                "candidate": args.candidate,
                "algorithm": "HistGradientBoostingRegressor",
                "hyperparameters": hyperparameters,
                "features": FEATURE_COLUMNS,
                "target": TARGET_COLUMN,
                "data_version": data_version,
                "split_bounds": bounds,
                "compared_with_version": str(champion.version),
            },
            "config/training_config.json",
        )
        run_id = run.info.run_id

    # Registrar como nueva versión del mismo modelo, con alias challenger.
    registered = mlflow.register_model(model_uri=model_info.model_uri, name=REGISTERED_NAME)
    number = str(registered.version)
    for key, value in {
        "team": "46",
        "question": "P1",
        "data_version": data_version,
        "source_run_id": run_id,
        "source_model_uri": model_info.model_uri,
        "selected_run_name": args.candidate,
        "validation_mae": f"{metrics['validation_mae']:.6f}",
        "validation_status": "pending_ab_test",
        "environment": "academic",
    }.items():
        client.set_model_version_tag(REGISTERED_NAME, number, key, str(value))
    client.set_registered_model_alias(REGISTERED_NAME, args.alias, number)

    # Comprobar que ambos alias resuelven y que el champion no cambió.
    resolved_champion = client.get_model_version_by_alias(REGISTERED_NAME, CHAMPION_ALIAS)
    resolved_challenger = client.get_model_version_by_alias(REGISTERED_NAME, args.alias)
    if str(resolved_champion.version) != str(champion.version):
        raise RuntimeError("El alias champion cambió durante el registro.")
    if str(resolved_challenger.version) != number:
        raise RuntimeError("El alias challenger no apunta a la versión recién registrada.")
    mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}")
    mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@{args.alias}")

    champion_run = client.get_run(champion.run_id) if champion.run_id else None
    champion_metrics = champion_run.data.metrics if champion_run else {}

    print("")
    print(f"Datos: versión {data_version}")
    print(f"{'Métrica':<16}{'champion v' + str(champion.version):>18}{args.alias + ' v' + number:>20}")
    for key in ("validation_mae", "validation_rmse", "validation_r2",
                "test_mae", "test_rmse", "test_r2"):
        old = champion_metrics.get(key)
        old_text = f"{old:.6f}" if old is not None else "-"
        print(f"{key:<16}{old_text:>18}{metrics[key]:>20.6f}")
    print("")
    print(f"OK: {REGISTERED_NAME} | champion=v{champion.version} | {args.alias}=v{number}")
    print("Ambas versiones se recuperan desde MLflow Model Registry.")


if __name__ == "__main__":
    main()
