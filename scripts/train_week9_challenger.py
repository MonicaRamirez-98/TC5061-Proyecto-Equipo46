"""Reentrena P1 hasta mitad de octubre y registra el challenger en MLflow."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mibici.features import make_hourly_panel
from mibici.serving.contract import FEATURE_COLUMNS
from scripts.prepare_week9_datasets import load_config, load_trips


CONFIG_PATH = ROOT / "configs" / "week9_abtest.json"
MANIFEST_PATH = ROOT / "data" / "manifest" / "data_manifest.json"
REGISTRATION_PATH = ROOT / "reports" / "semana9" / "challenger_registration.json"
TARGET_COLUMN = "target_departures_next_hour"
EXPERIMENT_NAME = "mibici-equipo46"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    run_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset_config = load_config()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    model_name = run_config["model_name"]
    champion_version = str(run_config["champion_version"])
    champion = client.get_model_version(model_name, champion_version)
    champion_alias = client.get_model_version_by_alias(model_name, "champion")
    if str(champion_alias.version) != champion_version:
        raise RuntimeError(
            "El alias champion cambió desde la versión base esperada; "
            "se cancela para no evaluar contra un baseline distinto."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    supplemental_path = ROOT / dataset_config["inputs"]["supplemental_current_file"]
    supplemental_hash = sha256_file(supplemental_path)
    training_data_version = hashlib.sha256(
        "|".join(
            [
                manifest["data_version"],
                supplemental_hash,
                run_config["training_target_time_end_exclusive"],
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    # Si la misma mezcla y fecha de corte ya se registraron, recuperar la
    # versión existente en vez de entrenar y registrar un duplicado.
    existing = next(
        (
            item
            for item in client.search_model_versions(f"name='{model_name}'")
            if (item.tags or {}).get("training_data_version") == training_data_version
        ),
        None,
    )
    if existing is not None:
        existing_run = client.get_run(existing.run_id)
        if existing_run.info.status == "RUNNING":
            client.set_terminated(existing.run_id, status="FINISHED")
        if existing.status != "READY":
            raise RuntimeError(
                f"La versión existente {existing.version} no está lista: {existing.status}."
            )
        current_alias = client.get_model_version_by_alias(model_name, "champion")
        if str(current_alias.version) != champion_version:
            raise RuntimeError("El alias champion ya no apunta al baseline v1.")
        loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{existing.version}")
        params = existing_run.data.params
        registration = {
            "model_name": model_name,
            "champion_version": champion_version,
            "challenger_version": str(existing.version),
            "challenger_alias": None,
            "source_run_id": existing.run_id,
            "source_model_uri": existing.source,
            "registry_model_uri": f"models:/{model_name}/{existing.version}",
            "candidate": (existing.tags or {}).get("candidate", run_config["candidate"]),
            "training_rows": int(params["training_rows"]),
            "training_target_start": params["training_target_start"],
            "training_target_end_inclusive": params["training_target_end_inclusive"],
            "training_target_end_exclusive": params["training_target_end_exclusive"],
            "evaluation_holdout": {
                "start_inclusive": run_config[
                    "evaluation_target_time_start_inclusive"
                ],
                "end_exclusive": run_config[
                    "evaluation_target_time_end_exclusive"
                ],
            },
            "training_data_version": training_data_version,
            "model_ready": loaded is not None and existing.status == "READY",
            "champion_alias_still_points_to": str(current_alias.version),
        }
        REGISTRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRATION_PATH.write_text(
            json.dumps(registration, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(registration, ensure_ascii=False, indent=2))
        return

    print("Cargando viajes históricos y el CSV suplementario de octubre...", flush=True)
    trips, supplemental = load_trips(dataset_config)
    if supplemental["sha256"] != supplemental_hash:
        raise RuntimeError("El CSV de octubre cambió durante la preparación del challenger.")
    panel = make_hourly_panel(trips)
    del trips

    panel = panel.dropna(subset=[TARGET_COLUMN]).copy()
    panel["target_time"] = pd.to_datetime(panel["target_time"])
    train_start = pd.Timestamp(run_config["training_target_time_start_inclusive"])
    train_end = pd.Timestamp(run_config["training_target_time_end_exclusive"])
    train = panel.loc[
        (panel["target_time"] >= train_start)
        & (panel["target_time"] < train_end)
    ]
    if train.empty:
        raise ValueError("El conjunto de entrenamiento del challenger quedó vacío.")

    X_train = train[list(FEATURE_COLUMNS)].astype("float32")
    y_train = train[TARGET_COLUMN].astype("float32").to_numpy()
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(y_train).all():
        raise ValueError("El entrenamiento contiene NaN o valores infinitos.")

    train_start_actual = train["target_time"].min().isoformat()
    train_end_actual = train["target_time"].max().isoformat()
    training_rows = len(train)
    del panel, train

    parameters = run_config["hyperparameters"]
    model = HistGradientBoostingRegressor(**parameters)
    print(
        f"Entrenando {run_config['candidate']} con {training_rows:,} filas "
        f"({train_start_actual} a {train_end_actual})...",
        flush=True,
    )
    # El proceso local de Windows puede bloquear los pipes que usa el backend
    # paralelo de scikit-learn; un hilo mantiene el ajuste reproducible.
    with threadpool_limits(limits=1):
        model.fit(X_train, y_train)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento MLflow {EXPERIMENT_NAME}.")

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_config["candidate"],
    ) as run:
        mlflow.set_tags(
            {
                "team": "46",
                "question": "P1",
                "stage_of_work": "week9_challenger",
                "candidate": run_config["candidate"],
                "base_model_version": champion_version,
                "historical_data_version": manifest["data_version"],
                "training_data_version": training_data_version,
                "supplemental_october_sha256": supplemental["sha256"],
                "evaluation_holdout": "2026-10-16/2026-10-31",
            }
        )
        mlflow.log_params(
            {
                **parameters,
                "candidate": run_config["candidate"],
                "algorithm": "HistGradientBoostingRegressor",
                "training_rows": training_rows,
                "training_target_start": train_start_actual,
                "training_target_end_inclusive": train_end_actual,
                "training_target_end_exclusive": run_config[
                    "training_target_time_end_exclusive"
                ],
                "base_model_version": champion_version,
                "historical_data_version": manifest["data_version"],
                "training_data_version": training_data_version,
                "supplemental_october_rows": supplemental["normalized_rows"],
                "holdout_start_inclusive": run_config[
                    "evaluation_target_time_start_inclusive"
                ],
                "holdout_end_exclusive": run_config[
                    "evaluation_target_time_end_exclusive"
                ],
            }
        )
        mlflow.log_dict(
            {
                "model_name": model_name,
                "base_model_version": champion_version,
                "candidate": run_config["candidate"],
                "algorithm": "HistGradientBoostingRegressor",
                "hyperparameters": parameters,
                "features": FEATURE_COLUMNS,
                "target": TARGET_COLUMN,
                "training_target_window": {
                    "start_inclusive": train_start_actual,
                    "end_exclusive": run_config[
                        "training_target_time_end_exclusive"
                    ],
                    "last_observed_target_time": train_end_actual,
                    "rows": training_rows,
                },
                "evaluation_holdout": {
                    "start_inclusive": run_config[
                        "evaluation_target_time_start_inclusive"
                    ],
                    "end_exclusive": run_config[
                        "evaluation_target_time_end_exclusive"
                    ],
                    "labels_used_in_training": False,
                },
                "historical_data_version": manifest["data_version"],
                "supplemental_current_data": supplemental,
                "training_data_version": training_data_version,
            },
            "config/week9_challenger_training.json",
        )
        mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")
        input_example = X_train.head(5).copy()
        signature = infer_signature(input_example, model.predict(input_example))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        mlflow.set_tag("logged_model_uri", model_info.model_uri)

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name,
            await_registration_for=300,
        )
        version = str(registered.version)
        version_description = (
            "Challenger P1 reentrenado con datos históricos hasta 2026-10-15; "
            "validación reservada para 2026-10-16 a 2026-10-31. "
            f"Training data version: {training_data_version}."
        )
        client.update_model_version(
            name=model_name,
            version=version,
            description=version_description,
        )
        for key, value in {
            "team": "46",
            "question": "P1",
            "candidate": run_config["candidate"],
            "stage_of_work": "week9_challenger",
            "base_model_version": champion_version,
            "historical_data_version": manifest["data_version"],
            "training_data_version": training_data_version,
            "training_cutoff_exclusive": run_config[
                "training_target_time_end_exclusive"
            ],
            "evaluation_holdout": "2026-10-16/2026-10-31",
            "source_run_id": run.info.run_id,
            "source_model_uri": model_info.model_uri,
            "supplemental_october_sha256": supplemental["sha256"],
            "validation_status": "pending_ab_test",
        }.items():
            client.set_model_version_tag(model_name, version, key, str(value))

    # Registro como versión nueva; el alias vigente permanece en v1 hasta
    # que el A/B testing entregue una recomendación explícita.
    current_alias = client.get_model_version_by_alias(model_name, "champion")
    if str(current_alias.version) != champion_version:
        raise RuntimeError("El registro del challenger alteró inesperadamente el alias champion.")
    loaded_challenger = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    probe = np.asarray(loaded_challenger.predict(X_train.head(5)), dtype=float).reshape(-1)
    if len(probe) != 5 or not np.isfinite(probe).all():
        raise RuntimeError("El challenger registrado no produjo predicciones válidas.")

    registration = {
        "model_name": model_name,
        "champion_version": champion_version,
        "challenger_version": version,
        "challenger_alias": None,
        "source_run_id": run.info.run_id,
        "source_model_uri": model_info.model_uri,
        "registry_model_uri": f"models:/{model_name}/{version}",
        "candidate": run_config["candidate"],
        "training_rows": training_rows,
        "training_target_start": train_start_actual,
        "training_target_end_inclusive": train_end_actual,
        "training_target_end_exclusive": run_config[
            "training_target_time_end_exclusive"
        ],
        "evaluation_holdout": {
            "start_inclusive": run_config[
                "evaluation_target_time_start_inclusive"
            ],
            "end_exclusive": run_config["evaluation_target_time_end_exclusive"],
        },
        "training_data_version": training_data_version,
        "model_ready": True,
        "champion_alias_still_points_to": str(current_alias.version),
    }
    REGISTRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRATION_PATH.write_text(
        json.dumps(registration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
