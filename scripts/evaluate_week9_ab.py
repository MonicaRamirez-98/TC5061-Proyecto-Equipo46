"""Evalúa champion y challenger en el holdout temporal común de octubre."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mibici.serving.contract import FEATURE_COLUMNS


CONFIG_PATH = ROOT / "configs" / "week9_abtest.json"
REGISTRATION_PATH = ROOT / "reports" / "semana9" / "challenger_registration.json"
CURRENT_PATH = ROOT / "data" / "monitoring" / "current_p1_2026_10.parquet"
REPORT_DIR = ROOT / "reports" / "semana9"
RESULT_JSON = REPORT_DIR / "ab_test_october_second_half.json"
RESULT_MD = REPORT_DIR / "ab_test_october_second_half.md"
EXPERIMENT_NAME = "mibici-equipo46"
TARGET_COLUMN = "target_departures_next_hour"


def evaluate(model, features: pd.DataFrame, target: np.ndarray) -> tuple[dict, np.ndarray]:
    prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)
    if len(prediction) != len(target) or not np.isfinite(prediction).all():
        raise RuntimeError("Una versión devolvió predicciones no válidas.")
    prediction = np.maximum(prediction, 0.0)
    metrics = {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
        "mean_prediction_bias": float(np.mean(prediction - target)),
        "underprediction_rate": float(np.mean(prediction < target)),
    }
    return metrics, prediction


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    if not CURRENT_PATH.is_file():
        raise FileNotFoundError("Falta el conjunto actual; ejecuta primero la preparación de datos.")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    model_name = config["model_name"]
    champion_version = str(config["champion_version"])
    challenger_version = str(registration["challenger_version"])
    current_alias = client.get_model_version_by_alias(model_name, "champion")
    if str(current_alias.version) != champion_version:
        raise RuntimeError(
            f"El alias champion ahora apunta a {current_alias.version}; "
            f"se esperaba v{champion_version}."
        )

    data = pd.read_parquet(CURRENT_PATH)
    data["target_time"] = pd.to_datetime(data["target_time"])
    eval_start = pd.Timestamp(config["evaluation_target_time_start_inclusive"])
    eval_end = pd.Timestamp(config["evaluation_target_time_end_exclusive"])
    holdout = data.loc[
        (data["target_time"] >= eval_start) & (data["target_time"] < eval_end)
    ].copy()
    if holdout.empty:
        raise ValueError("El holdout de octubre quedó vacío.")

    features = holdout[list(FEATURE_COLUMNS)].astype("float32")
    target = holdout[TARGET_COLUMN].astype("float32").to_numpy()
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target).all():
        raise ValueError("El holdout contiene NaN o valores infinitos.")

    champion_uri = f"models:/{model_name}/{champion_version}"
    challenger_uri = f"models:/{model_name}/{challenger_version}"
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    challenger_model = mlflow.pyfunc.load_model(challenger_uri)
    champion_metrics, champion_predictions = evaluate(champion_model, features, target)
    challenger_metrics, challenger_predictions = evaluate(challenger_model, features, target)

    if challenger_metrics["mae"] < champion_metrics["mae"]:
        recommended = f"version_{challenger_version}"
        rationale = (
            "El challenger obtuvo menor MAE en el holdout temporal común; "
            "MAE es la métrica primaria usada para seleccionar el champion original."
        )
    elif challenger_metrics["mae"] > champion_metrics["mae"]:
        recommended = f"version_{champion_version}"
        rationale = (
            "El champion obtuvo menor MAE en el holdout temporal común; "
            "MAE es la métrica primaria usada para seleccionar el champion original."
        )
    else:
        recommended = (
            f"version_{challenger_version}"
            if challenger_metrics["rmse"] < champion_metrics["rmse"]
            else f"version_{champion_version}"
        )
        rationale = "El MAE empató; se desempató con menor RMSE."

    results = {
        "model_name": model_name,
        "evaluation": {
            "period_start_inclusive": eval_start.isoformat(),
            "period_end_exclusive": eval_end.isoformat(),
            "last_target_time": holdout["target_time"].max().isoformat(),
            "rows": len(holdout),
            "same_rows_for_both_models": True,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
        },
        "models": {
            f"version_{champion_version}": {
                "role": "champion",
                "model_uri": champion_uri,
                "metrics": champion_metrics,
            },
            f"version_{challenger_version}": {
                "role": "challenger",
                "model_uri": challenger_uri,
                "metrics": challenger_metrics,
            },
        },
        "delta_challenger_minus_champion": {
            metric: challenger_metrics[metric] - champion_metrics[metric]
            for metric in champion_metrics
        },
        "recommendation": {
            "version": recommended,
            "primary_metric": "mae",
            "rationale": rationale,
            "alias_changed": False,
            "current_champion_alias": str(current_alias.version),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    table = [
        "| Versión | Rol | MAE | RMSE | R² | Sesgo medio (predicción − real) | Subestimación |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in results["models"].items():
        m = item["metrics"]
        table.append(
            f"| {key.replace('version_', 'v')} | {item['role']} | {m['mae']:.4f} "
            f"| {m['rmse']:.4f} | {m['r2']:.4f} | {m['mean_prediction_bias']:.4f} "
            f"| {m['underprediction_rate']:.2%} |"
        )
    markdown = "\n".join(
        [
            "# A/B testing P1: octubre de 2026",
            "",
            f"Holdout: `{eval_start.isoformat()}` a antes de `{eval_end.isoformat()}` "
            f"({len(holdout):,} filas estación-hora). Ambos modelos recibieron las mismas filas.",
            "",
            *table,
            "",
            f"**Recomendación por MAE:** {recommended}. {rationale}",
            "",
            f"El alias `champion` permanece en la versión {current_alias.version}; "
            "este resultado no cambia el modelo servido.",
        ]
    )
    RESULT_MD.write_text(markdown + "\n", encoding="utf-8")

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="P1_AB_test_champion_vs_week9_challenger",
    ) as run:
        mlflow.set_tags(
            {
                "team": "46",
                "question": "P1",
                "stage_of_work": "week9_ab_test",
                "champion_version": champion_version,
                "challenger_version": challenger_version,
                "recommended_version": recommended.replace("version_", ""),
                "evaluation_window": "2026-10-16/2026-10-31",
            }
        )
        mlflow.log_params(
            {
                "model_name": model_name,
                "champion_version": champion_version,
                "challenger_version": challenger_version,
                "holdout_rows": len(holdout),
                "holdout_start_inclusive": eval_start.isoformat(),
                "holdout_end_exclusive": eval_end.isoformat(),
                "primary_metric": "mae",
                "alias_changed": False,
            }
        )
        metric_values = {
            f"champion_{key}": value
            for key, value in champion_metrics.items()
        }
        metric_values.update(
            {f"challenger_{key}": value for key, value in challenger_metrics.items()}
        )
        metric_values.update(
            {
                f"delta_challenger_minus_champion_{key}": value
                for key, value in results["delta_challenger_minus_champion"].items()
            }
        )
        mlflow.log_metrics(metric_values)
        mlflow.log_artifact(str(RESULT_JSON), artifact_path="reports")
        mlflow.log_artifact(str(RESULT_MD), artifact_path="reports")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
