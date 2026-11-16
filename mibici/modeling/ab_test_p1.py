"""A/B testing de P1: compara champion y challenger desde MLflow Model Registry.

Uso, desde la raíz del repositorio con MLFLOW_TRACKING_URI configurada:
    python -m mibici.modeling.ab_test_p1

Ambas versiones se recuperan por alias y se evalúan sobre el mismo
conjunto del periodo actual: la partición de prueba (15 % final de la
división cronológica 70/15/15 por target_time), el mismo periodo usado
como conjunto actual en el reporte de drift con Evidently. Calcula MAE,
RMSE y R², guarda la tabla y una gráfica en reports/semana9_ab_test/ y
registra el resultado en MLflow. El alias champion solo se mueve si se
indica --promote y el challenger cumple la regla de decisión.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mibici.config import MLFLOW_EXPERIMENT_NAME, PROJ_ROOT
from mibici.features import make_hourly_panel
from mibici.modeling.experiment_p1 import get_git_commit, prepare_xy
from mibici.modeling.train import chronological_split, load_inputs

REGISTERED_NAME = "mibici-equipo46-p1-demanda"
OUTPUT_DIR = PROJ_ROOT / "reports" / "semana9_ab_test"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-alias", default="champion")
    parser.add_argument("--challenger-alias", default="challenger")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Mejora relativa mínima de MAE para preferir al challenger (0.01 = 1 %%).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Mueve el alias champion al challenger si cumple la regla de decisión.",
    )
    return parser.parse_args()


def metrics_for(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    args = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    versions = {
        "champion": client.get_model_version_by_alias(REGISTERED_NAME, args.champion_alias),
        "challenger": client.get_model_version_by_alias(REGISTERED_NAME, args.challenger_alias),
    }
    if versions["champion"].version == versions["challenger"].version:
        raise SystemExit("champion y challenger apuntan a la misma versión; no hay nada que comparar.")

    logger.info("Cargando datos P1 y construyendo el periodo actual...")
    trips, manifest = load_inputs()
    data_version = str(manifest["data_version"])
    panel = make_hourly_panel(trips)
    _, _, test, bounds = chronological_split(panel)
    del panel, trips
    X_test, y_test = prepare_xy(test)
    period = {
        "particion": "test (15 % final, división cronológica 70/15/15 por target_time)",
        "inicio": bounds["test_target_start"] if "test_target_start" in bounds else str(test["target_time"].min()),
        "fin": bounds["test_target_end"] if "test_target_end" in bounds else str(test["target_time"].max()),
        "filas": int(len(X_test)),
        "estaciones": int(test["station_id"].nunique()),
        "data_version": data_version,
    }
    del test

    rows = []
    for role, version in versions.items():
        uri = f"models:/{REGISTERED_NAME}/{version.version}"
        logger.info(f"Evaluando {role}: {uri}")
        model = mlflow.pyfunc.load_model(uri)
        predictions = np.maximum(np.asarray(model.predict(X_test), dtype=float).reshape(-1), 0.0)
        rows.append({
            "rol": role,
            "alias": args.champion_alias if role == "champion" else args.challenger_alias,
            "version": str(version.version),
            "candidato": (version.tags or {}).get("selected_run_name", ""),
            **metrics_for(y_test, predictions),
        })
    table = pd.DataFrame(rows)

    champ = table.loc[table["rol"] == "champion"].iloc[0]
    chall = table.loc[table["rol"] == "challenger"].iloc[0]
    mae_improvement = (champ["mae"] - chall["mae"]) / champ["mae"]
    challenger_wins = bool(
        mae_improvement >= args.min_improvement
        and chall["rmse"] <= champ["rmse"]
        and chall["r2"] >= champ["r2"]
    )
    decision = {
        "regla": (
            f"El challenger reemplaza al champion solo si reduce el MAE al menos "
            f"{args.min_improvement:.0%} y no empeora RMSE ni R²."
        ),
        "mejora_relativa_mae": float(mae_improvement),
        "version_vigente_recomendada": str(chall["version"] if challenger_wins else champ["version"]),
        "ganador": "challenger" if challenger_wins else "champion",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "ab_test_metricas.csv", index=False)
    (OUTPUT_DIR / "ab_test_resultado.json").write_text(
        json.dumps({"periodo_actual": period, "metricas": rows, "decision": decision},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    labels = [f"{r['rol']}\nv{r['version']}" for r in rows]
    for ax, metric, title in zip(axes, ("mae", "rmse", "r2"), ("MAE (menor es mejor)",
                                                               "RMSE (menor es mejor)",
                                                               "R² (mayor es mejor)")):
        values = table[metric].to_list()
        bars = ax.bar(labels, values, color=["#4C78A8", "#F58518"])
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.15)
    fig.suptitle(f"A/B testing P1 · periodo actual {period['inicio'][:10]} a {period['fin'][:10]}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ab_test_metricas.png", dpi=150)

    with mlflow.start_run(run_name="P1_AB_test"):
        mlflow.set_tags({
            "team": "46",
            "question": "P1",
            "stage_of_work": "ab_test",
            "data_version": data_version,
            "git_commit": get_git_commit(),
            "winner": decision["ganador"],
        })
        mlflow.log_params({
            "champion_version": champ["version"],
            "challenger_version": chall["version"],
            "evaluation_rows": period["filas"],
            "evaluation_start": period["inicio"],
            "evaluation_end": period["fin"],
            "min_improvement": args.min_improvement,
        })
        for r in rows:
            for metric in ("mae", "rmse", "r2"):
                mlflow.log_metric(f"{r['rol']}_{metric}", r[metric])
        mlflow.log_metric("mae_relative_improvement", decision["mejora_relativa_mae"])
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="ab_test")
    plt.close(fig)

    promoted = False
    if args.promote and challenger_wins:
        client.set_registered_model_alias(REGISTERED_NAME, args.champion_alias, chall["version"])
        client.set_model_version_tag(REGISTERED_NAME, chall["version"], "validation_status", "approved")
        promoted = True

    print("")
    print(f"Periodo actual: {period['inicio']} a {period['fin']} | "
          f"{period['filas']:,} filas | {period['estaciones']} estaciones | datos {data_version}")
    print(table[["rol", "alias", "version", "candidato", "mae", "rmse", "r2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("")
    print(f"Mejora relativa de MAE del challenger: {mae_improvement:+.2%}")
    print(decision["regla"])
    print(f"Versión vigente recomendada: v{decision['version_vigente_recomendada']} ({decision['ganador']})")
    if promoted:
        print(f"Alias {args.champion_alias} movido a la versión {chall['version']}.")
    elif args.promote:
        print("No se movió el alias champion: el challenger no cumple la regla.")
    print(f"Resultados guardados en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
