"""Prepara las muestras de referencia y actuales para monitorear drift de P1.

Entradas esperadas: data/processed/trips_2026_01.parquet a trips_2026_09.parquet
y el CSV complementario del mes actual indicado en configs/week9_datasets.json.
Salida local (ignorada por Git):
  data/monitoring/reference_p1_train.parquet
  data/monitoring/current_p1_2026_10.parquet
  reports/semana9/dataset_summary.json

Desde la raíz del repositorio:
  python scripts/prepare_week9_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mibici.features import make_hourly_panel
from mibici.dataset import normalize_trips, read_csv_any_encoding, sha256


CONFIG_PATH = ROOT / "configs" / "week9_datasets.json"
MANIFEST_PATH = ROOT / "data" / "manifest" / "data_manifest.json"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "monitoring"
SUMMARY_PATH = ROOT / "reports" / "semana9" / "dataset_summary.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_trips(config: dict) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_version = config["data_version"]
    if manifest.get("data_version") != expected_version:
        raise ValueError(
            "La versión de datos no coincide con la usada para el champion: "
            f"esperada={expected_version}, encontrada={manifest.get('data_version')}"
        )
    if not manifest.get("complete"):
        raise ValueError("La ingesta indicada por el manifiesto está incompleta.")

    start_month = config["inputs"]["start_month"]
    end_month = config["inputs"]["end_month"]
    months = pd.period_range(start=start_month, end=end_month, freq="M")
    paths = [
        PROCESSED_DIR / f"trips_{month.strftime('%Y_%m')}.parquet"
        for month in months
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan Parquet mensuales requeridos. Ejecuta la ingesta del rango "
            f"{start_month} a {end_month}. Archivos faltantes: {missing}"
        )

    historical_trips = [pd.read_parquet(path) for path in paths]

    # El manifiesto versionado corresponde a los datos usados al registrar el
    # champion (enero-septiembre). Octubre se agrega como periodo actual aparte.
    supplemental_month = config["inputs"]["supplemental_current_month"]
    year, month = (int(part) for part in supplemental_month.split("-"))
    supplemental_path = ROOT / config["inputs"]["supplemental_current_file"]
    if not supplemental_path.is_file():
        raise FileNotFoundError(
            "Falta el CSV complementario del periodo actual: "
            f"{supplemental_path}. Colócalo en data/raw/ y vuelve a ejecutar."
        )

    raw_supplemental, encoding = read_csv_any_encoding(supplemental_path)
    current_trips, supplemental_stats = normalize_trips(
        raw_supplemental, year=year, month=month
    )
    supplemental_info = {
        "month": supplemental_month,
        "file": str(supplemental_path.relative_to(ROOT)),
        "sha256": sha256(supplemental_path),
        "encoding": encoding,
        "raw_rows": supplemental_stats["raw_rows"],
        "normalized_rows": supplemental_stats["final_rows"],
        "date_format": supplemental_stats["date_format"],
        "dropped": supplemental_stats["dropped"],
    }
    return pd.concat([*historical_trips, current_trips], ignore_index=True), supplemental_info


def main() -> None:
    config = load_config()
    trips, supplemental_info = load_trips(config)
    panel = make_hourly_panel(trips)
    target = config["target_column"]
    features = config["feature_columns"]
    columns = [*features, target, "target_time"]

    # Las horas sin etiqueta no sirven para la evaluación ni para el reporte
    # retrospectivo del objetivo. Las entradas ya se calcularon con el historial
    # completo, por lo que las ventanas móviles conservan su contexto temporal.
    panel = panel.dropna(subset=[target]).copy()
    panel["target_time"] = pd.to_datetime(panel["target_time"])

    reference = config["reference_window"]
    reference_start = pd.Timestamp(reference["start_inclusive"])
    reference_end = pd.Timestamp(reference["end_inclusive"])
    reference_rows = panel.loc[
        panel["target_time"].between(reference_start, reference_end, inclusive="both"),
        columns,
    ].copy()

    current = config["current_window"]
    current_start = pd.Timestamp(current["start_inclusive"])
    current_end = pd.Timestamp(current["end_exclusive"])
    current_rows = panel.loc[
        (panel["target_time"] >= current_start)
        & (panel["target_time"] < current_end),
        columns,
    ].copy()

    if reference_rows.empty or current_rows.empty:
        raise ValueError(
            "Una de las muestras quedó vacía. Revisa las fechas y la cobertura de datos."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_path = OUTPUT_DIR / "reference_p1_train.parquet"
    current_path = OUTPUT_DIR / "current_p1_2026_10.parquet"
    reference_rows.to_parquet(reference_path, index=False)
    current_rows.to_parquet(current_path, index=False)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = {
        "model": config["model"],
        "historical_data_version": manifest["data_version"],
        "source_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "supplemental_current_data": supplemental_info,
        "reference": {
            "period_start_inclusive": reference_start.isoformat(),
            "period_end_inclusive": reference_end.isoformat(),
            "rows": len(reference_rows),
            "features": features,
            "target": target,
            "output": str(reference_path.relative_to(ROOT)),
        },
        "current": {
            "period_start_inclusive": current_start.isoformat(),
            "period_end_exclusive": current_end.isoformat(),
            "rows": len(current_rows),
            "features": features,
            "target": target,
            "output": str(current_path.relative_to(ROOT)),
        },
        "note": (
            "El target se conserva como etiqueta para evaluación retrospectiva; "
            "no es una característica de entrada del modelo."
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
