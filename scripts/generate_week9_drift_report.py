"""Compara el conjunto de entrenamiento de P1 con octubre usando Evidently."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "week9_datasets.json"
DATA_DIR = ROOT / "data" / "monitoring"
REPORT_DIR = ROOT / "reports" / "semana9"
HTML_PATH = REPORT_DIR / "evidently_drift_october_2026.html"
JSON_PATH = REPORT_DIR / "evidently_drift_october_2026.json"
SUMMARY_PATH = REPORT_DIR / "evidently_drift_summary.json"


def load_eval_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero scripts/prepare_week9_datasets.py."
        )
    frame = pd.read_parquet(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas de evaluación en {path.name}: {missing}")
    return frame[columns].copy()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    features = config["feature_columns"]
    target = config["target_column"]
    report_columns = [*features, target, "target_time"]
    categorical_columns = [
        "station_id",
        "target_hour",
        "target_dayofweek",
        "target_is_weekend",
    ]
    numerical_columns = [
        column
        for column in report_columns
        if column not in categorical_columns and column != "target_time"
    ]

    reference_path = DATA_DIR / "reference_p1_train.parquet"
    current_path = DATA_DIR / "current_p1_2026_10.parquet"
    reference = load_eval_frame(reference_path, report_columns)
    current = load_eval_frame(current_path, report_columns)
    for frame in (reference, current):
        frame["target_time"] = pd.to_datetime(frame["target_time"])

    # Los identificadores y variables de calendario son categorías, aunque estén
    # almacenados como enteros en el Parquet. La etiqueta se resume para calidad,
    # pero el preset de drift se limita a las ocho entradas del modelo.
    for frame in (reference, current):
        for column in categorical_columns:
            frame[column] = frame[column].astype("string")

    definition = DataDefinition(
        timestamp="target_time",
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        datetime_columns=["target_time"],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
    current_dataset = Dataset.from_pandas(current, data_definition=definition)

    report = Report(
        [
            DataDriftPreset(columns=features),
            # Keep quality statistics (nulls, duplicates, empty and constant
            # columns) without applying equality tests to periods whose sizes
            # and distributions are expected to differ.
            DataSummaryPreset(),
        ],
        metadata={
            "model": config["model"]["name"],
            "model_version": config["model"]["version"],
            "reference_period": "2026-01-01 01:00 a 2026-07-11 02:00",
            "current_period": "2026-10-01 01:00 a 2026-10-31 23:00",
        },
    )
    snapshot = report.run(
        current_dataset,
        reference_dataset,
        name="mibici-p1-drift-october-2026",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(HTML_PATH))
    JSON_PATH.write_text(snapshot.json(), encoding="utf-8")
    summary = {
        "evidently_version": version("evidently"),
        "model": config["model"],
        "reference": {
            "path": str(reference_path.relative_to(ROOT)),
            "rows": len(reference),
            "period": config["reference_window"],
        },
        "current": {
            "path": str(current_path.relative_to(ROOT)),
            "rows": len(current),
            "period": config["current_window"],
        },
        "drift_features": features,
        "quality_columns": report_columns,
        "artifacts": [
            str(HTML_PATH.relative_to(ROOT)),
            str(JSON_PATH.relative_to(ROOT)),
        ],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
