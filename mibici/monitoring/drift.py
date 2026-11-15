"""Semana 9 / Persona 1: monitoreo de data drift del panel P1 de MiBici.

Ejecutar desde la raíz del repositorio:
    python -m mibici.monitoring.drift
    python -m mibici.monitoring.drift --demo

No modifica modelos ni el registro de MLflow. Las observaciones de prueba se
reservan para la comparación de versiones que realizará otro integrante.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mibici.config import MANIFEST_PATH, PROCESSED_DATA_DIR, PROJ_ROOT, RANDOM_SEED
from mibici.features import make_hourly_panel

TARGET = "target_departures_next_hour"
TIME = "target_time"
NUMERIC = (
    "departures_this_hour",
    "arrivals_this_hour",
    "departures_mean_last_24h",
    "departures_mean_last_168h",
)
CATEGORICAL = ("target_hour", "target_dayofweek", "target_is_weekend")
FEATURES = NUMERIC + CATEGORICAL
REQUIRED_TRIP_COLUMNS = ["start_time", "end_time", "origin_id", "destination_id"]
METHOD_BY_COLUMN = {**{c: "wasserstein" for c in NUMERIC}, **{c: "jensenshannon" for c in CATEGORICAL}}
THRESHOLD = 0.10


def load_trip_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carga solo columnas necesarias de los Parquet citados por el manifiesto."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {MANIFEST_PATH}. Ejecuta python -m mibici.dataset"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise ValueError("El manifiesto indica ingesta incompleta. Reejecuta python -m mibici.dataset")
    items = [
        x for x in manifest.get("files", [])
        if re.fullmatch(r"trips_\d{4}_\d{2}\.parquet", str(x.get("name", "")))
    ]
    if not items:
        raise ValueError("El manifiesto no incluye archivos trips_AAAA_MM.parquet")
    tables = []
    for item in sorted(items, key=lambda x: x["name"]):
        path = PROCESSED_DATA_DIR / item["name"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Falta {path}. Ejecuta python -m mibici.dataset para descargar los datos."
            )
        print(f"[DATOS] Leyendo {path.name}", flush=True)
        tables.append(pd.read_parquet(path, columns=REQUIRED_TRIP_COLUMNS))
    return pd.concat(tables, ignore_index=True), manifest


def select_chronological_periods(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replica los cortes por hora objetivo 70/15/15 usados por P1.

    Referencia = 70 % inicial (entrenamiento); periodo actual = 15 % final
    (prueba), sin utilizar la validación intermedia. No usa objetivos como
    columnas de Evidently; el target se utiliza solo para identificar filas
    elegibles, como en el entrenamiento actual de P1.
    """
    must_have = {TIME, TARGET, "station_id", *FEATURES}
    missing = must_have - set(panel.columns)
    if missing:
        raise ValueError(f"Faltan columnas del panel P1: {sorted(missing)}")
    valid = panel[TARGET].notna()
    hours = pd.DatetimeIndex(
        pd.to_datetime(panel.loc[valid, TIME]).drop_duplicates().sort_values()
    )
    if len(hours) < 10:
        raise ValueError("Se requieren al menos 10 horas objetivo para dividir el panel")
    train_end = hours[int(len(hours) * 0.70)]
    test_start = hours[int(len(hours) * 0.85)]
    ref_mask = valid & (panel[TIME] < train_end)
    curr_mask = valid & (panel[TIME] >= test_start)
    fields = ["station_id", TIME, *FEATURES]
    reference = panel.loc[ref_mask, fields].copy()
    current = panel.loc[curr_mask, fields].copy()
    if reference.empty or current.empty:
        raise ValueError("Las particiones de referencia o periodo actual están vacías")
    if not (reference[TIME].max() < current[TIME].min()):
        raise AssertionError("Se detectó solapamiento temporal entre periodos")
    meta = {
        "criterio": "split_cronologico_70_15_15_por_target_time_P1",
        "referencia": {
            "particion": "train (70 % inicial)",
            "filas_completas": int(len(reference)),
            "inicio": str(reference[TIME].min()),
            "fin": str(reference[TIME].max()),
            "estaciones": int(reference["station_id"].nunique()),
        },
        "actual": {
            "particion": "test (15 % final)",
            "filas_completas": int(len(current)),
            "inicio": str(current[TIME].min()),
            "fin": str(current[TIME].max()),
            "estaciones": int(current["station_id"].nunique()),
        },
        "validacion": {
            "particion": "validation (15 % intermedio), no se usa en este reporte",
            "inicio": str(train_end),
            "fin_exclusivo": str(test_start),
        },
    }
    return reference, current, meta


def sample_period(data: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Muestreo aleatorio uniforme, reproducible, sin reemplazo."""
    if limit < 100:
        raise ValueError("--sample-size debe ser >= 100")
    if data.empty:
        raise ValueError("No puede muestrearse un conjunto vacío")
    size = min(limit, len(data))
    return data.sample(n=size, random_state=seed).reset_index(drop=True)


def quality_summary(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos descriptivos y nulos, independientes de Evidently."""
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        for period_name, frame in (("referencia", reference), ("actual", current)):
            series = frame[feature]
            numeric = pd.to_numeric(series, errors="coerce")
            rows.append(
                {
                    "variable": feature,
                    "periodo": period_name,
                    "registros": len(series),
                    "nulos": int(series.isna().sum()),
                    "porcentaje_nulos": round(float(series.isna().mean() * 100), 4),
                    "unicos": int(series.nunique(dropna=True)),
                    "promedio": round(float(numeric.mean()), 4) if numeric.notna().any() else None,
                    "mediana": round(float(numeric.median()), 4) if numeric.notna().any() else None,
                    "porcentaje_ceros": round(float((numeric == 0).mean() * 100), 4),
                }
            )
    return pd.DataFrame(rows)


def prepare_evidently_frames(reference: pd.DataFrame, current: pd.DataFrame):
    """Conserva únicamente variables observables; nunca objetivos futuros."""
    from evidently import Dataset, DataDefinition

    for name, frame in (("referencia", reference), ("actual", current)):
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Faltan columnas de {name}: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"El periodo {name} está vacío")
        if frame[list(FEATURES)].isna().all(axis=0).any():
            raise ValueError(f"{name} contiene al menos una variable completamente vacía")

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.loc[:, FEATURES].copy()
        for feature in NUMERIC:
            out[feature] = pd.to_numeric(out[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        for feature in CATEGORICAL:
            # Tratamos el calendario como categorías, no como escala continua.
            out[feature] = out[feature].astype("Int64").astype("string")
        return out

    schema = DataDefinition(
        numerical_columns=list(NUMERIC), categorical_columns=list(CATEGORICAL)
    )
    return (
        Dataset.from_pandas(transform(reference), data_definition=schema),
        Dataset.from_pandas(transform(current), data_definition=schema),
    )


def parse_evidently_drift(snapshot_dict: dict[str, Any]) -> pd.DataFrame:
    """Extrae ValueDrift de la estructura 0.7.x sin inventar resultados."""
    records = []
    for metric in snapshot_dict.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        if "ValueDrift" not in name:
            continue
        config = metric.get("config") or {}
        column = config.get("column")
        if not column:
            matched = re.search(r"\bcolumn=['\"]?([^,'\")]+)", name)
            column = matched.group(1) if matched else None
        if column not in FEATURES:
            continue
        value = metric.get("value")
        if isinstance(value, dict):
            value = value.get("value", value.get("score"))
        if not isinstance(value, (int, float)):
            continue
        score = float(value)
        if not np.isfinite(score):
            continue
        records.append(
            {
                "variable": column,
                "metodo": METHOD_BY_COLUMN[column],
                "score": round(score, 6),
                "umbral": THRESHOLD,
                "drift_detectado": score >= THRESHOLD,
            }
        )
    return pd.DataFrame(records).drop_duplicates(subset=["variable"]) if records else pd.DataFrame()


def make_readme_summary(meta: dict[str, Any], parsed: pd.DataFrame) -> str:
    lines = [
        "# Semana 9 – Evidently: data drift de MiBici P1",
        "",
        "## Periodos utilizados y justificación",
        "",
        "Se utiliza la misma lógica temporal 70/15/15 del entrenamiento P1: "
        "referencia = entrenamiento (primer 70 % de horas objetivo), "
        "actual = prueba (último 15 %). El 15 % intermedio es validación.",
        "Esta comparación es **retrospectiva**, no monitoreo de tráfico en vivo.",
        "",
        f"- Referencia: {meta['referencia']['inicio']} a {meta['referencia']['fin']}",
        f"- Actual: {meta['actual']['inicio']} a {meta['actual']['fin']}",
        f"- Registros originales: {meta['referencia']['filas_completas']} y {meta['actual']['filas_completas']}",
        f"- Muestras Evidently: {meta['referencia']['filas_muestra']} y {meta['actual']['filas_muestra']}",
        f"- Versión de datos: {meta.get('data_version', 'sin dato')}",
        "- Muestreo: aleatorio uniforme sin reemplazo, semilla fija (muestra representativa de filas, no de meses).",
        "",
        "## Variables examinadas",
        "",
        *[f"- `{c}`: {METHOD_BY_COLUMN[c]} (umbral {THRESHOLD})." for c in FEATURES],
        "",
        "Se excluyen expresamente objetivos futuros (`target_*_next_hour`), "
        "identificadores (`station_id`) y marcas de tiempo del análisis "
        "estadístico. `target_hour`, `target_dayofweek` y `target_is_weekend` "
        "son características de calendario conocidas antes de la hora objetivo.",
        "",
        "## Resultados de Evidently por variable",
        "",
    ]
    if parsed.empty or set(parsed["variable"]) != set(FEATURES):
        lines += [
            "No fue posible extraer todas las métricas tabulares de Evidently ",
            "en formato automático. Consulta el HTML y el JSON para interpretar cada variable.",
            "**No debe asignarse un resultado de drift sin revisar los datos.**",
            "",
        ]
    else:
        lines += [
            "| Variable | Método | Score | Umbral | Drift |",
            "|---|---|---:|---:|---|",
        ]
        for rec in parsed.to_dict("records"):
            status = "Sí" if rec["drift_detectado"] else "No"
            lines.append(
                f"| {rec['variable']} | {rec['metodo']} | {rec['score']:.4f} | "
                f"{rec['umbral']:.2f} | {status} |"
            )
        lines.append("")
        lines.append(f"**Variables con drift (muestra analizada):** {int(parsed['drift_detectado'].sum())} de {len(parsed)}.")
    lines += [
        "",
        "## Interpretación responsable",
        "",
        "- El drift estadístico no demuestra por sí solo degradación del desempeño.",
        "- Revisar las diferencias de medias, ceros y faltantes en `calidad_variables.csv`.",
        "- El P1 baseline público es `DummyRegressor` sin variables predictoras: "
        "estos indicadores describen el entorno y las posibles entradas para modelos futuros; "
        "no se puede afirmar que cada variable modifique directamente sus predicciones.",
        "- Para demostrar impacto en desempeño, comparar modelos contra valores observados "
        "sobre el mismo periodo de prueba (Persona 3).",
        "- Para la conclusión, examinar cuáles variables tienen drift, dirección del cambio "
        "y su significado operativo para las estaciones y las horas de demanda.",
        "",
        "## Archivos de evidencia",
        "",
        "- `evidently_data_drift.html`: análisis de drift y gráficas de Evidently.",
        "- `evidently_calidad_datos.html`: calidad y estadísticas de los datos.",
        "- `calidad_variables.csv`: registros, valores nulos, medias y ceros.",
        "- `periodos_dataset.json`: datos y cortes exactos de la ejecución.",
        "- `evidently_resultado_drift.json`: resultados estructurados de Evidently.",
        "",
        "## Fuentes",
        "",
        "- Evidently. Documentación oficial: https://docs.evidentlyai.com/quickstart_ml",
        "- Código propio del Equipo 46: `mibici/features.py` y `mibici/modeling/train.py`.",
        "",
    ]
    return "\n".join(lines)


def make_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Datos sintéticos solo para probar instalación, NO para entregar como resultados."""
    rng = np.random.default_rng(RANDOM_SEED)

    def gen(n: int, mean: float, start: str) -> pd.DataFrame:
        hours = rng.integers(0, 24, size=n)
        dow = rng.integers(0, 7, size=n)
        return pd.DataFrame({
            "station_id": rng.integers(1, 30, size=n),
            TIME: pd.date_range(start, periods=n, freq="h"),
            "departures_this_hour": rng.poisson(mean, n).astype(float),
            "arrivals_this_hour": rng.poisson(mean * 0.95, n).astype(float),
            "departures_mean_last_24h": rng.normal(mean, 1.2, n).clip(0),
            "departures_mean_last_168h": rng.normal(mean, 0.9, n).clip(0),
            "target_hour": hours,
            "target_dayofweek": dow,
            "target_is_weekend": (dow >= 5).astype(int),
        })

    ref, cur = gen(1200, 5.0, "2026-01-01"), gen(1200, 9.0, "2026-07-01")
    meta = {
        "criterio": "DEMOSTRACION_SINTETICA_NO_USAR_EN_ENTREGA",
        "data_version": "SINTETICA",
        "referencia": {"inicio": str(ref[TIME].min()), "fin": str(ref[TIME].max()),
                       "filas_completas": len(ref), "estaciones": int(ref["station_id"].nunique())},
        "actual": {"inicio": str(cur[TIME].min()), "fin": str(cur[TIME].max()),
                   "filas_completas": len(cur), "estaciones": int(cur["station_id"].nunique())},
    }
    return ref, cur, meta


def generate_reports(
    ref: pd.DataFrame, cur: pd.DataFrame, meta: dict[str, Any], outdir: Path
) -> None:
    """Ejecuta Evidently 0.7.x, guarda HTML y salidas para el informe."""
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    outdir.mkdir(parents=True, exist_ok=True)
    quality_summary(ref, cur).to_csv(outdir / "calidad_variables.csv", index=False, encoding="utf-8-sig")
    ref_dataset, cur_dataset = prepare_evidently_frames(ref, cur)
    print("[EVIDENTLY] Calculando data drift...", flush=True)
    drift_report = Report([
        DataDriftPreset(
            num_method="wasserstein", cat_method="jensenshannon",
            num_threshold=THRESHOLD, cat_threshold=THRESHOLD, drift_share=0.5,
        )
    ])
    drift_snapshot = drift_report.run(cur_dataset, ref_dataset)
    drift_snapshot.save_html(str(outdir / "evidently_data_drift.html"))
    raw_results = drift_snapshot.dict()
    (outdir / "evidently_resultado_drift.json").write_text(
        json.dumps(raw_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    parsed = parse_evidently_drift(raw_results)
    if not parsed.empty:
        parsed.to_csv(outdir / "drift_por_variable.csv", index=False, encoding="utf-8-sig")

    print("[EVIDENTLY] Calculando calidad de datos...", flush=True)
    quality_report = Report([DataSummaryPreset()])
    quality_snapshot = quality_report.run(cur_dataset, ref_dataset)
    quality_snapshot.save_html(str(outdir / "evidently_calidad_datos.html"))

    (outdir / "periodos_dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "resumen_evidently.md").write_text(
        make_readme_summary(meta, parsed), encoding="utf-8"
    )
    print("\n[OK] Reportes generados en:", outdir.resolve())
    print("  - evidently_data_drift.html")
    print("  - evidently_calidad_datos.html")
    print("  - resumen_evidently.md")
    print("  - periodos_dataset.json")
    print("  - calidad_variables.csv")
    if not parsed.empty:
        print("  - drift_por_variable.csv")
        for row in parsed.to_dict("records"):
            status = "SI" if row["drift_detectado"] else "NO"
            print(f"    {row['variable']}: drift={status} score={row['score']:.4f}")
    else:
        print("[AVISO] Revisa el HTML/JSON: no se pudo extraer el resumen tabular automáticamente.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semana 9: Evidently MiBici P1")
    parser.add_argument("--sample-size", type=int, default=10_000,
                        help="Filas máximas por periodo para el reporte (default: 10000)")
    parser.add_argument("--outdir", type=Path, default=PROJ_ROOT / "reports" / "semana9_evidently",
                        help="Carpeta para reportes")
    parser.add_argument("--demo", action="store_true",
                        help="Usa datos SINTÉTICOS para probar instalación, no válido para entregar")
    args = parser.parse_args()

    if args.sample_size < 100:
        parser.error("--sample-size debe ser >= 100")
    if args.demo:
        print("[ADVERTENCIA] DEMOSTRACIÓN SINTÉTICA. NO usar como evidencia real.")
        reference, current, meta = make_demo_data()
    else:
        print("[P1] Leyendo la ingesta real del proyecto MiBici...")
        trips, manifest = load_trip_data()
        print(f"[P1] Construyendo panel horario con {len(trips):,} viajes...")
        panel = make_hourly_panel(trips)
        del trips
        reference, current, meta = select_chronological_periods(panel)
        del panel
        meta["data_version"] = manifest.get("data_version", "sin dato")
        meta["mes_inicio"] = manifest.get("start_month")
        meta["mes_fin"] = manifest.get("end_month")

    reference = sample_period(reference, args.sample_size, RANDOM_SEED)
    current = sample_period(current, args.sample_size, RANDOM_SEED + 1)
    meta["referencia"]["filas_muestra"] = int(len(reference))
    meta["actual"]["filas_muestra"] = int(len(current))
    meta["semilla_muestreo"] = int(RANDOM_SEED)
    meta["tipo_ejecucion"] = "demostracion" if args.demo else "datos_reales_del_repositorio"
    output = args.outdir / "demo" if args.demo else args.outdir
    generate_reports(reference, current, meta, output)


if __name__ == "__main__":
    main()
