"""Ingesta reproducible de MiBici.

Uso:
    python -m mibici.dataset                              # rango definido en config.py
    python -m mibici.dataset --start 2026-01 --end 2026-03
    python -m mibici.dataset --verify                     # comprueba los hashes del manifiesto

Pasos: 1) descarga los CSV mensuales y la nomenclatura, 2) valida y normaliza
columnas, codificación y fechas, 3) limpia y cuenta los registros descartados por
motivo, 4) guarda Parquet, 5) escribe un manifiesto con meses procesados y
faltantes, número de registros y hash SHA-256 de cada archivo, y 6) genera un
resumen en reports/ingesta_resumen.md.
"""
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import typer
from loguru import logger
from tqdm import tqdm

from mibici.config import (
    END_MONTH,
    GDL_LAT_RANGE,
    GDL_LON_RANGE,
    MANIFEST_PATH,
    MIBICI_FILES_BASE,
    MIBICI_PAGE_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    START_MONTH,
    STATIONS_FILE,
    TRIPS_FILE_TEMPLATE,
)

app = typer.Typer()

HEADERS = {"User-Agent": "Mozilla/5.0 (TC5061-Proyecto-MiBici)"}
SUMMARY_PATH = REPORTS_DIR / "ingesta_resumen.md"

# Nombres de columna observados en distintos años -> nombre estándar
TRIP_COLUMNS = {
    "viaje_id": "trip_id", "trip_id": "trip_id",
    "usuario_id": "user_id", "user_id": "user_id",
    "genero": "gender", "sex": "gender", "sexo": "gender", "gender": "gender",
    "ano_de_nacimiento": "birth_year", "anio_de_nacimiento": "birth_year",
    "birth_year": "birth_year", "a}ae_nacimiento": "birth_year",
    "inicio_del_viaje": "start_time", "trip_start": "start_time",
    "fin_del_viaje": "end_time", "trip_end": "end_time",
    "origen_id": "origin_id", "origin_id": "origin_id",
    "destino_id": "destination_id", "destination_id": "destination_id",
}
TRIP_REQUIRED = ["start_time", "end_time", "origin_id", "destination_id"]
TRIP_KEEP = ["trip_id", "gender", "birth_year", *TRIP_REQUIRED]
TRIP_DROPPED = ["user_id"]  # dato personal, no se usa en P1-P3

STATION_COLUMNS = {
    "id": "station_id", "name": "name", "nombre": "name",
    "obcn": "code", "latitude": "lat", "latitud": "lat",
    "longitude": "lon", "longitud": "lon", "status": "status",
    "capacity": "capacity", "capacidad": "capacity", "dpcapacity": "capacity", "location": "location",
}
STATION_REQUIRED = ["station_id", "lat", "lon"]

# Formatos de fecha observados en los archivos de MiBici; se elige el que convierte más filas
DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
]


# ----------------------------------------------------------------- utilidades
def month_range(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_name(col: str) -> str:
    """'Año_de_nacimiento ' -> 'ano_de_nacimiento'."""
    col = unicodedata.normalize("NFKD", str(col)).encode("ascii", "ignore").decode()
    return re.sub(r"[\s\-]+", "_", col.strip().lower().replace('"', ""))


def read_csv_any_encoding(path: Path) -> tuple[pd.DataFrame, str]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        return pd.read_csv(StringIO(text), sep=None, engine="python", dtype=str), enc
    raise ValueError(f"No se pudo leer {path}")


def choose_date_format(values: pd.Series, year: int, month: int) -> tuple[str, dict]:
    """Elige el formato de fecha que ubica más viajes dentro del mes del archivo.

    Contar solo fechas "válidas" no basta: 04/05/2026 es válida como 4 de mayo y como
    5 de abril. Como cada archivo corresponde a un mes conocido, el formato correcto es
    el que coloca los viajes en ese mes. Se usa una muestra repartida en todo el archivo.
    """
    step = max(1, len(values) // 20000)
    sample = values.dropna().iloc[::step]
    scores = {}
    for fmt in DATE_FORMATS:
        d = pd.to_datetime(sample, format=fmt, errors="coerce")
        scores[fmt] = int(((d.dt.year == year) & (d.dt.month == month)).sum())
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    if scores[best] == 0:
        raise ValueError(f"Ningún formato de fecha ubica los viajes en {year}-{month:02d}")
    if scores[ranked[1]] == scores[best]:
        raise ValueError(f"Formato de fecha ambiguo entre {best} y {ranked[1]}")
    return best, {"date_format_scores": scores, "date_sample_size": len(sample)}


def find_links() -> dict[str, str]:
    """Busca en el portal los enlaces reales a los CSV (por si cambia la carpeta)."""
    try:
        html = requests.get(MIBICI_PAGE_URL, headers=HEADERS, timeout=30).text
    except requests.RequestException as e:
        logger.warning(f"No se pudo leer el portal ({e}); se usará la URL base.")
        return {}
    links = re.findall(r'href="([^"]+\.csv)"', html)
    return {Path(l).name: requests.compat.urljoin(MIBICI_PAGE_URL, l) for l in links}


def download(name: str, links: dict[str, str]) -> Path:
    dest = RAW_DATA_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Ya existe {name}, se omite la descarga.")
        return dest
    url = links.get(name, MIBICI_FILES_BASE + name)
    logger.info(f"Descargando {url}")
    r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    r.raise_for_status()
    if "text/html" in r.headers.get("Content-Type", ""):
        raise requests.HTTPError(f"{url} devolvió una página HTML, no un CSV")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.replace(dest)  # evita dejar archivos a medias si la descarga se interrumpe
    return dest


# ------------------------------------------------------- limpieza de viajes
def normalize_trips(df: pd.DataFrame, year: int, month: int) -> tuple[pd.DataFrame, dict]:
    original_columns = list(df.columns)
    df = df.rename(columns=lambda c: TRIP_COLUMNS.get(clean_name(c), clean_name(c)))
    missing = set(TRIP_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas {missing}; columnas: {original_columns}")

    stats = {"original_columns": original_columns, "raw_rows": len(df)}
    df = df.drop(columns=[c for c in TRIP_DROPPED if c in df.columns])
    df = df[[c for c in TRIP_KEEP if c in df.columns]]

    fmt, date_info = choose_date_format(df["start_time"], year, month)
    df["start_time"] = pd.to_datetime(df["start_time"], format=fmt, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], format=fmt, errors="coerce")
    stats.update(date_format=fmt, **date_info)
    for c in ("trip_id", "origin_id", "destination_id", "birth_year"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.strip().str.upper().astype("category")

    dropped = {}
    mask = df["start_time"].isna() | df["end_time"].isna()
    dropped["fecha_invalida"] = int(mask.sum()); df = df[~mask]
    mask = df["origin_id"].isna() | df["destination_id"].isna()
    dropped["sin_estacion"] = int(mask.sum()); df = df[~mask]
    mask = df["end_time"] < df["start_time"]
    dropped["fin_antes_de_inicio"] = int(mask.sum()); df = df[~mask]
    mask = (df["start_time"].dt.year != year) | (df["start_time"].dt.month != month)
    dropped["fuera_del_mes"] = int(mask.sum()); df = df[~mask]
    if "trip_id" in df.columns:
        mask = df["trip_id"].notna() & df.duplicated("trip_id")
        dropped["duplicado"] = int(mask.sum()); df = df[~mask]

    df = df.sort_values("start_time", kind="stable").reset_index(drop=True)
    df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    stats.update(dropped=dropped, final_rows=len(df), final_columns=list(df.columns))
    logger.info(f"  {year}-{month:02d}: {len(df):,} de {stats['raw_rows']:,} filas "
                f"(formato de fecha {fmt}); descartadas: {dropped}")
    return df, stats


# --------------------------------------------------- limpieza de estaciones
def normalize_stations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_columns = list(df.columns)
    df = df.rename(columns=lambda c: STATION_COLUMNS.get(clean_name(c), clean_name(c)))
    missing = set(STATION_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Nomenclatura sin columnas {missing}; columnas: {original_columns}")

    stats = {"original_columns": original_columns, "raw_rows": len(df)}
    df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").astype("Int64")
    for c in ("lat", "lon", "capacity"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    dropped = {}
    mask = df["station_id"].isna()
    dropped["sin_id"] = int(mask.sum()); df = df[~mask]
    mask = df.duplicated("station_id")
    dropped["id_duplicado"] = int(mask.sum()); df = df[~mask]
    # Las coordenadas fuera del área se marcan, no se eliminan: la estación existe y
    # sus viajes son válidos para P1 y P2; solo P3 (ubicación) debe excluirla.
    # Las estaciones retiradas aparecen con el prefijo "ex-" en el nombre
    df["active"] = ~df["name"].astype(str).str.strip().str.lower().str.startswith("ex-")
    stats["stations_inactive"] = int((~df["active"]).sum())
    df["coords_valid"] = df["lat"].between(*GDL_LAT_RANGE) & df["lon"].between(*GDL_LON_RANGE)
    invalid = df.loc[~df["coords_valid"], "station_id"].astype(int).tolist()
    stats["stations_invalid_coords"] = invalid

    warnings = []
    if invalid:
        warnings.append(f"{len(invalid)} estaciones con coordenadas fuera del área "
                        f"metropolitana (marcadas coords_valid=False): {invalid[:20]}")
    if "capacity" not in df.columns:
        warnings.append("La nomenclatura no incluye capacidad; P2 debe aproximarla.")
    else:
        bad = int((df["capacity"].isna() | (df["capacity"] <= 0)).sum())
        if bad:
            warnings.append(f"{bad} estaciones sin capacidad válida (se conservan).")
    for w in warnings:
        logger.warning(w)
    df = df.sort_values("station_id").reset_index(drop=True)
    stats.update(dropped=dropped, warnings=warnings, final_rows=len(df),
                 final_columns=list(df.columns))
    return df, stats


# ------------------------------------------------------------------ resumen
def write_summary(m: dict) -> None:
    lines = [
        "# Resumen de la ingesta", "",
        f"- Versión de datos: `{m['data_version']}`",
        f"- Periodo solicitado: {m['start_month']} a {m['end_month']}",
        f"- Ingesta completa: {'sí' if m['complete'] else 'NO'}",
        f"- Meses faltantes: {', '.join(m['missing_months']) or 'ninguno'}",
        f"- Estaciones: {m['n_stations']}", "",
        "| Mes | Registros originales | Descartados | Registros finales | Formato de fecha |",
        "|---|---:|---:|---:|---|",
    ]
    for f in m["files"]:
        if f["name"].startswith("trips_"):
            s = f["stats"]
            lines.append(f"| {f['month']} | {s['raw_rows']:,} | {sum(s['dropped'].values()):,} "
                         f"| {s['final_rows']:,} | `{s['date_format']}` |")
    lines.append(f"| **Total** | {m['n_raw_trips']:,} | {m['n_raw_trips'] - m['n_trips']:,} "
                 f"| {m['n_trips']:,} | |")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify() -> None:
    """Recalcula los hashes de los archivos en disco y los compara con el manifiesto."""
    m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    errors = 0
    for f in m["files"]:
        for path, expected in ((RAW_DATA_DIR / f["source"], f["raw_sha256"]),
                               (PROCESSED_DATA_DIR / f["name"], f["sha256"])):
            if not path.exists():
                logger.error(f"No existe {path.name}"); errors += 1
            elif sha256(path) != expected:
                logger.error(f"Hash distinto en {path.name}"); errors += 1
    if errors:
        raise typer.Exit(1)
    logger.success(f"Los {len(m['files']) * 2} archivos coinciden con la versión {m['data_version']}.")


# ------------------------------------------------------------------ comando
def parse_month(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


@app.command()
def main(
    start: str = typer.Option(None, help="AAAA-MM"),
    end: str = typer.Option(None, help="AAAA-MM"),
    verify_only: bool = typer.Option(False, "--verify", help="Solo comprobar hashes"),
):
    if verify_only:
        return verify()
    start_m = parse_month(start) if start else START_MONTH
    end_m = parse_month(end) if end else END_MONTH
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    links = find_links()
    files, missing = [], []

    raw = download(STATIONS_FILE, links)
    st_raw, enc = read_csv_any_encoding(raw)
    st, st_stats = normalize_stations(st_raw)
    out = PROCESSED_DATA_DIR / "stations.parquet"
    st.to_parquet(out, index=False)
    files.append({"name": out.name, "source": raw.name, "encoding": enc, "rows": len(st),
                  "raw_sha256": sha256(raw), "sha256": sha256(out), "stats": st_stats})
    station_ids = set(st["station_id"])

    months = list(month_range(start_m, end_m))
    for y, m in tqdm(months, desc="meses"):
        name = TRIPS_FILE_TEMPLATE.format(year=y, month=m)
        # Se borra el Parquet previo del mes para que un fallo no deje datos de otra ejecución
        (PROCESSED_DATA_DIR / f"trips_{y}_{m:02d}.parquet").unlink(missing_ok=True)
        try:
            raw = download(name, links)
            df_raw, enc = read_csv_any_encoding(raw)
            df, stats = normalize_trips(df_raw, y, m)
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Mes {y}-{m:02d} no procesado: {e}")
            missing.append(f"{y}-{m:02d}")
            continue
        unknown = ~df["origin_id"].isin(station_ids) | ~df["destination_id"].isin(station_ids)
        stats["trips_with_unknown_station"] = int(unknown.sum())  # se conservan: estaciones dadas de baja
        out = PROCESSED_DATA_DIR / f"trips_{y}_{m:02d}.parquet"
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)
        files.append({"name": out.name, "month": f"{y}-{m:02d}", "source": raw.name,
                      "encoding": enc, "rows": len(df), "raw_sha256": sha256(raw),
                      "sha256": sha256(out), "stats": stats})

    # Solo quedan en data/processed/ los archivos de esta ejecución
    current = {f["name"] for f in files}
    for old in PROCESSED_DATA_DIR.glob("*.parquet"):
        if old.name not in current:
            logger.warning(f"Se elimina {old.name}: no pertenece a esta ingesta.")
            old.unlink()

    # La versión depende solo de los archivos de entrada y del periodo,
    # así dos ejecuciones con los mismos CSV producen el mismo identificador.
    trips = [f for f in files if f["name"].startswith("trips_")]
    key = f"{start_m}{end_m}" + "".join(f["raw_sha256"] for f in files)
    manifest = {
        "data_version": hashlib.sha256(key.encode()).hexdigest()[:12],
        "start_month": f"{start_m[0]}-{start_m[1]:02d}",
        "end_month": f"{end_m[0]}-{end_m[1]:02d}",
        "requested_months": [f"{y}-{m:02d}" for y, m in months],
        "processed_months": [f["month"] for f in trips],
        "missing_months": missing,
        "complete": not missing,
        "n_months": len(trips),
        "n_raw_trips": sum(f["stats"]["raw_rows"] for f in trips),
        "n_trips": sum(f["rows"] for f in trips),
        "n_stations": len(st),
        "kept_columns": TRIP_KEEP + ["duration_min"],
        "dropped_columns": TRIP_DROPPED,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(manifest)

    msg = (f"Datos versión {manifest['data_version']}: {manifest['n_trips']:,} viajes, "
           f"{manifest['n_months']} de {len(months)} meses, {len(st)} estaciones")
    if missing:
        logger.error(f"INGESTA INCOMPLETA. {msg}. Faltan: {', '.join(missing)}")
        raise typer.Exit(1)
    logger.success(f"{msg}. Resumen en {SUMMARY_PATH}")


def load_manifest() -> dict:
    """Devuelve el manifiesto de datos (incluye `data_version`, usado como parámetro en MLflow)."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
