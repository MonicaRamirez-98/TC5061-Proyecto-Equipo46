"""Configuración central del proyecto: rutas, rango de fechas, semilla y URLs.

Todos los módulos del proyecto leen estos valores desde aquí.
"""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# --- Rutas ------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MANIFEST_PATH = DATA_DIR / "manifest" / "data_manifest.json"  # versionado en Git
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Rango de fechas (año, mes) inclusive ----------------------------------
START_MONTH = (2026, 1)
END_MONTH = (2026, 9)

# --- Validación de estaciones ---------------------------------------------
# Caja geográfica del Área Metropolitana de Guadalajara (Guadalajara, Zapopan,
# Tlaquepaque y Tonalá) con margen de ~10 km. Las estaciones fuera de ella se
# marcan con coordenadas inválidas (p. ej. 0,0 o lat/lon invertidas), no se eliminan.
GDL_LAT_RANGE = (20.50, 20.80)
GDL_LON_RANGE = (-103.50, -103.20)

# --- Reproducibilidad -------------------------------------------------------
RANDOM_SEED = 42

# --- Fuentes MiBici ---------------------------------------------------------
MIBICI_PAGE_URL = "https://mibici.net/es/datos-abiertos/"
MIBICI_FILES_BASE = "https://mibici.net/site/assets/files/1118/"
TRIPS_FILE_TEMPLATE = "datos_abiertos_{year}_{month:02d}.csv"
STATIONS_FILE = "nomenclatura_2026_09.csv"

# --- MLflow -----------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "mibici-equipo46"  # experimento del equipo en MLflow

PROJECT_TRAIN_DEFAULTS = {
    "p1": {
        "model": "dummy_regressor",
        "hyperparameters": {"strategy": "mean"},
    },
    "p2": {
        "model": "dummy_classifier",
        "hyperparameters": {
            "strategy": "most_frequent",
            "risk_fraction": 0.10,
        },
    },
    "p3": {
        "model": "kmeans",
        "hyperparameters": {
            "n_clusters": 3,
            "n_init": 10,
            "max_iter": 300,
        },
    },
}

logger.info(f"PROJ_ROOT: {PROJ_ROOT}")
