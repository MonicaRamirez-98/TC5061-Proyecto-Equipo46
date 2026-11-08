"""Contrato de las ocho variables reales del modelo P1 de MiBici.

Fuente: mibici/modeling/experiment_p1.py (FEATURE_COLUMNS).
Los datos describen una estación y una hora observada; el objetivo es la
cantidad de viajes iniciados en la hora siguiente.
"""

import math
from collections.abc import Mapping

import pandas as pd

FEATURE_COLUMNS = (
    "station_id",
    "departures_this_hour",
    "arrivals_this_hour",
    "departures_mean_last_24h",
    "departures_mean_last_168h",
    "target_hour",
    "target_dayofweek",
    "target_is_weekend",
)

INTEGER_COLUMNS = (
    "station_id", "departures_this_hour", "arrivals_this_hour",
    "target_hour", "target_dayofweek", "target_is_weekend",
)
NONNEGATIVE_COLUMNS = (
    "departures_this_hour", "arrivals_this_hour",
    "departures_mean_last_24h", "departures_mean_last_168h",
)

SAMPLE_P1 = {
    "station_id": 1,
    "departures_this_hour": 5,
    "arrivals_this_hour": 4,
    "departures_mean_last_24h": 3.5,
    "departures_mean_last_168h": 4.2,
    "target_hour": 9,
    "target_dayofweek": 1,
    "target_is_weekend": 0,
}


def prepare_p1_features(payload: Mapping | list[Mapping]) -> pd.DataFrame:
    """Valida y ordena entradas P1; rechaza campos extra/faltantes.

    Esta validación del módulo del modelo no sustituye a Pydantic en la API:
    el compañero a cargo del endpoint definirá un schema equivalente.
    """
    rows = [payload] if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Se requiere un objeto JSON o una lista no vacía de objetos.")

    clean = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Fila {index}: se requiere un objeto con las ocho variables.")
        missing = set(FEATURE_COLUMNS) - set(row)
        extra = set(row) - set(FEATURE_COLUMNS)
        if missing or extra:
            raise ValueError(f"Fila {index}: faltan={sorted(missing)}, sobran={sorted(extra)}")
        values = {}
        for field in FEATURE_COLUMNS:
            raw = row[field]
            # Evitamos conversiones silenciosas de texto/bool a números.
            if isinstance(raw, (str, bytes, bool)):
                raise ValueError(f"Fila {index}: {field} debe ser numérico.")
            try:
                number = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Fila {index}: {field} debe ser numérico.") from exc
            if not math.isfinite(number):
                raise ValueError(f"Fila {index}: {field} debe ser finito.")
            if field in INTEGER_COLUMNS and not number.is_integer():
                raise ValueError(f"Fila {index}: {field} debe ser un entero.")
            values[field] = number

        if values["station_id"] <= 0:
            raise ValueError(f"Fila {index}: station_id debe ser positivo.")
        if any(values[field] < 0 for field in NONNEGATIVE_COLUMNS):
            raise ValueError(f"Fila {index}: conteos y promedios no pueden ser negativos.")
        if not 0 <= values["target_hour"] <= 23:
            raise ValueError(f"Fila {index}: target_hour debe estar entre 0 y 23.")
        if not 0 <= values["target_dayofweek"] <= 6:
            raise ValueError(f"Fila {index}: target_dayofweek debe estar entre 0 y 6.")
        if values["target_is_weekend"] not in (0, 1):
            raise ValueError(f"Fila {index}: target_is_weekend debe ser 0 o 1.")
        expected_weekend = int(values["target_dayofweek"] >= 5)
        if values["target_is_weekend"] != expected_weekend:
            raise ValueError(f"Fila {index}: indicador de fin de semana inconsistente.")
        clean.append(values)

    # El entrenamiento convirtió estas ocho características a float32.
    return pd.DataFrame(clean, columns=FEATURE_COLUMNS).astype("float32")
