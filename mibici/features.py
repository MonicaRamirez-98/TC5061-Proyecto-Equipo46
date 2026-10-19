"""Transformaciones reutilizables para las preguntas P1, P2 y P3 de MiBici."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_hourly_panel(
    trips: pd.DataFrame,
    stations: pd.DataFrame | None = None,
    risk_fraction: float = 0.10,
) -> pd.DataFrame:
    """Crea variables horarias y objetivos para P1 y P2.

    Cada fila representa una estación y una hora observada. Para P1, el objetivo
    es el número de salidas de esa estación durante la hora siguiente, útil para
    anticipar la demanda local. Para P2, genera un proxy de presión operativa:
    el flujo neto absoluto de la hora siguiente alcanza al menos una fracción
    de la capacidad de la estación.

    Las salidas, llegadas y promedios móviles describen la hora actual y horas
    anteriores. El calendario de la hora objetivo es conocido de antemano.
    Los objetivos se desplazan una hora hacia adelante y no deben incluirse
    entre las variables predictoras. El proxy de P2 no identifica directamente
    estaciones vacías o llenas porque los datos no contienen inventario real.
    """
    required = {"start_time", "end_time", "origin_id", "destination_id"}
    missing = required - set(trips.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if not 0 < risk_fraction <= 1:
        raise ValueError("risk_fraction debe estar entre 0 y 1.")

    data = trips[list(required)].copy()
    data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")
    data["end_time"] = pd.to_datetime(data["end_time"], errors="coerce")
    data["origin_id"] = pd.to_numeric(data["origin_id"], errors="coerce")
    data["destination_id"] = pd.to_numeric(
        data["destination_id"], errors="coerce"
    )
    data = data.dropna(subset=list(required))

    if data.empty:
        raise ValueError("No hay viajes válidos para construir el panel horario.")

    data["origin_id"] = data["origin_id"].astype("int64")
    data["destination_id"] = data["destination_id"].astype("int64")
    data["start_hour"] = data["start_time"].dt.floor("h")
    data["end_hour"] = data["end_time"].dt.floor("h")

    departures = (
        data.groupby(["origin_id", "start_hour"])
        .size()
        .rename("departures")
        .reset_index()
        .rename(
            columns={
                "origin_id": "station_id",
                "start_hour": "timestamp",
            }
        )
    )
    arrivals = (
        data.groupby(["destination_id", "end_hour"])
        .size()
        .rename("arrivals")
        .reset_index()
        .rename(
            columns={
                "destination_id": "station_id",
                "end_hour": "timestamp",
            }
        )
    )

    station_ids = pd.Index(
        pd.concat(
            [data["origin_id"], data["destination_id"]],
            ignore_index=True,
        ).unique()
    )

    capacities = pd.Series(dtype="float64")
    if stations is not None and {"station_id", "capacity"}.issubset(
        stations.columns
    ):
        capacity_data = stations[["station_id", "capacity"]].copy()
        capacity_data["station_id"] = pd.to_numeric(
            capacity_data["station_id"], errors="coerce"
        )
        capacity_data["capacity"] = pd.to_numeric(
            capacity_data["capacity"], errors="coerce"
        )
        capacity_data = capacity_data.dropna(
            subset=["station_id", "capacity"]
        )
        capacity_data["station_id"] = capacity_data["station_id"].astype(
            "int64"
        )
        capacities = (
            capacity_data.drop_duplicates("station_id")
            .set_index("station_id")["capacity"]
        )
        station_ids = station_ids.union(capacities.index)

    hours = pd.date_range(
        min(data["start_hour"].min(), data["end_hour"].min()),
        max(data["start_hour"].max(), data["end_hour"].max()),
        freq="h",
    )
    panel = pd.MultiIndex.from_product(
        [station_ids.sort_values(), hours],
        names=["station_id", "timestamp"],
    ).to_frame(index=False)

    panel = panel.merge(
        departures,
        on=["station_id", "timestamp"],
        how="left",
    )
    panel = panel.merge(
        arrivals,
        on=["station_id", "timestamp"],
        how="left",
    )
    panel[["departures", "arrivals"]] = (
        panel[["departures", "arrivals"]].fillna(0).astype("int32")
    )

    if capacities.empty:
        panel["capacity"] = 1.0
    else:
        median_capacity = max(float(capacities.median()), 1.0)
        panel["capacity"] = (
            panel["station_id"].map(capacities).fillna(median_capacity)
        )
        panel["capacity"] = panel["capacity"].clip(lower=1)

    panel["risk_threshold"] = np.maximum(
        1,
        np.ceil(panel["capacity"] * risk_fraction),
    ).astype("int32")

    by_station = panel.groupby("station_id", sort=False)
    panel["departures_this_hour"] = panel["departures"]
    panel["arrivals_this_hour"] = panel["arrivals"]
    panel["departures_mean_last_24h"] = by_station["departures"].transform(
        lambda values: values.rolling(24, min_periods=1).mean()
    )
    panel["departures_mean_last_168h"] = by_station["departures"].transform(
        lambda values: values.rolling(168, min_periods=1).mean()
    )

    target_time = panel["timestamp"] + pd.Timedelta(hours=1)
    panel["target_hour"] = target_time.dt.hour
    panel["target_dayofweek"] = target_time.dt.dayofweek
    panel["target_is_weekend"] = (
        target_time.dt.dayofweek >= 5
    ).astype("int8")
    panel["target_time"] = target_time

    future = panel.groupby("station_id", sort=False)
    panel["target_departures_next_hour"] = future["departures"].shift(-1)
    panel["target_arrivals_next_hour"] = future["arrivals"].shift(-1)

    net_flow = (
        panel["target_departures_next_hour"]
        - panel["target_arrivals_next_hour"]
    )
    panel["target_risk_next_hour"] = (
        (net_flow.abs() >= panel["risk_threshold"])
        .where(net_flow.notna())
        .astype("float32")
    )

    return panel


def build_station_profiles(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye perfiles de uso para agrupar estaciones en P3.

    Cada fila de la matriz representa una estación. Sus columnas contienen
    conteos de salidas y llegadas para cada combinación de día de la semana y
    hora (7 × 24 por tipo de flujo). Estos perfiles resumen cuándo se usa cada
    estación y permiten agrupar patrones útiles para planeación y expansión.

    La función agrega únicamente los viajes recibidos en `trips`; no calcula
    objetivos futuros. Para un análisis temporal, se debe pasar solo el periodo
    permitido para ese análisis.
    """
    required_trip_columns = {
        "start_time",
        "end_time",
        "origin_id",
        "destination_id",
    }
    missing = required_trip_columns - set(trips.columns)
    if missing:
        raise ValueError(f"Faltan columnas de viajes: {sorted(missing)}")

    required_station_columns = {"station_id", "lat", "lon"}
    missing = required_station_columns - set(stations.columns)
    if missing:
        raise ValueError(f"Faltan columnas de estaciones: {sorted(missing)}")

    metadata = stations.copy()
    metadata["station_id"] = pd.to_numeric(
        metadata["station_id"], errors="coerce"
    )
    metadata["lat"] = pd.to_numeric(metadata["lat"], errors="coerce")
    metadata["lon"] = pd.to_numeric(metadata["lon"], errors="coerce")
    metadata = metadata.dropna(subset=["station_id", "lat", "lon"])

    if "active" in metadata.columns:
        metadata = metadata[metadata["active"].fillna(False).astype(bool)]
    if "coords_valid" in metadata.columns:
        metadata = metadata[
            metadata["coords_valid"].fillna(False).astype(bool)
        ]

    metadata["station_id"] = metadata["station_id"].astype("int64")
    metadata = metadata.drop_duplicates("station_id").sort_values("station_id")
    station_ids = metadata["station_id"].tolist()

    if not station_ids:
        raise ValueError("No hay estaciones activas con coordenadas válidas.")

    data = trips[
        ["start_time", "end_time", "origin_id", "destination_id"]
    ].copy()
    data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")
    data["end_time"] = pd.to_datetime(data["end_time"], errors="coerce")
    data["origin_id"] = pd.to_numeric(data["origin_id"], errors="coerce")
    data["destination_id"] = pd.to_numeric(
        data["destination_id"], errors="coerce"
    )
    data = data.dropna(
        subset=["start_time", "end_time", "origin_id", "destination_id"]
    )

    profile_columns = pd.MultiIndex.from_product(
        [range(7), range(24)],
        names=["day", "hour"],
    )
    profiles = []

    flows = [
        ("departures", "origin_id", "start_time"),
        ("arrivals", "destination_id", "end_time"),
    ]

    for prefix, station_column, time_column in flows:
        events = data[[station_column, time_column]].copy()
        events["day"] = events[time_column].dt.dayofweek
        events["hour"] = events[time_column].dt.hour

        counts = events.groupby(
            [station_column, "day", "hour"],
            observed=True,
        ).size()

        matrix = counts.unstack(["day", "hour"])
        matrix = matrix.reindex(
            index=station_ids,
            columns=profile_columns,
            fill_value=0,
        ).fillna(0)

        matrix.columns = [
            f"{prefix}_day{day}_hour{hour:02d}"
            for day, hour in matrix.columns
        ]
        profiles.append(matrix.astype("float32"))

    feature_matrix = pd.concat(profiles, axis=1)
    feature_matrix.index.name = "station_id"
    metadata = metadata.set_index("station_id").loc[feature_matrix.index]

    return feature_matrix, metadata.reset_index()