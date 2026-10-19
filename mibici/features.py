"""Variables horarias para los baselines MiBici."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_hourly_panel(
    trips: pd.DataFrame,
    stations: pd.DataFrame | None = None,
    risk_fraction: float = 0.10,
) -> pd.DataFrame:
    """Crea filas por estación y hora con objetivos para P1 y P2.

    P1: salidas de la estación durante la hora siguiente.
    P2: proxy de presión cuando el flujo neto absoluto alcanza al menos
    risk_fraction de la capacidad de la estación, con un mínimo de un viaje.

    El proxy no indica directamente si una estación quedó vacía o llena:
    los viajes no contienen el inventario real de bicicletas.
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
    data["destination_id"] = pd.to_numeric(data["destination_id"], errors="coerce")
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
        .rename(columns={"origin_id": "station_id", "start_hour": "timestamp"})
    )
    arrivals = (
        data.groupby(["destination_id", "end_hour"])
        .size()
        .rename("arrivals")
        .reset_index()
        .rename(columns={"destination_id": "station_id", "end_hour": "timestamp"})
    )

    station_ids = pd.Index(
        pd.concat(
            [data["origin_id"], data["destination_id"]],
            ignore_index=True,
        ).unique()
    )

    capacities = pd.Series(dtype="float64")
    if stations is not None and {"station_id", "capacity"}.issubset(stations.columns):
        capacity_data = stations[["station_id", "capacity"]].copy()
        capacity_data["station_id"] = pd.to_numeric(
            capacity_data["station_id"], errors="coerce"
        )
        capacity_data["capacity"] = pd.to_numeric(
            capacity_data["capacity"], errors="coerce"
        )
        capacity_data = capacity_data.dropna(subset=["station_id", "capacity"])
        capacity_data["station_id"] = capacity_data["station_id"].astype("int64")
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

    panel = panel.merge(departures, on=["station_id", "timestamp"], how="left")
    panel = panel.merge(arrivals, on=["station_id", "timestamp"], how="left")
    panel[["departures", "arrivals"]] = (
        panel[["departures", "arrivals"]].fillna(0).astype("int32")
    )

    if capacities.empty:
        median_capacity = 1.0
        panel["capacity"] = median_capacity
    else:
        median_capacity = max(float(capacities.median()), 1.0)
        panel["capacity"] = panel["station_id"].map(capacities).fillna(median_capacity)
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
    panel["target_is_weekend"] = (target_time.dt.dayofweek >= 5).astype("int8")
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