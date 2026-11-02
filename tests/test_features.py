"""Semana 7: pruebas de ingeniería de variables P1/P2 y perfiles P3."""
import pandas as pd
import pytest

from mibici.features import build_station_profiles, make_hourly_panel


@pytest.fixture
def viajes_sinteticos():
    """Datos pequeños e independientes del portal de MiBici."""
    return pd.DataFrame(
        {
            "start_time": [
                "2026-01-05 10:10:00",
                "2026-01-05 11:10:00",
                "2026-01-05 12:10:00",
                "2026-01-05 13:10:00",
            ],
            "end_time": [
                "2026-01-05 10:25:00",
                "2026-01-05 11:25:00",
                "2026-01-05 12:25:00",
                "2026-01-05 13:25:00",
            ],
            "origin_id": [1, 1, 2, 1],
            "destination_id": [2, 2, 1, 2],
        }
    )


@pytest.fixture
def estaciones_sinteticas():
    return pd.DataFrame(
        {
            "station_id": [1, 2],
            "name": ["Centro", "Norte"],
            "lat": [20.67, 20.68],
            "lon": [-103.35, -103.34],
            "capacity": [20, 15],
            "active": [True, True],
            "coords_valid": [True, True],
        }
    )


def test_make_hourly_panel_objetivo_es_la_hora_siguiente(
    viajes_sinteticos, estaciones_sinteticas
):
    """El target futuro no debe confundirse con las salidas de la hora actual."""
    panel = make_hourly_panel(viajes_sinteticos, estaciones_sinteticas)
    station_1 = panel.loc[panel["station_id"] == 1].set_index("timestamp")
    t = pd.Timestamp("2026-01-05 10:00:00")
    assert station_1.loc[t, "departures_this_hour"] == 1
    assert station_1.loc[t, "target_departures_next_hour"] == pytest.approx(1.0)
    assert station_1.loc[t, "target_time"] == t + pd.Timedelta(hours=1)
    assert station_1.loc[t, "risk_threshold"] == 2
    assert pd.isna(station_1.iloc[-1]["target_departures_next_hour"])


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_make_hourly_panel_rechaza_riesgo_fuera_de_rango(
    viajes_sinteticos, fraction
):
    with pytest.raises(ValueError, match="risk_fraction"):
        make_hourly_panel(viajes_sinteticos, risk_fraction=fraction)


def test_make_hourly_panel_rechaza_viajes_vacios(viajes_sinteticos):
    with pytest.raises(ValueError, match="No hay viajes válidos"):
        make_hourly_panel(viajes_sinteticos.iloc[0:0].copy())


def test_make_hourly_panel_rechaza_columnas_faltantes(viajes_sinteticos):
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        make_hourly_panel(viajes_sinteticos.drop(columns=["end_time"]))


def test_build_station_profiles_genera_matriz_de_7_dias_por_24_horas(
    viajes_sinteticos, estaciones_sinteticas
):
    """P3 construye 168 horas de salidas y 168 de llegadas por estación."""
    profiles, metadata = build_station_profiles(
        viajes_sinteticos, estaciones_sinteticas
    )
    assert profiles.shape == (2, 336)
    assert set(profiles.index.tolist()) == {1, 2}
    assert profiles.loc[1, "departures_day0_hour10"] == pytest.approx(1.0)
    assert set(metadata["station_id"].tolist()) == {1, 2}


def test_build_station_profiles_excluye_estaciones_inactivas(
    viajes_sinteticos, estaciones_sinteticas
):
    stations = estaciones_sinteticas.copy()
    stations.loc[stations["station_id"] == 2, "active"] = False
    profiles, metadata = build_station_profiles(viajes_sinteticos, stations)
    assert list(profiles.index) == [1]
    assert metadata["station_id"].tolist() == [1]


def test_build_station_profiles_falla_si_no_hay_estaciones_validas(
    viajes_sinteticos, estaciones_sinteticas
):
    stations = estaciones_sinteticas.copy()
    stations["active"] = False
    with pytest.raises(ValueError, match="No hay estaciones activas"):
        build_station_profiles(viajes_sinteticos, stations)
