"""Semana 7: pruebas de ingesta del proyecto MiBici (sin datos reales ni red)."""
import pandas as pd
import pytest

from mibici.dataset import (
    clean_name,
    month_range,
    normalize_stations,
    normalize_trips,
)


def test_month_range_cruza_cambio_de_anio():
    """La iteración mensual debe pasar correctamente de diciembre a enero."""
    assert list(month_range((2026, 11), (2027, 2))) == [
        (2026, 11), (2026, 12), (2027, 1), (2027, 2)
    ]


def test_clean_name_normaliza_acentos_y_espacios():
    """La estandarización debe generar un identificador comparable."""
    assert clean_name(" Año_de_nacimiento ") == "ano_de_nacimiento"
    assert clean_name("INICIO DEL VIAJE") == "inicio_del_viaje"


def test_normalize_trips_limpia_duplicados_y_elimina_dato_personal():
    """Dos viajes distintos válidos; un identificador duplicado se descarta."""
    raw = pd.DataFrame(
        {
            "Viaje_Id": [101, 101, 102],
            "Usuario_Id": [10, 10, 20],
            "Inicio_del_viaje": [
                "2026-01-05 10:00:00",
                "2026-01-05 10:00:00",
                "2026-01-05 11:00:00",
            ],
            "Fin_del_viaje": [
                "2026-01-05 10:20:00",
                "2026-01-05 10:20:00",
                "2026-01-05 11:15:00",
            ],
            "Origen_Id": [1, 1, 2],
            "Destino_Id": [2, 2, 1],
        }
    )
    clean, stats = normalize_trips(raw, 2026, 1)
    assert len(clean) == 2
    assert stats["dropped"]["duplicado"] == 1
    assert "user_id" not in clean.columns
    assert clean["duration_min"].tolist() == pytest.approx([20.0, 15.0])


def test_normalize_trips_rechaza_columnas_obligatorias_faltantes():
    """Una ingesta incompleta no debe pasar silenciosamente."""
    bad = pd.DataFrame({"Viaje_Id": [1], "Origen_Id": [1]})
    with pytest.raises(ValueError, match="Faltan columnas"):
        normalize_trips(bad, 2026, 1)


def test_normalize_stations_conserva_estacion_con_coordenadas_invalidas():
    """Una estación fuera del área se marca, y un ID duplicado se descarta."""
    raw = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "name": ["Centro", "Centro duplicado", "ex-Antigua", "Fuera"],
            "latitude": [20.67, 20.67, 20.68, 0.0],
            "longitude": [-103.35, -103.35, -103.34, 0.0],
            "dpcapacity": [20, 20, 15, 10],
        }
    )
    clean, stats = normalize_stations(raw)
    assert len(clean) == 3
    assert stats["dropped"]["id_duplicado"] == 1
    by_id = clean.set_index("station_id")
    assert bool(by_id.loc[1, "active"])
    assert not bool(by_id.loc[2, "active"])
    assert not bool(by_id.loc[3, "coords_valid"])
    assert 3 in stats["stations_invalid_coords"]


def test_normalize_stations_rechaza_columnas_obligatorias_faltantes():
    """La función debe reportar una nomenclatura insuficiente."""
    bad = pd.DataFrame({"id": [1], "name": ["Centro"]})
    with pytest.raises(ValueError, match="Nomenclatura sin columnas"):
        normalize_stations(bad)
