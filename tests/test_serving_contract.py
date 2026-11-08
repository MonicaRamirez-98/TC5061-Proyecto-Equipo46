"""Pruebas unitarias nuevas para el contrato P1 de la Semana 8."""

import math

import pytest

from mibici.serving.contract import FEATURE_COLUMNS, SAMPLE_P1, prepare_p1_features


def test_p1_valid_columns_order_and_types():
    frame = prepare_p1_features(dict(reversed(list(SAMPLE_P1.items()))))
    assert frame.shape == (1, 8)
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert all(str(dtype) == "float32" for dtype in frame.dtypes)


def test_p1_multiple_observations():
    frame = prepare_p1_features([SAMPLE_P1, SAMPLE_P1])
    assert len(frame) == 2


@pytest.mark.parametrize("change", [
    {"target_hour": 25},
    {"target_dayofweek": 7},
    {"target_is_weekend": 2},
    {"station_id": -1},
    {"departures_this_hour": -1},
    {"departures_mean_last_24h": math.nan},
    {"target_hour": 8.5},
    {"station_id": "estacion_1"},
])
def test_p1_rejects_invalid_values(change):
    with pytest.raises(ValueError):
        prepare_p1_features({**SAMPLE_P1, **change})


def test_p1_rejects_weekend_mismatch():
    with pytest.raises(ValueError, match="fin de semana"):
        prepare_p1_features({**SAMPLE_P1, "target_dayofweek": 6})


def test_p1_missing_and_extra():
    with pytest.raises(ValueError, match="faltan"):
        prepare_p1_features({"station_id": 1})
    with pytest.raises(ValueError, match="sobran"):
        prepare_p1_features({**SAMPLE_P1, "temperatura": 23})


def test_p1_empty_list():
    with pytest.raises(ValueError):
        prepare_p1_features([])
