"""Semana 7: pruebas de partición cronológica de P1 y P2."""
import numpy as np
import pandas as pd
import pytest

from mibici.modeling.train import chronological_split as split_p1
from mibici.modeling.train_p2 import chronological_split as split_p2


@pytest.fixture
def panel_sintetico():
    times = pd.date_range("2026-01-05 00:00:00", periods=24, freq="h")
    return pd.DataFrame({
        "target_time": times,
        "target_departures_next_hour": np.arange(24, dtype=float),
        "target_risk_next_hour": np.arange(24) % 2,
    })


@pytest.mark.parametrize("split_function", [split_p1, split_p2])
def test_split_mantiene_orden_y_evitar_fuga_temporal(
    split_function, panel_sintetico
):
    train, validation, test, bounds = split_function(panel_sintetico)
    assert len(train) + len(validation) + len(test) == len(panel_sintetico)
    assert not train.empty and not validation.empty and not test.empty
    assert train["target_time"].max() < validation["target_time"].min()
    assert validation["target_time"].max() < test["target_time"].min()
    assert bounds["train_target_start"] == train["target_time"].min().isoformat()


@pytest.mark.parametrize("split_function", [split_p1, split_p2])
def test_split_rechaza_periodo_demasiado_corto(split_function, panel_sintetico):
    with pytest.raises(ValueError, match="suficientes horas"):
        split_function(panel_sintetico.iloc[:9].copy())
