"""Pruebas unitarias de la preparación y análisis del monitoreo P1.

Estas pruebas no requieren Evidently; la generación HTML se valida por separado
al ejecutar el módulo con --demo, una vez instalada la dependencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibici.monitoring.drift import (
    FEATURES,
    THRESHOLD,
    parse_evidently_drift,
    quality_summary,
    sample_period,
    select_chronological_periods,
)


def fake_panel(hours: int = 40, n_stations: int = 2) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01 01:00:00")
    for i in range(hours):
        t = start + pd.Timedelta(hours=i)
        for station in range(n_stations):
            rows.append({
                "station_id": station + 1,
                "target_time": t,
                "target_departures_next_hour": float(i % 4),
                "departures_this_hour": float(i % 5),
                "arrivals_this_hour": float(i % 3),
                "departures_mean_last_24h": i / 24,
                "departures_mean_last_168h": i / 168,
                "target_hour": int(t.hour),
                "target_dayofweek": int(t.dayofweek),
                "target_is_weekend": int(t.dayofweek >= 5),
            })
    return pd.DataFrame(rows)


def test_temporal_split_matches_p1_and_no_leakage():
    ref, cur, meta = select_chronological_periods(fake_panel())
    assert len(ref) == 40 * 0.7 * 2
    assert len(cur) == 40 * 0.15 * 2
    assert ref["target_time"].max() < cur["target_time"].min()
    assert "target_departures_next_hour" not in ref.columns
    assert "target_departures_next_hour" not in cur.columns
    assert set(FEATURES).issubset(ref.columns)
    assert meta["criterio"].startswith("split_cronologico")


def test_sampling_is_reproducible():
    frame = fake_panel(hours=160)
    a = sample_period(frame, 120, 42)
    b = sample_period(frame, 120, 42)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 120
    assert not a.duplicated(["station_id", "target_time"]).any()


def test_invalid_sample_size():
    with pytest.raises(ValueError):
        sample_period(fake_panel(), 99, 42)


def test_summary_reports_nulls_and_periods():
    ref = fake_panel().iloc[:10].copy()
    cur = fake_panel().iloc[10:20].copy()
    cur.loc[cur.index[:2], "arrivals_this_hour"] = np.nan
    summary = quality_summary(ref, cur)
    assert len(summary) == len(FEATURES) * 2
    match = summary.query("variable == 'arrivals_this_hour' and periodo == 'actual'")
    assert match.iloc[0]["nulos"] == 2
    assert match.iloc[0]["porcentaje_nulos"] == pytest.approx(20.0)


def test_parser_reads_metric_data_from_evidently():
    result = {"metrics": [
        {"metric_name": "ValueDrift(column=departures_this_hour,method=wasserstein,threshold=0.1)", "value": 0.14},
        {"metric_name": "ValueDrift(column=target_hour,method=jensenshannon,threshold=0.1)", "value": 0.02},
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 1, "share": 0.2}},
    ]}
    parsed = parse_evidently_drift(result).set_index("variable")
    assert bool(parsed.loc["departures_this_hour", "drift_detectado"])
    assert not bool(parsed.loc["target_hour", "drift_detectado"])
    assert parsed.loc["departures_this_hour", "umbral"] == THRESHOLD


def test_fails_if_model_panel_missing_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        select_chronological_periods(pd.DataFrame({"target_time": [pd.Timestamp("2026-01-01")]}))
