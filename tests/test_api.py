"""Pruebas de la API FastAPI con un servicio simulado (no requiere MLflow)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mlflow")

from fastapi.testclient import TestClient  # noqa: E402

import api.main as main  # noqa: E402
from api.schemas import PredictionInput  # noqa: E402
from mibici.serving.contract import FEATURE_COLUMNS, SAMPLE_P1  # noqa: E402


class FakeService:
    model_name = "mibici-equipo46-p1-demanda"
    version = "3"
    model_uri = "models:/mibici-equipo46-p1-demanda/3"

    def __init__(self):
        self.calls = []

    def predict(self, payload):
        self.calls.append(payload)
        # Igual que el servicio real: lista de floats no negativos.
        return [max(row["departures_mean_last_24h"] * 1.5 - 1.0, 0.0) for row in payload]


class BrokenService:
    def __init__(self):
        raise ConnectionError("MLflow no responde")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "MiBiciModelService", FakeService)
    with TestClient(main.app) as test_client:
        yield test_client


def test_schema_matches_model_contract():
    # Los campos de Pydantic deben ser exactamente las 8 variables del modelo.
    assert tuple(PredictionInput.model_fields) == FEATURE_COLUMNS


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "3"
    assert body["feature_columns"] == list(FEATURE_COLUMNS)


def test_predict_ok(client):
    response = client.post("/predict", json=SAMPLE_P1)
    assert response.status_code == 200
    body = response.json()
    assert body["station_id"] == SAMPLE_P1["station_id"]
    assert body["predicted_departures_next_hour"] == pytest.approx(4.25)
    assert body["model_name"] == "mibici-equipo46-p1-demanda"
    assert body["model_version"] == "3"


@pytest.mark.parametrize(
    "change",
    [
        {"target_hour": 24},
        {"target_dayofweek": 7},
        {"departures_this_hour": -1},
        {"station_id": 0},
        {"target_dayofweek": 6},  # domingo con target_is_weekend=0
        {"departures_this_hour": "5"},  # texto en lugar de número
        {"temperatura": 25},  # campo que el modelo no usa
    ],
)
def test_predict_rejects_invalid_input(client, change):
    response = client.post("/predict", json={**SAMPLE_P1, **change})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    data = {k: v for k, v in SAMPLE_P1.items() if k != "station_id"}
    assert client.post("/predict", json=data).status_code == 422


def test_batch(client):
    records = [SAMPLE_P1, {**SAMPLE_P1, "station_id": 2, "departures_mean_last_24h": 0.0}]
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [p["station_id"] for p in predictions] == [1, 2]
    assert predictions[1]["predicted_departures_next_hour"] == 0.0


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_unavailable_model_returns_503(monkeypatch):
    monkeypatch.setattr(main, "MiBiciModelService", BrokenService)
    with TestClient(main.app) as test_client:
        assert test_client.get("/health").status_code == 503
        assert test_client.post("/predict", json=SAMPLE_P1).status_code == 503
