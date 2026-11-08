"""Pruebas unitarias del adaptador MLflow sin requerir un servidor activo."""

from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")
from mibici.serving import model_loader
from mibici.serving.contract import FEATURE_COLUMNS, SAMPLE_P1


def test_alias_resolves_to_concrete_version_and_prediction(monkeypatch):
    calls = {}

    def fake_client(**kwargs):
        calls["config"] = kwargs
        return SimpleNamespace(
            get_model_version_by_alias=lambda name, alias: SimpleNamespace(version="3")
        )

    signature = SimpleNamespace(
        inputs=SimpleNamespace(inputs=[SimpleNamespace(name=name) for name in FEATURE_COLUMNS])
    )

    class FakeModel:
        metadata = SimpleNamespace(signature=signature)

        def predict(self, frame):
            calls["columns"] = list(frame.columns)
            return [-0.1]  # Se aplica la misma restricción que en evaluación.

    monkeypatch.setattr(model_loader, "MlflowClient", fake_client)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", lambda value: None)
    monkeypatch.setattr(model_loader.mlflow, "set_registry_uri", lambda value: None)
    monkeypatch.setattr(
        model_loader.mlflow.pyfunc,
        "load_model",
        lambda uri: calls.setdefault("model_uri", uri) and FakeModel(),
    )

    service = model_loader.MiBiciModelService(tracking_uri="http://localhost:5000")
    assert service.version == "3"
    assert calls["model_uri"] == "models:/mibici-equipo46-p1-demanda/3"
    assert service.predict(SAMPLE_P1) == [0.0]
    assert calls["columns"] == list(FEATURE_COLUMNS)


def test_rejects_mismatched_mlflow_signature(monkeypatch):
    signature = SimpleNamespace(
        inputs=SimpleNamespace(inputs=[SimpleNamespace(name="temperatura")])
    )
    bad_model = SimpleNamespace(metadata=SimpleNamespace(signature=signature))
    monkeypatch.setattr(
        model_loader, "MlflowClient", lambda **kwargs: SimpleNamespace(
            get_model_version_by_alias=lambda name, alias: SimpleNamespace(version="1")
        ),
    )
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", lambda value: None)
    monkeypatch.setattr(model_loader.mlflow, "set_registry_uri", lambda value: None)
    monkeypatch.setattr(model_loader.mlflow.pyfunc, "load_model", lambda uri: bad_model)
    with pytest.raises(RuntimeError, match="no coinciden"):
        model_loader.MiBiciModelService(tracking_uri="http://localhost:5000")
