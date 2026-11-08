"""Carga una versión concreta de P1 desde MLflow y realiza inferencia.

Contrato compartido con FastAPI: la API envía un dict o lista de dicts;
esta clase devuelve una lista de números reales no negativos.
"""

import os

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow import MlflowClient

from mibici.serving.contract import FEATURE_COLUMNS, prepare_p1_features

DEFAULT_MODEL_NAME = "mibici-equipo46-p1-demanda"
DEFAULT_MODEL_ALIAS = "champion"


class MiBiciModelService:
    def __init__(self, tracking_uri=None, model_name=None, alias=None, version=None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
        )
        self.model_name = model_name or os.getenv("MIBICI_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.alias = alias or os.getenv("MIBICI_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
        self.requested_version = version or os.getenv("MIBICI_MODEL_VERSION")
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", self.tracking_uri)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=registry_uri)

        # Resolvemos el alias a una versión inmutable al iniciar el servicio.
        if self.requested_version:
            selected = client.get_model_version(self.model_name, str(self.requested_version))
        else:
            selected = client.get_model_version_by_alias(self.model_name, self.alias)
        self.version = str(selected.version)
        self.model_uri = f"models:/{self.model_name}/{self.version}"
        self.model = mlflow.pyfunc.load_model(self.model_uri)

        signature = getattr(getattr(self.model, "metadata", None), "signature", None)
        if signature is None or signature.inputs is None:
            raise RuntimeError("El modelo registrado no incluye firma de entrada verificable.")
        recorded_columns = [col.name for col in signature.inputs.inputs]
        if recorded_columns != list(FEATURE_COLUMNS):
            raise RuntimeError(
                "Las características del registro no coinciden con P1. "
                f"Firma={recorded_columns}; contrato={list(FEATURE_COLUMNS)}"
            )

    def predict(self, payload):
        frame = prepare_p1_features(payload)
        raw = np.asarray(self.model.predict(frame), dtype=float).reshape(-1)
        if len(raw) != len(frame) or not np.isfinite(raw).all():
            raise RuntimeError("MLflow devolvió predicciones inválidas.")
        # La evaluación de P1 del repositorio aplica el mismo límite inferior.
        return np.maximum(raw, 0.0).astype(float).tolist()
