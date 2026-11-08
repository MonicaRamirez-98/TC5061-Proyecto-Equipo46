"""Prueba real opcional: necesita un MLflow local con alias champion.

Activación: $env:RUN_MLFLOW_INTEGRATION = "1"
"""

import os
import pytest


@pytest.mark.skipif(
    os.getenv("RUN_MLFLOW_INTEGRATION") != "1",
    reason="Requiere acceso a un Model Registry real (ejecución explícita)",
)
def test_load_p1_champion_and_predict():
    from mibici.serving.contract import SAMPLE_P1
    from mibici.serving.model_loader import MiBiciModelService

    service = MiBiciModelService()
    values = service.predict(SAMPLE_P1)
    assert service.version
    assert len(values) == 1
    assert values[0] >= 0
