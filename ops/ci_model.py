"""Genera SOLO PARA LA PRUEBA DE INTEGRACIÓN EN CI un modelo sintético.

NO se usa para la prueba de carga final ni sustituye al champion real.
La prueba demuestra cableado de Compose, Registry, alias, API y predicción.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from mibici.serving.contract import FEATURE_COLUMNS, SAMPLE_P1


class SyntheticCiModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        # Predicción determinista ficticia: NO equivale al modelo entrenado.
        return np.asarray(model_input["departures_mean_last_24h"], dtype="float64")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/compose-champion")
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit("Destino no vacío; no se sobrescribirá un champion por accidente.")
    example = pd.DataFrame([SAMPLE_P1], columns=FEATURE_COLUMNS).astype("float32")
    signature = infer_signature(example, np.array([3.5], dtype="float64"))
    mlflow.pyfunc.save_model(
        path=str(folder),
        python_model=SyntheticCiModel(),
        signature=signature,
        input_example=example,
        pip_requirements=["mlflow==3.6.0", "pandas==2.3.3", "numpy>=1.26.0"],
    )
    print("Solo CI: artefacto sintético preparado; nunca usar como resultado de MiBici.")


if __name__ == "__main__":
    main()
