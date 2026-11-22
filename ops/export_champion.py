"""Exporta desde el Registry EXISTENTE el champion REAL de MiBici P1.

Uso (PowerShell, ANTES de levantar Compose):
  $env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
  python -m ops.export_champion

El destino models/compose-champion está ignorado por Git. Nunca se crea un
modelo falso para este comando ni se modifican experimentos existentes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import mlflow
import mlflow.artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/compose-champion")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    destination = Path(args.output).resolve()
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"El destino ya existe: {destination}. Usa --overwrite SOLO si deseas actualizar el exportado."
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_name = os.getenv("MIBICI_MODEL_NAME", "mibici-equipo46-p1-demanda")
    alias = os.getenv("MIBICI_MODEL_ALIAS", "champion")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", tracking_uri))
    model_uri = f"models:/{model_name}@{alias}"
    print(f"Registry existente: {tracking_uri}")
    print(f"Exportando modelo REAL: {model_uri}")

    with tempfile.TemporaryDirectory(prefix="mibici_champion_") as temp:
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=temp))
        if not (downloaded / "MLmodel").is_file():
            raise RuntimeError("MLflow no devolvió un directorio con el archivo MLmodel.")
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(downloaded, destination)

    print(f"Modelo exportado a: {destination}")
    print(f"MLmodel encontrado: {(destination / 'MLmodel').is_file()}")
    print("No añadir modelos/ ni artefactos serializados al repositorio Git.")


if __name__ == "__main__":
    main()
