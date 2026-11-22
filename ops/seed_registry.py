"""Importa el champion exportado al Registry aislado de Docker Compose.

No reentrena el modelo. Registra una copia en MLflow del entorno Compose
para evitar depender de rutas C:\\... del MLflow original de Windows.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np

from mibici.serving.contract import FEATURE_COLUMNS, SAMPLE_P1


def _sha256_dir(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _verify_model(model_uri: str) -> None:
    model = mlflow.pyfunc.load_model(model_uri)
    signature = getattr(getattr(model, "metadata", None), "signature", None)
    if signature is None or signature.inputs is None:
        raise RuntimeError("El modelo no tiene firma de entrada verificable (Semana 8).")
    columns = [col.name for col in signature.inputs.inputs]
    if columns != list(FEATURE_COLUMNS):
        raise RuntimeError(f"Firma incorrecta: {columns} vs {list(FEATURE_COLUMNS)}")
    from mibici.serving.contract import prepare_p1_features

    prediction = np.asarray(model.predict(prepare_p1_features(SAMPLE_P1)), dtype=float).reshape(-1)
    if len(prediction) != 1 or not np.isfinite(prediction).all():
        raise RuntimeError("El champion importado no produce una predicción válida.")
    print(f"Champion comprobado. Predicción de muestra: {prediction[0]:.4f}")


def main() -> None:
    folder = Path(os.getenv("CHAMPION_EXPORT_DIR", "/model-export"))
    if not (folder / "MLmodel").is_file():
        raise RuntimeError(
            "No se encontró /model-export/MLmodel. "
            "Primero exporta TU champion real con 'python -m ops.export_champion' "
            "desde el MLflow antiguo. Consulta docs/Semana10_Axel_Guia_Paso_a_Paso.md."
        )
    fingerprint = _sha256_dir(folder)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MIBICI_MODEL_NAME", "mibici-equipo46-p1-demanda")
    alias = os.getenv("MIBICI_MODEL_ALIAS", "champion")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", tracking_uri))
    client = MlflowClient()

    # Si ya se sembró EXACTAMENTE el mismo artefacto, no duplicar versiones.
    try:
        current = client.get_model_version_by_alias(model_name, alias)
    except mlflow.exceptions.MlflowException as exc:
        if "RESOURCE_DOES_NOT_EXIST" not in str(exc) and "not found" not in str(exc).lower():
            raise
    else:
        if current.tags.get("export_sha256") == fingerprint:
            print(f"Modelo existente {model_name} v{current.version}: mismo SHA-256; reutilizando.")
            _verify_model(f"models:/{model_name}/{current.version}")
            return
        print("El exportado cambió; se registrará una nueva versión del mismo modelo.")

    # Se publican los archivos (incluido MLmodel) como artefactos en el servidor
    # Compose, de modo que la API pueda recuperarlos por la red interna.
    with mlflow.start_run(run_name="semana10_importacion_champion") as run:
        mlflow.log_artifacts(str(folder), artifact_path="champion_importado")
        source = mlflow.get_artifact_uri("champion_importado")
        mlflow.set_tag("semana10.imported_from_export", "true")
        mlflow.set_tag("semana10.export_sha256", fingerprint)
        run_id = run.info.run_id

    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException as exc:
        if "RESOURCE_DOES_NOT_EXIST" not in str(exc) and "not found" not in str(exc).lower():
            raise
        client.create_registered_model(model_name)

    version = client.create_model_version(name=model_name, source=source, run_id=run_id)
    client.set_model_version_tag(model_name, version.version, "export_sha256", fingerprint)
    _verify_model(f"models:/{model_name}/{version.version}")
    client.set_registered_model_alias(model_name, alias, version.version)
    print(f"Registro listo: models:/{model_name}@{alias}, versión {version.version}.")
    print(f"SHA-256 de artefactos exportados: {fingerprint}")


if __name__ == "__main__":
    main()
