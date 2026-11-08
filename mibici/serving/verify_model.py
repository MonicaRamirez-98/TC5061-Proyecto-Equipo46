"""Prueba manual P1 sin levantar FastAPI: python -m mibici.serving.verify_model."""

import json

from mibici.serving.contract import FEATURE_COLUMNS, SAMPLE_P1
from mibici.serving.model_loader import MiBiciModelService


def main():
    try:
        service = MiBiciModelService()
        result = service.predict(SAMPLE_P1)
    except Exception as exc:
        print(f"ERROR al recuperar o consultar P1: {type(exc).__name__}: {exc}")
        print("Verifica que MLflow esté activo, que el alias champion exista,")
        print("y que los artefactos estén accesibles desde este entorno.")
        raise SystemExit(1) from exc
    print("Verificación de recuperación desde Model Registry: OK")
    print("Tracking URI:", service.tracking_uri)
    print("Modelo:", service.model_name)
    print("Versión resuelta:", service.version)
    print("URI utilizada:", service.model_uri)
    print("Variables:", ", ".join(FEATURE_COLUMNS))
    print("Entrada de prueba:", json.dumps(SAMPLE_P1, ensure_ascii=False))
    print("Salida de prueba:", json.dumps({"predicciones": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
