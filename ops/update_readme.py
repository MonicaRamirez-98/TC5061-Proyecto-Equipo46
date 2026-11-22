"""Agrega idempotentemente una sección Semana 10 al README EXISTENTE.

Nunca reescribe el historial de las semanas 7 a 9.
"""
from pathlib import Path

HEADER = "## Semana 10 — Orquestación Docker Compose y CI/CD (Entregable 7)"

SECTION = f"""
\n---\n\n{HEADER}

Responsable técnico de los puntos 1 y 2: Axel Fernando Rábago Díaz.
Pruebas de carga, recursos, FinOps y reporte: Mónica María Ramírez Mejía.

### Arquitectura

Tres servicios en `docker-compose.yml` dentro de la red interna `backend`:
`mlflow` (tracking y Model Registry con SQLite/artefactos persistentes),
`model-init` (registra el modelo P1 exportado y valida su firma) y
`api` (FastAPI con `/health`, `/model-info`, `/predict` y `/predict/batch`).
La API espera a que el Registry esté sano y la inicialización del modelo termine.
La base y los artefactos de este stack se conservan en el volumen `mlflow_data`.

**Importante:** `models/compose-champion/` se excluye de Git. Para utilizar el
modelo real debe exportarse una vez desde el Registry ORIGINAL, por lo que un
clon limpio por sí solo no incluye los pesos entrenados. Un modelo sintético
se usa exclusivamente en el smoke test del workflow de GitHub Actions.

### Arranque local (Windows / PowerShell)

1. Instalar y abrir Docker Desktop; disponer de Python 3.11 y dependencias.
2. Antes de levantar Compose, iniciar el MLflow ORIGINAL (Registro Semana 8)
   y comprobar que el alias `champion` existe y sus artefactos se descargan.
3. Exportar el modelo real mientras esté activo ese servidor original:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m ops.export_champion
```

4. **Detener el MLflow original** para liberar el puerto 5000.
5. Arrancar la solución completa desde la raíz del repositorio:

```powershell
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml up -d --build
docker compose -f docker-compose.yml ps --all
python -m ops.smoke_test --url http://localhost:8000
```

API: `http://localhost:8000/docs`. MLflow: `http://localhost:5000`.
Para revisar logs: `docker compose logs --tail 100 mlflow model-init api`.
Para detener SIN borrar modelos: `docker compose down`.
No ejecutar `docker compose down -v` salvo que se desee destruir el Registry
local y sus artefactos del volumen de Compose.

### Pipeline CI/CD

El workflow `.github/workflows/semana10-ci.yml` se ejecuta con `push`,
`pull_request` y ejecución manual. Ejecuta `pytest` sobre `tests/`,
construye la imagen OCI con caché de Buildx, valida Compose, levanta los
servicios y comprueba `/health`, `/model-info` y `/predict`. Publica como
artefactos los resultados de Pytest (JUnit) y los logs de los contenedores.
Se utiliza un modelo sintético SOLO en el smoke test de CI para evitar
publicar un modelo real en GitHub. La evidencia del rendimiento real
corresponde al análisis de Locust realizado por el equipo por separado.

El job opcional de publicación GHCR se habilita estableciendo en GitHub
`Settings > Secrets and variables > Actions > Variables` la variable
`PUBLISH_GHCR` con valor `true` y verificando permiso de publicación.
Sin esa variable, CI valida y construye la imagen, pero **no la publica**.

### Evidencias y responsabilidad

- Captura de `docker compose ps --all` con `mlflow` y `api` saludables y
  `model-init` completado.
- Captura del resultado real de `python -m ops.smoke_test`.
- Captura del workflow verde en **Actions**, con hash de commit y rama.
- PR y commits de Axel; los archivos de Locust/FinOps y sus análisis
  serán realizados y documentados por Mónica.

Guía: [`docs/Semana10_Axel_Guia_Paso_a_Paso.md`](docs/Semana10_Axel_Guia_Paso_a_Paso.md).
Notas: [`docs/Semana10_Axel_Notas_Estudio.md`](docs/Semana10_Axel_Notas_Estudio.md).
"""


def main() -> None:
    readme = Path("README.md")
    if not readme.is_file():
        raise SystemExit("No se encontró README.md. Ejecutar desde la raíz real del repositorio.")
    text = readme.read_text(encoding="utf-8")
    if HEADER in text:
        print("El README ya contiene la sección de la Semana 10; no se duplicó.")
        return
    readme.write_text(text.rstrip() + "\n" + SECTION, encoding="utf-8")
    print("README.md actualizado sin modificar las semanas anteriores.")


if __name__ == "__main__":
    main()
