# Semana 8 — Recuperación del modelo desde MLflow (Axel)

## Objetivo de mi contribución

Me corresponde recuperar el modelo de demanda **P1** desde el Model Registry, validar sus ocho variables reales y entregar una función de inferencia que pueda integrar el equipo con FastAPI. Conservamos el modelo entrenado en semanas anteriores: **no se reentrena ni se modifican sus métricas**.

## Modelo y decisión técnica

- Nombre: `mibici-equipo46-p1-demanda`
- Alias: `champion`
- Algoritmo seleccionado por el equipo: HistGradientBoostingRegressor (P1-HGB-02).
- Selección original: menor MAE de validación.
- Referencia: `models:/mibici-equipo46-p1-demanda@champion`.
- La aplicación resuelve el alias al arrancar y conserva la versión concreta durante su ejecución.
- Se utiliza `mlflow.pyfunc.load_model` para que el servicio no dependa de los detalles internos del estimador.
- Los datos de entrada se convierten a `float32`, tal como se hizo al entrenar P1.
- La salida se restringe a valores no negativos como en la evaluación original de P1.

## Variables del contrato (sin inventar características)

| Nombre | Qué representa | Regla |
|---|---|---|
| `station_id` | Identificador de estación | Entero positivo |
| `departures_this_hour` | Viajes que salieron en la hora observada | Entero ≥ 0 |
| `arrivals_this_hour` | Viajes que llegaron en la hora observada | Entero ≥ 0 |
| `departures_mean_last_24h` | Media histórica de salidas, ventana de 24 h | Número ≥ 0 |
| `departures_mean_last_168h` | Media histórica de salidas, ventana de 168 h | Número ≥ 0 |
| `target_hour` | Hora objetivo (la siguiente) | Entero 0–23 |
| `target_dayofweek` | Día de la semana objetivo (lunes = 0) | Entero 0–6 |
| `target_is_weekend` | Indicador sábado o domingo | 0 o 1 |

**Importante:** No se envían temperatura ni fecha en texto. Las medias de 24/168 horas se calculan previamente con el historial de la estación; `station_id` y el calendario, por sí solos, no bastan para crear estas características.

## Cómo probar mi parte (Windows PowerShell)

Desde la raíz del proyecto, con `.venv` activado, dependencias instaladas y el servidor MLflow funcionando:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m pytest tests/test_serving_contract.py -v
python -m mibici.serving.verify_model
```

El segundo comando requiere el registro **y los artefactos existentes en la misma máquina**. No basta con clonar Git: `.gitignore` excluye la base SQLite y los artefactos locales.

Si `champion` no existe y el equipo ya tiene los datos procesados, se deberá reproducir ingesta, experimentos y registro de P1 antes de verificar el modelo. El script `mibici.modeling.registry` actual registra **P1, P2 y P3**, por lo que deben existir las tres búsquedas antes de ejecutarlo sin modificaciones.

## Contrato para el compañero de FastAPI

```python
from mibici.serving.model_loader import MiBiciModelService

service = MiBiciModelService()  # cargar una vez, no por solicitud
resultado = service.predict({
    "station_id": 1,
    "departures_this_hour": 5,
    "arrivals_this_hour": 4,
    "departures_mean_last_24h": 3.5,
    "departures_mean_last_168h": 4.2,
    "target_hour": 9,
    "target_dayofweek": 1,
    "target_is_weekend": 0,
})
# resultado: lista de un float no negativo; el número real depende del modelo.
```

La API será responsable de convertir errores de validación a respuestas HTTP 422 y fallos internos a respuestas 5xx, sin devolver detalles sensibles. El responsable de Pydantic replicará los rangos de esta tabla. **No volver a llamar a MLflow en cada solicitud.**

## Integración con el contenedor

El servicio lee `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI` (opcional), `MIBICI_MODEL_NAME`, `MIBICI_MODEL_ALIAS` y `MIBICI_MODEL_VERSION` (opcional). Dentro de un contenedor, `localhost` **no** señala el equipo anfitrión. Para Docker/Podman en Windows puede requerirse `http://host.containers.internal:5000` u otro nombre accesible; comprobar comunicación real con MLflow y ubicación de artefactos.

Si el registro y los artefactos están guardados solo localmente y no son accesibles desde el contenedor, se debe decidir con el responsable de Docker entre configurar el servidor de artefactos o exportar el modelo registrado e incorporarlo a la imagen (documentando procedencia y versión). No afirmar portabilidad hasta ejecutar la prueba de inferencia **dentro** del contenedor.

## Evidencias a integrar antes del Pull Request

1. Captura de MLflow que muestre nombre, alias y versión de P1.
2. Terminal con la ejecución real de `python -m mibici.serving.verify_model` y una predicción.
3. Terminal con las pruebas unitarias de contrato aprobadas.
4. Pull Request propio y pruebas de integración del equipo.

No se incluyen capturas simuladas: los resultados deben obtenerse del equipo que ejecuta el proyecto.

## Observación de reproducibilidad del profesor (Semana 6)

Se reportó que, después de clonar el proyecto y seguir el README, fallaba el comando de entrenamiento P1. Aún no se dispone del *traceback*, por lo que no está confirmada una causa única.

**Comprobaciones propuestas:**

1. Usar `main` actualizado y `python --version`, `mlflow --version`.
2. Crear un entorno aislado, instalar `requirements.txt` y `pip install -e .`.
3. Iniciar MLflow en una terminal; exportar `MLFLOW_TRACKING_URI` en otra.
4. Ejecutar `mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo46` y `python -m mibici.dataset --verify`.
5. Probar con parámetros predeterminados, sin transportar JSON por la terminal:

```powershell
mlflow run . -e train -P question=P1 -P model=dummy_regressor --env-manager local --experiment-name mibici-equipo46
```

6. Si se necesita especificar el JSON, usar un archivo versionado para evitar ambigüedad de comillas en PowerShell:

```powershell
mlflow run . -e train -P question=P1 -P model=dummy_regressor -P hyperparameters=@configs/empty.json --env-manager local --experiment-name mibici-equipo46
```

7. Conservar el mensaje de error completo y contrastar la versión de Python con `python_env.yaml`. Revisar además que el manifiesto y los datos procesados coincidan; los archivos Parquet y los artefactos no se comparten mediante Git.

**Criterio de cierre:** demostrar en un clon limpio que la ingesta y P1 terminan y que la ejecución deja métricas comparables. No basta con aprobar pruebas unitarias.
