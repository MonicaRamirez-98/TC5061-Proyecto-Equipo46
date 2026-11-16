# Semana 9 – Evidently: data drift de MiBici P1

## Periodos utilizados y justificación

Se utiliza la misma lógica temporal 70/15/15 del entrenamiento P1: referencia = entrenamiento (primer 70 % de horas objetivo), actual = prueba (último 15 %). El 15 % intermedio es validación.
Esta comparación es **retrospectiva**, no monitoreo de tráfico en vivo.

- Referencia: 2026-01-01 01:00:00 a 2026-07-11 02:00:00
- Actual: 2026-08-21 02:00:00 a 2026-10-01 00:00:00
- Registros originales: 2160006 y 462993
- Muestras Evidently: 10000 y 10000
- Versión de datos: d038fe16493d
- Muestreo: aleatorio uniforme sin reemplazo, semilla fija (muestra representativa de filas, no de meses).

## Variables examinadas

- `departures_this_hour`: wasserstein (umbral 0.1).
- `arrivals_this_hour`: wasserstein (umbral 0.1).
- `departures_mean_last_24h`: wasserstein (umbral 0.1).
- `departures_mean_last_168h`: wasserstein (umbral 0.1).
- `target_hour`: jensenshannon (umbral 0.1).
- `target_dayofweek`: jensenshannon (umbral 0.1).
- `target_is_weekend`: jensenshannon (umbral 0.1).

Se excluyen expresamente objetivos futuros (`target_*_next_hour`), identificadores (`station_id`) y marcas de tiempo del análisis estadístico. `target_hour`, `target_dayofweek` y `target_is_weekend` son características de calendario conocidas antes de la hora objetivo.

## Resultados de Evidently por variable

| Variable | Método | Score | Umbral | Drift |
|---|---|---:|---:|---|
| departures_this_hour | wasserstein | 0.1456 | 0.10 | Sí |
| arrivals_this_hour | wasserstein | 0.1518 | 0.10 | Sí |
| departures_mean_last_24h | wasserstein | 0.2592 | 0.10 | Sí |
| departures_mean_last_168h | wasserstein | 0.2511 | 0.10 | Sí |
| target_hour | jensenshannon | 0.0309 | 0.10 | No |
| target_dayofweek | jensenshannon | 0.0329 | 0.10 | No |
| target_is_weekend | jensenshannon | 0.0067 | 0.10 | No |

**Variables con drift (muestra analizada):** 4 de 7.

## Interpretación responsable

- El drift estadístico no demuestra por sí solo degradación del desempeño.
- Revisar las diferencias de medias, ceros y faltantes en `calidad_variables.csv`.
- El P1 baseline público es `DummyRegressor` sin variables predictoras: estos indicadores describen el entorno y las posibles entradas para modelos futuros; no se puede afirmar que cada variable modifique directamente sus predicciones.
- Para demostrar impacto en desempeño, comparar modelos contra valores observados sobre el mismo periodo de prueba (Persona 3).
- Para la conclusión, examinar cuáles variables tienen drift, dirección del cambio y su significado operativo para las estaciones y las horas de demanda.

## Archivos de evidencia

- `evidently_data_drift.html`: análisis de drift y gráficas de Evidently.
- `evidently_calidad_datos.html`: calidad y estadísticas de los datos.
- `calidad_variables.csv`: registros, valores nulos, medias y ceros.
- `periodos_dataset.json`: datos y cortes exactos de la ejecución.
- `evidently_resultado_drift.json`: resultados estructurados de Evidently.

## Fuentes

- Evidently. Documentación oficial: https://docs.evidentlyai.com/quickstart_ml
- Código propio del Equipo 46: `mibici/features.py` y `mibici/modeling/train.py`.
