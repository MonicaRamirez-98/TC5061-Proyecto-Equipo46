# Semana 9: preparación de los conjuntos para P1

## Modelo seleccionado

Se trabajará con **P1, regresión de salidas por estación para la siguiente hora**. El modelo de referencia es `mibici-equipo46-p1-demanda`, versión `1`, alias `champion`, candidato `P1-HGB-02`. Se elige porque ya está registrado en MLflow y produce una variable continua, por lo que su desempeño puede compararse con MAE y RMSE.

## Periodos

| Conjunto | Ventana de `target_time` | Uso |
|---|---|---|
| Referencia | 2026-01-01 01:00 a 2026-07-11 02:00, ambos inclusive | Reproduce exactamente las filas de entrenamiento del champion indicadas en su `training_config.json`. |
| Actual | 2026-10-01 01:00 a antes de 2026-11-01 00:00 | Representa octubre de 2026, un periodo posterior al entrenamiento y al intervalo de prueba original. Se usará para comparar la distribución de las variables y conservar la etiqueta real para evaluación retrospectiva. |

El manifiesto versionado `d038fe16493d` cubre enero–septiembre de 2026 y no reporta meses faltantes; se conserva sin cambios porque representa la fuente usada para el champion. El CSV abierto de MiBici `datos_abiertos_2026_10.csv` se incorpora como fuente suplementaria de octubre y se normaliza con las mismas reglas de la ingesta. El script registra su SHA-256, filas originales y filas conservadas en el resumen generado. Las características comparadas son las ocho entradas de P1: `station_id`, salidas y llegadas de la hora actual, promedios móviles de salidas de 24 y 168 horas, hora objetivo, día de la semana y bandera de fin de semana. `target_departures_next_hour` se guarda como etiqueta para evaluar; no se entrega a Evidently como característica de entrada del modelo.

El panel horario se construye con todo el historial disponible antes de recortar las ventanas. Así, los promedios móviles de 24 y 168 horas de octubre conservan el historial de septiembre. Se eliminan únicamente las filas sin objetivo observado, que no sirven para evaluar predicciones.

## Separación de la evaluación

El intervalo de prueba original del experimento P1 termina en `2026-10-01 00:00`; por eso, la ventana actual comienza a `2026-10-01 01:00`. Octubre queda fuera de ese test previo. El reporte de drift compara la referencia histórica con todo octubre. Para el A/B testing se reservó temporalmente la segunda mitad de octubre (`2026-10-16 00:00` a antes de `2026-11-01 00:00`) como holdout común. El challenger se entrenó con las etiquetas disponibles hasta el 15 de octubre inclusive; no se usaron las etiquetas del holdout.

## Preparación reproducible

Los archivos de viajes y los Parquet derivados no se versionan en Git. Coloca el CSV de octubre como `data/raw/datos_abiertos_2026_10.csv`. Con los Parquet históricos de enero a septiembre, el manifiesto correspondiente y ese CSV, desde la raíz del repositorio se ejecuta:

```powershell
python scripts/prepare_week9_datasets.py
```

El script guarda `data/monitoring/reference_p1_train.parquet`, `data/monitoring/current_p1_2026_10.parquet` y `reports/semana9/dataset_summary.json`. La configuración de fechas y características está en `configs/week9_datasets.json`.

## Reporte de drift y calidad (Evidently)

El reporte compara las ocho entradas de P1 entre la referencia y octubre. `station_id` y las variables de calendario se declaran categóricas; los conteos y promedios móviles se analizan como numéricos. La etiqueta real `target_departures_next_hour` y la marca temporal `target_time` se incluyen en el resumen de calidad, pero no como entradas del reporte de drift. La marca temporal permite comprobar duplicados con respecto a la identidad estación-hora. El resumen de calidad presenta faltantes, duplicados, filas/columnas vacías y columnas constantes. Las diferencias esperadas de tamaño y distribución se reportan como métricas, sin convertirlas en pruebas de calidad. Se generan un HTML visual y un JSON con resultados en `reports/semana9/`.

Con el entorno del proyecto activo, instala las dependencias de monitoreo y ejecuta:

```powershell
python -m pip install -r requirements-week9.txt
python scripts/generate_week9_drift_report.py
```

Evidently está fijado en `requirements-week9.txt` como versión 0.7.23, con Plotly 5.24.1 para compatibilidad. Los archivos de entrada Parquet son locales e ignorados por Git; primero deben prepararse según los pasos anteriores.

## Interpretación de los resultados

Evidently usa para estas variables un umbral de drift de `0.10`. Cuatro de las ocho entradas lo superan; el porcentaje agregado es 50%, justo el umbral configurado para marcar que la proporción de columnas con drift es alta.

| Variable de entrada | Método | Puntaje | Resultado | Media referencia → octubre |
|---|---|---:|---|---:|
| `station_id` | Jensen–Shannon | 0.0000 | Sin drift | 473 estaciones en ambos conjuntos |
| `departures_this_hour` | Wasserstein normalizado | 0.3444 | Drift | 1.0253 → 1.8070 |
| `arrivals_this_hour` | Wasserstein normalizado | 0.3396 | Drift | 1.0253 → 1.8069 |
| `departures_mean_last_24h` | Wasserstein normalizado | 0.5882 | Drift | 1.0221 → 1.8130 |
| `departures_mean_last_168h` | Wasserstein normalizado | 0.6185 | Drift | 1.0106 → 1.7864 |
| `target_hour` | Jensen–Shannon | 0.0023 | Sin drift | — |
| `target_dayofweek` | Jensen–Shannon | 0.0346 | Sin drift | — |
| `target_is_weekend` | Jensen–Shannon | 0.0058 | Sin drift | — |

El cambio está concentrado en la intensidad de uso: los conteos actuales de salidas y llegadas, además de sus promedios móviles, son mayores que en el entrenamiento. Las medias subieron aproximadamente entre 76% y 77%. A la vez, la cobertura de estaciones y la distribución de hora, día de semana y fin de semana se mantuvieron similares. Esto apunta a un aumento de actividad en estaciones y horarios conocidos, en vez de una incorporación de estaciones o patrones de calendario distintos.

La etiqueta real de P1 (`target_departures_next_hour`), que no se incluyó como característica en la detección, también subió: su media pasó de 1.0253 a 1.8082 salidas; la mediana, de 0 a 1, y el percentil 90, de 3 a 5. Por tanto, el modelo recibe entradas en rangos de demanda más altos y el resultado que debe predecir también cambió. Esto puede elevar el error o producir subestimaciones si el modelo no generaliza a ese nivel de demanda, en especial durante horas de alta actividad.

En las verificaciones de calidad incluidas, octubre tiene 351,439 filas, cero valores faltantes, cero filas duplicadas, cero filas vacías, cero columnas vacías y cero columnas constantes. No se observan señales de problemas básicos de integridad que expliquen por sí solos el cambio detectado.

El drift es una señal para investigar, no una demostración de pérdida de desempeño: un cambio de distribución no siempre reduce la calidad del modelo ([Evidently, Data and prediction drift](https://learn.evidentlyai.com/ml-observability-course/module-2-ml-monitoring-metrics/data-prediction-drift-in-ml)). La comparación del champion y el challenger sobre las mismas filas de octubre se presenta en el documento de A/B testing. Para decidir Continuous Training, el análisis se limitará a vacaciones, días festivos, contingencias del transporte público, cambios de tarifa, verano y fin de año. Estos contextos no están registrados en los conjuntos analizados ni son entradas del modelo actual; por tanto, el reporte presente no permite medir su efecto por separado. Se recomienda documentar sus fechas con fuentes verificables y evaluar el error del modelo dentro de esos periodos antes de activar un nuevo reentrenamiento.

## Trazabilidad

- Modelo y límites de entrenamiento: `training_config.json` del run `731a0bf25cdd40e5b48a886a03a08aa0` en el MLflow local.
- Periodos de ingesta y versión de datos: `data/manifest/data_manifest.json`.
- Fuente de los datos abiertos: [MiBici, Datos abiertos](https://mibici.net/es/datos-abiertos/).
