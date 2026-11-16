# Interpretación técnica del reporte real de Evidently | Semana 9 – Equipo 46

> Basado en las salidas `periodos_dataset.json`, `drift_por_variable.csv`, `calidad_variables.csv` y el reporte de Evidently de la ejecución con datos reales compartida por Axel. NO corresponde a la demostración sintética.

## Periodos y muestra

- Referencia: `train`, 2026-01-01 01:00 a 2026-07-11 02:00. Total 2,160,006 registros estación-hora; 10,000 muestreados.
- Actual: `test`, 2026-08-21 02:00 a 2026-10-01 00:00. Total 462,993 registros; 10,000 muestreados.
- Validación excluida de esta comparación: 2026-07-11 03:00 a 2026-08-21 02:00 (límite exclusivo).
- 471 estaciones en cada partición completa, semilla 42, datos `d038fe16493d`.
- Se analizaron 7 variables (4 numéricas y 3 categóricas). `station_id` quedó fuera del cálculo.

## Resultados

Evidently detectó drift en 4 de 7 variables (57.14 %), con umbral individual 0.10 y umbral global de fracción de variables 0.50:

| Variable | Score | Drift | Media referencia | Media actual | Cambio de media | Nulos |
|---|---:|---|---:|---:|---:|---|
| `departures_mean_last_24h` | 0.259242 | Sí | 1.0114 | 1.3448 | +33.0 % | 0 % en ambas |
| `departures_mean_last_168h` | 0.251141 | Sí | 0.9997 | 1.3039 | +30.4 % | 0 % en ambas |
| `arrivals_this_hour` | 0.151835 | Sí | 0.9835 | 1.3077 | +33.0 % | 0 % en ambas |
| `departures_this_hour` | 0.145632 | Sí | 0.9962 | 1.3053 | +31.0 % | 0 % en ambas |
| `target_dayofweek` | 0.032862 | No | 3.0501 | 2.9652 | Estadístico descriptivo, no interpretar como demanda | 0 % |
| `target_hour` | 0.030943 | No | 11.3684 | 11.4607 | Estadístico descriptivo, no interpretar como demanda | 0 % |
| `target_is_weekend` | 0.006712 | No | 0.2927 | 0.2841 | Proporción fin de semana: 29.27 % a 28.41 % | 0 % |

## Lectura técnica

1. **Cambios en la demanda observada:** Las salidas y llegadas por estación-hora aumentan en promedio cerca de un tercio en la muestra posterior. La proporción de observaciones con cero salidas cae 64.24 % a 55.55 %, y con cero llegadas, de 64.57 % a 55.85 %. El cambio sugiere mayor actividad en el periodo actual, no una causa identificada.
2. **Persistencia en ventanas móviles:** La media de salidas en 24 horas aumenta 1.0114 → 1.3448, mientras que la de 168 horas sube 0.9997 → 1.3039. Sus medianas también aumentan (0.5833 → 0.8750 y 0.6131 → 0.8690). Esto apoya que el desplazamiento aparece tanto en una ventana diaria como semanal.
3. **Calendario relativamente estable:** `target_hour`, `target_dayofweek` y `target_is_weekend` permanecen debajo del umbral y no presentan drift detectado.
4. **Calidad de muestra:** No hay nulos en las siete variables monitoreadas (20,000 filas entre ambas muestras).
5. **Limitaciones:** Diferente extensión y estacionalidad de los periodos, dependencia entre observaciones estación-hora y muestreo aleatorio no estratificado. Los scores Wasserstein/Jensen–Shannon son distancias; no son porcentajes de error ni pruebas de causalidad.

## Implicación para la siguiente etapa del equipo

El reporte **no demuestra** deterioro de MAE/RMSE ni `concept drift`. La persona responsable de la comparación debe medir ambos modelos sobre el **mismo conjunto de evaluación**. La partición `test` ya se usó aquí solo para describir distribuciones; **no debe utilizarse para entrenar, seleccionar hiperparámetros ni decidir ajustes del modelo alternativo**. Si el modelo P1 comparado es un `DummyRegressor`, las variables aquí monitoreadas describen el entorno, pero no impactan directamente sus predicciones. Verificar el estimador de la versión efectivamente registrada en MLflow antes de redactar conclusiones predictivas.

**Fuentes:** Reportes locales de Evidently generados por el equipo; Evidently AI (https://docs.evidentlyai.com/quickstart_ml); Huyen (2022), cap. 8.
