# TC5061 – Semana 9 | MiBici, Equipo 46
## Secciones 1–3: Preparación de datos, reporte de data drift e interpretación

**Integrante responsable:** Axel Rábago
**Modelo/problema:** P1 – Predicción de demanda de MiBici
**Herramienta:** Evidently (ejecución confirmada con versión 0.7.23)
**Fuente:** Datos abiertos de MiBici procesados por el repositorio del Equipo 46
**Versión verificada de los datos:** `d038fe16493d`
**Naturaleza del ejercicio:** Monitoreo retrospectivo de datos; no es monitoreo de tráfico de una API en tiempo real.

## 1. Preparación de los conjuntos de datos

Para evaluar cambios en los patrones de utilización de MiBici, se utilizaron los archivos históricos procesados del repositorio del Equipo 46. La verificación de integridad (`python -m mibici.dataset --verify`) confirmó que **20 archivos coinciden con el manifiesto de la versión `d038fe16493d`**. Los registros de viajes utilizados se distribuyen en nueve archivos mensuales Parquet, de enero a septiembre de 2026.

El procesamiento reconstruyó las observaciones por **estación y hora**, siguiendo la estrategia de separación cronológica **70/15/15** del proyecto. Para la ejecución documentada se comparó la partición inicial de entrenamiento (referencia) con la partición final de prueba (actual); la validación intermedia no se incluyó en el monitoreo.

| Característica | Referencia: entrenamiento | Actual: prueba final |
|---|---|---|
| Periodo | 01/01/2026 01:00 a 11/07/2026 02:00 | 21/08/2026 02:00 a 01/10/2026 00:00 |
| Registros completos | 2,160,006 | 462,993 |
| Registros muestreados para Evidently | 10,000 | 10,000 |
| Estaciones identificadas en el conjunto completo | 471 | 471 |

La partición de validación se encuentra entre el **11/07/2026 03:00** y el **21/08/2026 02:00 (límite exclusivo)**. El muestreo fue aleatorio uniforme, sin reemplazo y con semilla **42**, para facilitar la reproducción de los resultados. Es importante señalar que esta muestra se seleccionó por registros estación-hora, no mediante un muestreo estratificado por estación o mes.

**Justificación.** La separación por tiempo evita comparar registros mezclados de un mismo horizonte cronológico y permite observar diferencias entre condiciones anteriores y posteriores del sistema. La referencia representa el comportamiento histórico disponible durante el entrenamiento; la partición final representa un periodo posterior. Esta selección debe comunicarse al integrante que realizará la comparación de modelos: **los datos de prueba pueden utilizarse para la evaluación final, pero no para entrenar ni ajustar el modelo alternativo o elegir hiperparámetros**, con el fin de limitar la filtración de información de evaluación.

## 2. Reporte de data drift con Evidently

Se ejecutó Evidently sobre dos muestras de **10,000 observaciones** cada una. Se empleó `DataDriftPreset` para comparar distribuciones y `DataSummaryPreset` para examinar la calidad de los datos. El reporte disponible analiza **siete variables**: cuatro numéricas y tres categóricas de calendario. La columna `station_id` no se incluyó en el cálculo estadístico de esta ejecución; se utilizó únicamente para identificar estaciones y construir las observaciones.

Para las variables numéricas se aplicó la **distancia de Wasserstein normalizada** y para las categóricas la **distancia de Jensen–Shannon**. Se configuró un **umbral de 0.10** para señalar drift por variable. El reporte consideró drift global si al menos el **50 %** de las columnas analizadas lo presentaban.

| Variable | Método | Puntuación | Umbral | ¿Drift? |
|---|---|---:|---:|---|
| `departures_mean_last_24h` | Wasserstein normalizada | 0.259242 | 0.10 | Sí |
| `departures_mean_last_168h` | Wasserstein normalizada | 0.251141 | 0.10 | Sí |
| `arrivals_this_hour` | Wasserstein normalizada | 0.151835 | 0.10 | Sí |
| `departures_this_hour` | Wasserstein normalizada | 0.145632 | 0.10 | Sí |
| `target_dayofweek` | Jensen–Shannon | 0.032862 | 0.10 | No |
| `target_hour` | Jensen–Shannon | 0.030943 | 0.10 | No |
| `target_is_weekend` | Jensen–Shannon | 0.006712 | 0.10 | No |

**Resultado general:** Evidently detectó drift en **4 de 7 variables (57.14 %)**. Por tanto, también activó la señal de drift global al superar el umbral del 50 % de columnas afectadas.

El reporte de calidad también mostró **0 valores nulos en las siete variables de las muestras comparadas**. Este resultado confirma la ausencia de nulos en las muestras analizadas; no demuestra por sí solo que todos los registros históricos estén libres de errores.

**Figuras que deben integrarse desde la ejecución real:**
**Figura 1.** Captura del resumen de `evidently_data_drift.html` con el resultado 4 de 7.
**Figura 2.** Distribuciones comparadas de `departures_mean_last_24h` desplegadas al expandir la fila correspondiente en Evidently.
**Figura 3 (opcional).** Distribuciones comparadas de `departures_mean_last_168h` o de `arrivals_this_hour`.
*Fuente de las figuras: elaboración propia mediante Evidently a partir del conjunto procesado por el Equipo 46.*

## 3. Interpretación de resultados e implicaciones

El principal hallazgo es que los cambios se concentran en las **variables de actividad de bicicletas**, mientras que las variables asociadas con el calendario muestran distribuciones relativamente estables para el umbral configurado.

### 3.1. Variables con drift

| Variable | Promedio referencia | Promedio actual | Cambio relativo | Registros con valor cero: referencia → actual |
|---|---:|---:|---:|---:|
| `departures_mean_last_24h` | 1.0114 | 1.3448 | +33.0 % | 21.01 % → 1.36 % |
| `departures_mean_last_168h` | 0.9997 | 1.3039 | +30.4 % | 20.55 % → 0.87 % |
| `arrivals_this_hour` | 0.9835 | 1.3077 | +33.0 % | 64.57 % → 55.85 % |
| `departures_this_hour` | 0.9962 | 1.3053 | +31.0 % | 64.24 % → 55.55 % |

**Promedio móvil de salidas de las últimas 24 horas.** Presenta la puntuación más alta de drift (**0.259242**). Su promedio aumenta de **1.0114** a **1.3448** y su mediana de **0.5833** a **0.8750**. El porcentaje de ceros cae de **21.01 %** a **1.36 %**. Estos resultados indican que, en las muestras comparadas, aumentó el nivel de actividad reciente y se redujo la proporción de periodos sin actividad según esta característica.

**Promedio móvil de salidas de las últimas 168 horas.** Registra una puntuación de **0.251141**. Su media pasa de **0.9997** a **1.3039** y la mediana de **0.6131** a **0.8690**. La reducción de ceros de **20.55 %** a **0.87 %** sugiere que el cambio también se refleja en una ventana semanal, y no exclusivamente en observaciones aisladas.

**Llegadas de bicicletas durante la hora observada.** La puntuación es **0.151835**. El promedio aumenta de **0.9835** a **1.3077**, mientras que los registros con cero llegadas disminuyen de **64.57 %** a **55.85 %**. La distribución posterior contiene más observaciones con actividad de llegada.

**Salidas de bicicletas durante la hora observada.** La puntuación es **0.145632**. El promedio se incrementa de **0.9962** a **1.3053**, y los registros sin salidas pasan de **64.24 %** a **55.55 %**. Esto es consistente con un mayor nivel de utilización de las bicicletas en la muestra del periodo posterior.

Estas diferencias pueden estar asociadas con estacionalidad, cambios de demanda, composición de los viajes o condiciones del sistema; **los resultados disponibles no permiten determinar la causa exacta**. Tampoco puede concluirse que se hayan inaugurado estaciones o construido nuevas ciclovías.

### 3.2. Variables sin drift detectado

Las variables de calendario tuvieron puntuaciones inferiores a **0.10**: `target_dayofweek` (**0.032862**), `target_hour` (**0.030943**) y `target_is_weekend` (**0.006712**). Por lo tanto, el reporte no identifica cambios relevantes en su distribución bajo las reglas elegidas. Por ejemplo, la proporción de fines de semana se mantiene cercana entre los periodos: **29.27 %** frente a **28.41 %**.

### 3.3. Calidad de datos, limitaciones y repercusiones sobre el modelo

No se encontraron valores nulos en las siete variables de ninguna de las dos muestras. Aun así, el análisis tiene limitaciones: compara periodos de distintas longitudes y estaciones del año; utiliza 10,000 registros por conjunto; y observa filas estación-hora que pueden compartir dependencia temporal o por estación. Además, `station_id` quedó fuera del análisis estadístico de drift, por lo que el porcentaje **57.14 % se refiere estrictamente a siete variables**.

La presencia de *data drift* **no demuestra que haya empeorado el desempeño predictivo** ni acredita *concept drift*. Para ello es necesario comparar predicciones con los valores reales y revisar métricas como MAE y RMSE. Asimismo, antes de afirmar que estas variables modifican las predicciones del modelo P1, debe verificarse la versión exacta del estimador que utiliza el equipo: si se emplea `DummyRegressor`, dicho modelo no utiliza estas variables como predictores, aunque sigan describiendo cambios relevantes en los datos del proyecto.

**Conclusión.** El monitoreo retrospectivo revela un desplazamiento consistente en las distribuciones de actividad de MiBici. Por su magnitud y recurrencia en diferentes ventanas temporales, conviene continuar el seguimiento, revisar representatividad por estación y temporada y contrastar estas señales con el error predictivo de los modelos durante la actividad de comparación. **No se recomienda activar un reentrenamiento automático únicamente por este reporte**; primero debe comprobarse si la versión candidata mejora efectivamente al modelo vigente.

## Evidencia y archivos de respaldo

- `reports/semana9_evidently/periodos_dataset.json`: fechas, cantidades de registros y versión de datos.
- `reports/semana9_evidently/evidently_data_drift.html`: resumen global, métricas y gráficas.
- `reports/semana9_evidently/evidently_calidad_datos.html`: exploración de calidad.
- `reports/semana9_evidently/drift_por_variable.csv`: resultados estadísticos por columna.
- `reports/semana9_evidently/calidad_variables.csv`: estadísticas descriptivas.
- `reports/semana9_evidently/evidently_resultado_drift.json`: salida estructurada.
- Capturas de ejecución y del reporte real: evidencia individual en GitHub.

## Referencias

Evidently AI. (s. f.). *Data and ML checks*. https://docs.evidentlyai.com/quickstart_ml

Huyen, C. (2022). *Designing machine learning systems: An iterative process for production-ready applications*. O’Reilly Media. Capítulo 8.

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps*. O’Reilly Media. Capítulo 7.

Equipo 46. (2026). *TC5061-Proyecto-Equipo46* [Repositorio de código]. GitHub. https://github.com/iplatamx/TC5061-Proyecto-Equipo46
