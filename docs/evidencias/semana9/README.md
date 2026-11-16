# Evidencias – TC5061 Semana 9, MiBici Equipo 46

Este paquete reúne **capturas de la ejecución aportadas durante la conversación** y cuatro gráficas complementarias construidas a partir de los CSV reales de la evaluación. No modifica el código del proyecto.

## Capturas y finalidad

| Archivo | Evidencia |
|---|---|
| Evidencia_01_Pytest.png | Ejecución Pytest: 6 pruebas aprobadas, 320 advertencias de deprecación. |
| Evidencia_02_Datos_MiBici.png | Nueve archivos Parquet de enero a septiembre de 2026. |
| Evidencia_03_Evidently_DEMO.png | Validación técnica preliminar con datos **sintéticos**; no representa resultados reales. |
| Evidencia_04_Evidently_Datos_Reales.png | Drift real: 4 de 7 variables (57.14 %). |
| Evidencia_05_Consulta_Reportes.png | Comandos para abrir reportes de resultados reales. |
| Evidencia_06_Drift_24h_Grafica.png | Evidently: pestaña Data Drift de `departures_mean_last_24h`. |
| Evidencia_07_Drift_168h_Grafica.png | Evidently: pestaña Data Drift de `departures_mean_last_168h`. |
| Evidencia_08_Distribucion_168h.png | Evidently: pestaña Data Distribution, variable de 168 horas. |
| Evidencia_09_Distribucion_24h.png | Evidently: pestaña Data Distribution, variable de 24 horas. |
| Evidencia_10_Cuatro_Graficas_Generadas.png | La terminal confirma la generación de cuatro PNG complementarios. |

## Métricas respaldadas por las capturas y salidas de la conversación

- Ventana de referencia: 2026-01-01 01:00 a 2026-07-11 02:00 (entrenamiento, 70 % inicial).
- Ventana actual: 2026-08-21 02:00 a 2026-10-01 00:00 (prueba, 15 % final).
- Muestreo reproducible: 10,000 registros de cada ventana; semilla 42; datos `d038fe16493d`.
- Drift: 4 de 7 variables estudiadas (57.14 %); criterios por variable con umbral 0.10. La regla de detección a nivel de conjunto de datos utiliza 0.50 como proporción mínima de columnas con drift.
- `departures_mean_last_24h`: Wasserstein normalizada 0.259242, drift detectado.
- `departures_mean_last_168h`: Wasserstein normalizada 0.251141, drift detectado.
- `arrivals_this_hour`: 0.151835, drift detectado.
- `departures_this_hour`: 0.145632, drift detectado.
- `target_dayofweek`, `target_hour`, `target_is_weekend`: sin drift detectado.
- `station_id` se excluyó deliberadamente del reporte estadístico, así como el objetivo futuro y las marcas de tiempo; por eso el reporte presenta siete variables.
- **No se infiere degradación del modelo directamente a partir de drift:** la calidad de las predicciones deberá evaluarse con las etiquetas observadas en el periodo de prueba.

## Inclusión en Markdown

Desde un archivo `docs/Semana9_Axel_Secciones_1_a_3_Resultado_Real.md`, para insertar una captura:

```markdown
![Distribución de salidas promedio de 24 horas](evidencias/semana9/Evidencia_09_Distribucion_24h.png)
```

Para insertar una gráfica complementaria generada a partir de los CSV:

```markdown
![Comparación de promedios de MiBici](../reports/semana9_evidently/figuras/01_promedios_referencia_vs_actual.png)
```

Abrir la vista previa en Visual Studio Code: `Ctrl + Shift + V`.

**Advertencia:** las capturas 06 y 07 muestran el gráfico de Data Drift, mientras que 08 y 09 muestran el histograma Data Distribution. Para respaldar diferencias de distribución en el PDF priorizar 08 y 09, conservando 06 y 07 como evidencia adicional.

### Nota de uso académico

La actividad pide un informe final de 3 a 5 cuartillas, que integra las contribuciones de los dos integrantes. Este paquete documenta únicamente los puntos 1 a 3 del trabajo aportado por el estudiante, sin afirmar que el A/B testing o la recomendación de Continuous Training ya se hayan realizado.
