# Semana 9: reentrenamiento y A/B testing de P1

## Diseño de la comparación

Se comparó la versión 1 (`champion`) del modelo `mibici-equipo46-p1-demanda` contra una versión challenger reentrenada, en un holdout cronológico común. El conjunto de referencia para el monitoreo de drift y el conjunto actual de octubre están descritos en [semana9_preparacion_datos.md](semana9_preparacion_datos.md). El reporte de drift usa todo octubre; para evitar evaluar el challenger con las mismas etiquetas con las que aprendió, se hizo una separación temporal dentro de ese mes:

| Uso | Ventana de `target_time` | Filas |
|---|---|---:|
| Entrenamiento del challenger | 2026-01-01 01:00 a 2026-10-15 23:00, ambos inclusive | 3,268,903 |
| Evaluación A/B | 2026-10-16 00:00 a antes de 2026-11-01 00:00 | 181,632 |

El entrenamiento combina el historial usado para el modelo vigente con la primera mitad de octubre. La segunda mitad de octubre se mantuvo fuera del entrenamiento y se entregó exactamente en las mismas filas a ambas versiones. El modelo base v1 no se reentrenó: funciona como referencia vigente para medir si el challenger mejora sobre datos recientes.

## Registro de la versión alternativa

El challenger reutiliza la configuración de P1-HGB-02 (`max_iter=100`, `learning_rate=0.08`, `max_leaf_nodes=31`, `random_state=42`, `early_stopping=false`) para que el cambio principal sea incorporar datos recientes al entrenamiento. Se registró en MLflow bajo el mismo nombre de modelo:

- Modelo: `mibici-equipo46-p1-demanda`.
- Versión base y alias `champion`: v1.
- Challenger registrado: v2, estado `READY`.
- Run de entrenamiento: `6792e50ce3f447e0910c3a0ac5119186`.
- Versión de los datos de entrenamiento: `11d629c108eb`.

## Resultado del A/B testing

Las métricas se calcularon sobre las 181,632 filas del holdout de octubre. MAE es la métrica primaria porque fue el criterio de selección del champion original; RMSE y R² sirven como referencias complementarias.

| Métrica | Champion v1 | Challenger v2 | Cambio v2 − v1 |
|---|---:|---:|---:|
| MAE (menor es mejor) | 1.0257 | **1.0179** | **−0.0078 (−0.76%)** |
| RMSE (menor es mejor) | 1.7848 | **1.7527** | **−0.0321 (−1.80%)** |
| R² (mayor es mejor) | 0.7071 | **0.7175** | **+0.0105** |
| Sesgo medio (predicción − real) | −0.0477 | **−0.0322** | +0.0155; menos subestimación media |
| Tasa de subestimación | 32.26% | **32.14%** | −0.12 puntos porcentuales |

La comparación quedó registrada como run de MLflow `8962cfb741df487380938339fe2deec0`.

## Recomendación

Se recomienda **v2 como candidata a versión vigente**: obtuvo un MAE menor, que es el criterio primario, y también mejoró RMSE y R² en el mismo holdout. La mejora es pequeña (0.76% en MAE), así que el resultado apoya la actualización, pero no demuestra por sí solo que la mejora sea estable en todos los meses o estaciones. Conviene revisar errores por estación y repetir la evaluación con otro periodo posterior antes de una promoción definitiva.

El alias `champion` **sigue apuntando a v1**. El ejercicio deja registrada la recomendación y no cambia automáticamente la versión que consume el servicio. Si el equipo aprueba la promoción tras revisar el reporte, se puede mover el alias en un paso separado.

## Recomendación de Continuous Training

La decisión de reentrenar debe considerar únicamente los contextos señalados para el análisis: vacaciones, días festivos, contingencias del transporte público, cambios de tarifa, verano y fin de año. Estos contextos no forman parte de las ocho entradas actuales de P1; por eso, los resultados de drift de octubre y el A/B testing no permiten atribuirles directamente los cambios observados.

Se recomienda mantener una estrategia de reentrenamiento **supervisada y condicionada a evidencia**. Para cada contexto, primero se deben identificar las fechas con una fuente verificable y comparar el error del modelo en esas observaciones con el de periodos comparables. Un contexto por sí solo no debe activar el reentrenamiento. Si se observa una degradación consistente y hay suficientes datos etiquetados, se entrena un challenger con datos recientes y se evalúa en un periodo posterior independiente antes de recomendar su promoción. Hasta contar con esas etiquetas de contexto, la recomendación es continuar el monitoreo y considerar v2 como candidata según el A/B test, sin mover automáticamente el alias `champion`.

## Archivos y reproducción

- Configuración de fechas y modelo: `configs/week9_abtest.json`.
- Registro de la versión challenger: `reports/semana9/challenger_registration.json`.
- Resultados legibles: `reports/semana9/ab_test_october_second_half.md`.
- Resultados completos: `reports/semana9/ab_test_october_second_half.json`.
- Scripts: `scripts/train_week9_challenger.py` y `scripts/evaluate_week9_ab.py`.

Con MLflow disponible en `http://localhost:5000`, los datos históricos y el CSV de octubre en sus rutas configuradas, y el entorno del proyecto activo, se reproduce así desde la raíz del repositorio:

```powershell
python scripts/prepare_week9_datasets.py
python scripts/train_week9_challenger.py
python scripts/evaluate_week9_ab.py
```

El entrenamiento detecta si ya existe una versión registrada con la misma huella de datos para evitar crear duplicados. La evaluación verifica que el alias `champion` todavía apunte a v1 antes de comparar los modelos.
