# A/B testing P1: octubre de 2026

Holdout: `2026-10-16T00:00:00` a antes de `2026-11-01T00:00:00` (181,632 filas estación-hora). Ambos modelos recibieron las mismas filas.

| Versión | Rol | MAE | RMSE | R² | Sesgo medio (predicción − real) | Subestimación |
|---|---|---:|---:|---:|---:|---:|
| v1 | champion | 1.0257 | 1.7848 | 0.7071 | -0.0477 | 32.26% |
| v2 | challenger | 1.0179 | 1.7527 | 0.7175 | -0.0322 | 32.14% |

**Recomendación por MAE:** version_2. El challenger obtuvo menor MAE en el holdout temporal común; MAE es la métrica primaria usada para seleccionar el champion original.

El alias `champion` permanece en la versión 1; este resultado no cambia el modelo servido.
