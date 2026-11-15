"""Gráficas complementarias para TC5061, Semana 9, MiBici Equipo 46.

Lee EXCLUSIVAMENTE los CSV que generó el análisis de Evidently con datos
reales. No modifica los datos ni vuelve a entrenar modelos.

Uso desde la RAÍZ del repositorio:
    python scripts/generar_graficas_semana9.py

Dependencia: matplotlib (python -m pip install matplotlib).

Referencias para comprender la implementación:
- Evidently: https://docs.evidentlyai.com/quickstart_ml
- Matplotlib: https://matplotlib.org/stable/users/index.html
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
INPUT = ROOT / "reports" / "semana9_evidently"
OUTPUT = INPUT / "figuras"
NUMERICAS = [
    "departures_mean_last_24h",
    "departures_mean_last_168h",
    "arrivals_this_hour",
    "departures_this_hour",
]
CATEGORICAS = ["target_dayofweek", "target_hour", "target_is_weekend"]
ALIAS = {
    "departures_mean_last_24h": "Promedio de salidas (24 h)",
    "departures_mean_last_168h": "Promedio de salidas (168 h)",
    "arrivals_this_hour": "Llegadas en la hora",
    "departures_this_hour": "Salidas en la hora",
    "target_dayofweek": "Día de la semana",
    "target_hour": "Hora del día",
    "target_is_weekend": "Fin de semana",
}


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró {ruta}. Ejecuta primero Evidently con los datos reales.")
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def numero(valor: str | None) -> float:
    return float(str(valor).strip())


def guardar(nombre: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destino = OUTPUT / nombre
    plt.savefig(destino, dpi=190, bbox_inches="tight")
    plt.close()
    print(f"[OK] {destino}")


def grafica_comparativa(
    valores: dict[tuple[str, str], dict[str, str]],
    campo: str,
    titulo: str,
    eje_x: str,
    nombre: str,
    variables: list[str],
) -> None:
    etiquetas = [ALIAS.get(col, col) for col in variables]
    referencia = [numero(valores[(col, "referencia")][campo]) for col in variables]
    actual = [numero(valores[(col, "actual")][campo]) for col in variables]
    pos = np.arange(len(variables))
    alto = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.barh(pos - alto / 2, referencia, alto, label="Referencia (entrenamiento)")
    ax.barh(pos + alto / 2, actual, alto, label="Periodo actual (prueba)")
    ax.set_yticks(pos, etiquetas)
    ax.invert_yaxis()
    ax.set_xlabel(eje_x)
    ax.set_title(titulo)
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.01, 0.01, "Fuente: calidad_variables.csv; resumen descriptivo de las muestras de 10 000 registros por periodo.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    guardar(nombre)


def grafica_scores(
    registros: list[dict[str, str]],
    variables: list[str],
    metodo: str,
    titulo: str,
    nombre: str,
) -> None:
    por_columna = {f["variable"]: f for f in registros}
    presentes = [col for col in variables if col in por_columna and por_columna[col].get("score")]
    if not presentes:
        print(f"[AVISO] No hay puntuaciones para {nombre}; revisa drift_por_variable.csv")
        return
    scores = [numero(por_columna[col]["score"]) for col in presentes]
    umbrales = [numero(por_columna[col]["umbral"]) for col in presentes]
    etiquetas = [ALIAS.get(col, col) for col in presentes]
    fig, ax = plt.subplots(figsize=(11, max(3.8, 1.0 * len(presentes) + 1.6)))
    pos = np.arange(len(presentes))
    ax.barh(pos, scores)
    ax.axvline(umbrales[0], linestyle="--", linewidth=1.5, label=f"Umbral = {umbrales[0]:.2f}")
    ax.set_yticks(pos, etiquetas)
    ax.invert_yaxis()
    ax.set_xlabel(f"Puntuación de drift ({metodo})")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    max_value = max(scores + umbrales) * 1.22
    ax.set_xlim(0, max_value)
    for p, s in zip(pos, scores):
        ax.text(s + max_value * 0.01, p, f"{s:.4f}", va="center", fontsize=9)
    fig.text(0.01, 0.01, "Fuente: drift_por_variable.csv; criterio de detección configurado en Evidently.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    guardar(nombre)


def main() -> None:
    calidad = leer_csv(INPUT / "calidad_variables.csv")
    drift = leer_csv(INPUT / "drift_por_variable.csv")
    valores = {(row["variable"], row["periodo"]): row for row in calidad}
    faltantes = [(col, periodo) for col in NUMERICAS for periodo in ("referencia", "actual") if (col, periodo) not in valores]
    if faltantes:
        raise ValueError(f"Faltan columnas/periodos en calidad_variables.csv: {faltantes}")
    grafica_comparativa(
        valores, "promedio", "MiBici P1: medias de variables con drift",
        "Promedio de la variable (bicicletas por estación-hora)",
        "01_promedios_referencia_vs_actual.png", NUMERICAS,
    )
    grafica_comparativa(
        valores, "porcentaje_ceros", "MiBici P1: proporción de valores cero",
        "Porcentaje de registros con valor cero (%)",
        "02_porcentaje_ceros.png", NUMERICAS,
    )
    grafica_scores(
        drift, NUMERICAS, "Wasserstein normalizada",
        "MiBici P1: drift en las variables numéricas", "03_scores_drift_numericas.png",
    )
    grafica_scores(
        drift, CATEGORICAS, "Jensen–Shannon",
        "MiBici P1: drift en las variables de calendario", "04_scores_drift_calendario.png",
    )
    print("\nEstas PNG son gráficas COMPARATIVAS construidas con los CSV de resultados.")
    print("Para ver las distribuciones originales de Evidently, abre evidently_data_drift.html.")


if __name__ == "__main__":
    main()
