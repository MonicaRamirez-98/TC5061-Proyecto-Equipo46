"""Resume las pruebas de carga de Locust y el consumo de recursos (Semana 10).

Lee, para cada escenario (10, 50 y 100 usuarios), los archivos que generan
locust --csv y loadtest/monitor_docker_stats.ps1 en reports/semana10_locust/,
y produce:
  - resumen_escenarios.csv: latencia, throughput, inferencias/s y CPU/memoria de la API
  - finops_costo_inferencia.csv: costo por inferencia con precios de referencia
  - figuras/*.png: gráficas de desempeño y recursos

Uso, desde la raíz del repositorio:
    python loadtest/analyze_results.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

USERS = (10, 50, 100)
BATCH_SIZE = 10  # registros por petición en /predict/batch (loadtest/locustfile.py)

# Precio de referencia: AWS EC2 c7i.xlarge (4 vCPU, 8 GiB), Linux, us-east-1, USD por hora.
# Fuente: instances.vantage.sh/aws/ec2/c7i.xlarge (consultado el 2026-11-22).
INSTANCE = "AWS c7i.xlarge (4 vCPU, 8 GiB)"
PRICES_USD_HOUR = {"On-Demand": 0.179, "Reservada 1 año": 0.118, "Spot": 0.064}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/semana10_locust")
    return parser.parse_args()


def load_scenario(folder: Path, users: int) -> dict:
    stats = pd.read_csv(folder / f"u{users}_stats.csv")
    agg = stats[stats["Name"] == "Aggregated"].iloc[0]
    rps = stats.set_index("Name")["Requests/s"]
    batch_name = next(n for n in rps.index if str(n).startswith("/predict/batch"))
    inferences_s = rps["/predict"] + BATCH_SIZE * rps[batch_name]

    # Ventana de la prueba según el historial de Locust (solo con usuarios activos).
    history = pd.read_csv(folder / f"u{users}_stats_history.csv")
    history = history[(history["Name"] == "Aggregated") & (history["User Count"] > 0)]
    start = datetime.fromtimestamp(history["Timestamp"].min())
    end = datetime.fromtimestamp(history["Timestamp"].max())

    docker = pd.read_csv(folder / f"docker_stats_u{users}.csv", encoding="utf-8-sig")
    docker["timestamp"] = pd.to_datetime(docker["timestamp"])
    docker = docker[(docker["timestamp"] >= start) & (docker["timestamp"] <= end)]
    docker["cpu"] = docker["cpu_percent"].str.rstrip("%").astype(float)
    docker["mem_mib"] = docker["mem_usage"].str.split(" / ").str[0].map(to_mib)
    api = docker[docker["container"].str.contains("api")]
    mlflow = docker[docker["container"].str.contains("mlflow")]

    return {
        "usuarios": users,
        "peticiones": int(agg["Request Count"]),
        "fallas": int(agg["Failure Count"]),
        "tasa_error_pct": round(100 * agg["Failure Count"] / agg["Request Count"], 2),
        "peticiones_s": round(agg["Requests/s"], 2),
        "inferencias_s": round(inferences_s, 2),
        "latencia_p50_ms": agg["50%"],
        "latencia_p95_ms": agg["95%"],
        "latencia_p99_ms": agg["99%"],
        "latencia_max_ms": round(agg["Max Response Time"]),
        "api_cpu_prom_pct": round(api["cpu"].mean(), 1),
        "api_cpu_max_pct": round(api["cpu"].max(), 1),
        "api_mem_max_mib": round(api["mem_mib"].max(), 1),
        "mlflow_cpu_prom_pct": round(mlflow["cpu"].mean(), 2),
        "lecturas_docker": int(len(api)),
        "_history": history,
        "_api": api,
        "_start": start,
    }


def to_mib(text: str) -> float:
    units = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1024, "MB": 1, "GB": 1024}
    for unit, factor in units.items():
        if text.endswith(unit):
            return float(text[: -len(unit)]) * factor
    return float(text)


def main():
    args = parse_args()
    folder = Path(args.input)
    figures = folder / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    scenarios = [load_scenario(folder, u) for u in USERS]
    summary = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in scenarios])
    summary.to_csv(folder / "resumen_escenarios.csv", index=False)

    # FinOps: costo por inferencia a la capacidad observada de cada escenario.
    rows = []
    for s in scenarios:
        per_hour = s["inferencias_s"] * 3600
        for plan, price in PRICES_USD_HOUR.items():
            rows.append({
                "usuarios": s["usuarios"],
                "instancia": INSTANCE,
                "plan": plan,
                "precio_usd_hora": price,
                "inferencias_hora": round(per_hour),
                "usd_por_millon_inferencias": round(price / per_hour * 1_000_000, 4),
            })
    finops = pd.DataFrame(rows)
    finops.to_csv(folder / "finops_costo_inferencia.csv", index=False)

    x = [str(u) for u in USERS]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(x, summary["latencia_p50_ms"], marker="o", label="p50")
    ax.plot(x, summary["latencia_p95_ms"], marker="o", label="p95")
    for i, row in summary.iterrows():
        ax.annotate(f"{row['latencia_p95_ms']:.0f}", (x[i], row["latencia_p95_ms"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.annotate(f"{row['latencia_p50_ms']:.0f}", (x[i], row["latencia_p50_ms"]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set(xlabel="Usuarios concurrentes", ylabel="Latencia (ms)", title="Latencia por escenario")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures / "01_latencia_por_escenario.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar([i - 0.2 for i in range(3)], summary["peticiones_s"], width=0.4, label="Peticiones/s")
    ax.bar([i + 0.2 for i in range(3)], summary["inferencias_s"], width=0.4, label="Inferencias/s")
    for i, row in summary.iterrows():
        ax.annotate(f"{row['peticiones_s']:.1f}", (i - 0.2, row["peticiones_s"]), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{row['inferencias_s']:.1f}", (i + 0.2, row["inferencias_s"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3), x)
    ax.set(xlabel="Usuarios concurrentes", ylabel="Por segundo", title="Throughput por escenario (tasa de error 0 %)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures / "02_throughput_por_escenario.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for s in scenarios:
        api = s["_api"]
        seconds = (api["timestamp"] - api["timestamp"].min()).dt.total_seconds()
        ax.plot(seconds, api["cpu"], label=f"{s['usuarios']} usuarios")
    ax.set(xlabel="Segundos de prueba", ylabel="CPU del contenedor api (%)", title="Uso de CPU de la API (100 % = 1 núcleo)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures / "03_cpu_api.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for s in scenarios:
        h = s["_history"].copy()
        h = h[pd.to_numeric(h["95%"], errors="coerce").notna()]
        seconds = h["Timestamp"] - h["Timestamp"].min()
        ax.plot(seconds, pd.to_numeric(h["95%"]), label=f"{s['usuarios']} usuarios")
    ax.set(xlabel="Segundos de prueba", ylabel="Latencia p95 (ms)", title="Latencia p95 durante la prueba")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures / "04_latencia_p95_tiempo.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print()
    print(finops.to_string(index=False))
    print(f"\nResultados guardados en {folder}")


if __name__ == "__main__":
    main()
