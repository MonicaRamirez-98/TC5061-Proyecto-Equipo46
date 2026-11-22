"""Prueba de carga de la API MiBici P1 con Locust.

Simula usuarios que consultan la predicción de salidas de la hora siguiente.
Cada usuario envía peticiones individuales (/predict), por lote (/predict/batch)
y, con menor frecuencia, revisa el estado del servicio (/health).

Uso, desde la raíz del repositorio y con la API levantada:
    locust -f loadtest/locustfile.py --host http://localhost:8000

Ejecución sin interfaz, guardando resultados:
    locust -f loadtest/locustfile.py --host http://localhost:8000 --headless \
        -u 50 -r 5 -t 3m --csv reports/semana10_locust/u50 \
        --html reports/semana10_locust/u50.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

BATCH_SIZE = 10


def random_record() -> dict:
    """Genera una entrada válida para el contrato de la API (8 variables)."""
    dayofweek = random.randint(0, 6)
    mean_24h = round(random.uniform(0.0, 6.0), 2)
    return {
        "station_id": random.randint(1, 473),
        "departures_this_hour": random.randint(0, 12),
        "arrivals_this_hour": random.randint(0, 12),
        "departures_mean_last_24h": mean_24h,
        "departures_mean_last_168h": round(max(mean_24h + random.uniform(-1.0, 1.0), 0.0), 2),
        "target_hour": random.randint(0, 23),
        "target_dayofweek": dayofweek,
        "target_is_weekend": 1 if dayofweek >= 5 else 0,
    }


class MiBiciUser(HttpUser):
    # Pausa entre peticiones de un mismo usuario, como una aplicación que consulta periódicamente.
    wait_time = between(0.5, 2.0)

    @task(6)
    def predict_one(self):
        with self.client.post("/predict", json=random_record(), name="/predict",
                              catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif "predicted_departures_next_hour" not in response.json():
                response.failure("Respuesta sin predicción")

    @task(2)
    def predict_batch(self):
        records = [random_record() for _ in range(BATCH_SIZE)]
        with self.client.post("/predict/batch", json={"records": records},
                              name=f"/predict/batch ({BATCH_SIZE})", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif len(response.json().get("predictions", [])) != BATCH_SIZE:
                response.failure("Número de predicciones distinto al enviado")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
