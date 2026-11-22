"""Comprueba API + carga del modelo + contrato P1 (sin dependencias externas)."""
from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SAMPLE = {
    "station_id": 1,
    "departures_this_hour": 5,
    "arrivals_this_hour": 4,
    "departures_mean_last_24h": 3.5,
    "departures_mean_last_168h": 4.2,
    "target_hour": 9,
    "target_dayofweek": 1,
    "target_is_weekend": 0,
}


def request_json(base: str, route: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(
        base.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urlopen(req, timeout=15) as resp:
        return json.load(resp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    try:
        health = request_json(args.url, "/health")
        assert health["status"] == "ok" and health["model_loaded"] is True, health
        info = request_json(args.url, "/model-info")
        assert info["model_name"] == "mibici-equipo46-p1-demanda", info
        assert info["model_version"], info
        prediction = request_json(args.url, "/predict", SAMPLE)
        assert prediction["station_id"] == SAMPLE["station_id"], prediction
        assert prediction["predicted_departures_next_hour"] >= 0, prediction
        assert prediction["model_version"] == info["model_version"], prediction
    except (AssertionError, HTTPError, URLError, ValueError) as exc:
        raise SystemExit(f"PRUEBA SMOKE FALLÓ: {exc}") from exc
    print("OK /health:", health)
    print("OK /model-info:", info)
    print("OK /predict:", prediction)
    print("SMOKE TEST APROBADO: API, Registry y predicción enlazados.")


if __name__ == "__main__":
    main()
