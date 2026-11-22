"""Controles estructurales de Entregable 7; no sustituyen prueba con Docker."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_compose_has_core_services_and_internal_network():
    spec = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = spec["services"]
    assert {"api", "mlflow", "model-init"}.issubset(services)
    assert services["api"]["depends_on"]["model-init"]["condition"] == "service_completed_successfully"
    assert services["model-init"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert "backend" in spec["networks"]
    assert "mlflow_data" in spec["volumes"]


def test_healthchecks_and_persistent_volume():
    spec = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "healthcheck" in spec["services"]["api"]
    assert "healthcheck" in spec["services"]["mlflow"]
    assert "mlflow_data:/mlflow" in spec["services"]["mlflow"]["volumes"]
    assert "./models/compose-champion:/model-export:ro" in spec["services"]["model-init"]["volumes"]


def test_pipeline_automates_pytest_build_and_compose_smoke():
    # BaseLoader evita que PyYAML 1.1 trate la clave 'on' como booleano True.
    workflow = yaml.load(
        (ROOT / ".github/workflows/semana10-ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert "push" in workflow["on"]
    assert "pull_request" in workflow["on"]
    assert "unit_tests" in workflow["jobs"]
    assert "build_and_smoke" in workflow["jobs"]
    steps = workflow["jobs"]["build_and_smoke"]["steps"]
    joined = "\n".join(str(step) for step in steps)
    assert "docker/build-push-action" in joined
    assert "docker compose" in joined
    assert "ops.smoke_test" in joined
