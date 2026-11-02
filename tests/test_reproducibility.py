"""Semana 7: entrenamiento determinista y ligero con la configuración real de P3.

Prueba de reproducibilidad aislada: no dispara el entrenamiento integral,
no escribe runs en MLflow y no exige descargar datos de producción.
"""
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import PROJECT_TRAIN_DEFAULTS, RANDOM_SEED


def _train_p3_synthetic(features):
    """Replica la configuración del pipeline KMeans usada en train_p3.py."""
    params = PROJECT_TRAIN_DEFAULTS["p3"]["hyperparameters"]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=int(params["n_clusters"]),
                    n_init=int(params["n_init"]),
                    max_iter=int(params["max_iter"]),
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    return pipeline.fit(features)


def test_p3_misma_semilla_mismas_predicciones_y_metrica():
    """Dos entrenamientos aislados P3 generan resultados consistentes."""
    rng = np.random.default_rng(2026)
    groups = [
        rng.normal(loc=center, scale=0.2, size=(12, 4))
        for center in (-4.0, 0.0, 4.0)
    ]
    x = np.vstack(groups).astype("float32")

    first = _train_p3_synthetic(x)
    second = _train_p3_synthetic(x)

    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert first.named_steps["kmeans"].inertia_ == pytest.approx(
        second.named_steps["kmeans"].inertia_, rel=1e-12
    )


def test_semilla_del_proyecto_es_entera():
    """El experimento referencia una semilla central definida en config.py."""
    assert isinstance(RANDOM_SEED, int)
