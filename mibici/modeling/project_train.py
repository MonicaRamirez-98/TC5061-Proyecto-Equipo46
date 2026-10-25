"""Punto de entrada de entrenamiento para MLflow Projects."""

import json
from typing import Any

import typer
from loguru import logger

from mibici.config import PROJECT_TRAIN_DEFAULTS

app = typer.Typer()

HYPERPARAMETERS_BY_QUESTION = {
    "p1": {"strategy", "quantile"},
    "p2": {"strategy", "risk_fraction"},
    "p3": {"n_clusters", "n_init", "max_iter"},
}


@app.command()
def main(
    question: str = typer.Option(..., help="Pregunta: p1, p2 o p3."),
    model: str = typer.Option("auto", help="Modelo o 'auto' para usar el predeterminado."),
    hyperparameters_json: str = typer.Option(
        "{}",
        help="Hiperparámetros en formato JSON.",
    ),
):
    question = question.lower()

    if question not in PROJECT_TRAIN_DEFAULTS:
        raise typer.BadParameter("La pregunta debe ser p1, p2 o p3.")

    defaults = PROJECT_TRAIN_DEFAULTS[question]
    selected_model = defaults["model"] if model == "auto" else model

    if selected_model != defaults["model"]:
        raise typer.BadParameter(
            f"Para {question}, el modelo disponible es '{defaults['model']}'."
        )

    try:
        supplied_hyperparameters: dict[str, Any] = json.loads(hyperparameters_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Los hiperparámetros no son JSON válido: {exc}"
        ) from exc

    if not isinstance(supplied_hyperparameters, dict):
        raise typer.BadParameter("Los hiperparámetros deben ser un objeto JSON.")

    allowed = HYPERPARAMETERS_BY_QUESTION[question]
    unknown = set(supplied_hyperparameters) - allowed
    if unknown:
        raise typer.BadParameter(
            f"Hiperparámetros no admitidos para {question}: {sorted(unknown)}"
        )

    hyperparameters = {
        **defaults["hyperparameters"],
        **supplied_hyperparameters,
    }

    logger.info(
        "Entrenamiento solicitado: pregunta={}, modelo={}, hiperparámetros={}",
        question,
        selected_model,
        hyperparameters,
    )

    if question == "p1":
        from mibici.modeling.train import main as train_main
    elif question == "p2":
        from mibici.modeling.train_p2 import main as train_main
    else:
        from mibici.modeling.train_p3 import main as train_main

    train_main(model_name=selected_model, hyperparameters=hyperparameters)


if __name__ == "__main__":
    app()