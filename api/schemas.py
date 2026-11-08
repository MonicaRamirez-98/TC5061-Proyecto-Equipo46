"""Esquemas Pydantic de entrada y salida de la API P1.

Replican las reglas del contrato de Axel (mibici/serving/contract.py)
para que FastAPI rechace con HTTP 422 cualquier petición inválida
antes de llegar al modelo.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mibici.serving.contract import SAMPLE_P1


class PredictionInput(BaseModel):
    """Estado de una estación MiBici en la hora observada."""

    # extra="forbid": rechaza campos que el modelo no conoce.
    # strict=True: no convierte "5" (texto) ni true (booleano) en números.
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [SAMPLE_P1]},
    )

    station_id: int = Field(..., gt=0, description="Identificador de la estación.")
    departures_this_hour: int = Field(
        ..., ge=0, description="Viajes que salieron en la hora observada."
    )
    arrivals_this_hour: int = Field(
        ..., ge=0, description="Viajes que llegaron en la hora observada."
    )
    departures_mean_last_24h: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="Media de salidas por hora en las últimas 24 h.",
    )
    departures_mean_last_168h: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="Media de salidas por hora en las últimas 168 h.",
    )
    target_hour: int = Field(..., ge=0, le=23, description="Hora objetivo (0-23).")
    target_dayofweek: int = Field(
        ..., ge=0, le=6, description="Día de la semana objetivo (lunes=0, domingo=6)."
    )
    target_is_weekend: int = Field(
        ..., ge=0, le=1, description="1 si el día objetivo es sábado o domingo."
    )

    @model_validator(mode="after")
    def check_weekend(self) -> "PredictionInput":
        expected = int(self.target_dayofweek >= 5)
        if self.target_is_weekend != expected:
            raise ValueError(
                f"target_is_weekend debe ser {expected} cuando "
                f"target_dayofweek es {self.target_dayofweek}."
            )
        return self


class PredictionOutput(BaseModel):
    station_id: int
    predicted_departures_next_hour: float = Field(
        ..., ge=0, description="Viajes estimados que iniciarán en la hora siguiente."
    )
    model_name: str
    model_version: str


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[PredictionInput] = Field(..., min_length=1, max_length=1000)


class BatchOutput(BaseModel):
    predictions: list[PredictionOutput]


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoOutput(BaseModel):
    model_name: str
    model_version: str
    model_uri: str
    feature_columns: list[str]
