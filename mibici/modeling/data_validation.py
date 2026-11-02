from pathlib import Path

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation


_TARGET_COLUMNS = {
    "target_departures_next_hour",
    "target_arrivals_next_hour",
    "target_risk_next_hour",
}

_CATEGORICAL_COLUMNS = {
    "station_id",
    "target_hour",
    "target_dayofweek",
    "target_is_weekend",
}


def _to_deepchecks_dataset(
    frame: pd.DataFrame,
    label: str,
) -> Dataset:
    if frame.empty:
        raise ValueError("El conjunto que se va a validar está vacío.")

    required_columns = {"station_id", "target_time", label}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    data = frame.copy()
    data["target_time"] = pd.to_datetime(
        data["target_time"],
        errors="coerce",
    )

    key_columns = ["station_id", "target_time"]
    if data[key_columns].isna().any().any():
        raise ValueError("Hay valores faltantes en station_id o target_time.")

    if data.duplicated(key_columns).any():
        raise ValueError("Hay registros duplicados de estación y hora.")

    # La clave se mantiene estable entre train, validation y test.
    data["sample_id"] = (
        data["station_id"].astype(str)
        + "|"
        + data["target_time"].astype(str)
    )

    # Evita incluir como características otros objetivos futuros.
    columns_to_drop = ["timestamp"] + sorted(_TARGET_COLUMNS - {label})
    data = data.drop(columns=columns_to_drop, errors="ignore")

    categorical = [
        column
        for column in _CATEGORICAL_COLUMNS
        if column in data.columns
    ]

    return Dataset(
        data,
        label=label,
        index_name="sample_id",
        datetime_name="target_time",
        cat_features=categorical,
    )


def validate_panel_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    label: str,
    report_dir: Path,
    report_prefix: str,
) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train": _to_deepchecks_dataset(train, label),
        "validation": _to_deepchecks_dataset(validation, label),
        "test": _to_deepchecks_dataset(test, label),
    }

    # Revisa la integridad de cada partición por separado.
    for split_name, dataset in datasets.items():
        result = data_integrity(
            n_samples=100_000,
            random_state=42,
        ).run(dataset)

        result.save_as_html(
            str(report_dir / f"{report_prefix}_{split_name}_integrity.html"),
            as_widget=False,
        )

    # Compara train con validation y train con test.
    for split_name in ("validation", "test"):
        result = train_test_validation().run(
            datasets["train"],
            datasets[split_name],
        )

        result.save_as_html(
            str(report_dir / f"{report_prefix}_train_vs_{split_name}.html"),
            as_widget=False,
        )