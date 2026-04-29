"""Persist model forecasts to parquet."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ivsurf.cleaning.derived_fields import DECISION_TIMESTAMP_COLUMN
from ivsurf.evaluation.metrics import validate_total_variance_array
from ivsurf.features.availability import TARGET_DECISION_TIMESTAMP_COLUMN
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
)
from ivsurf.surfaces.interpolation import COMPLETED_SURFACE_SCHEMA_VERSION


def _normalize_python_date(value: object) -> date:
    normalized = value.item() if isinstance(value, np.generic) else value
    if isinstance(normalized, datetime):
        return normalized.date()
    if isinstance(normalized, date):
        return normalized
    message = f"Expected forecast date value to be date-like, found {type(value).__name__}."
    raise TypeError(message)


def _normalize_python_dates(values: np.ndarray) -> list[date]:
    return [_normalize_python_date(value) for value in values]


def _normalize_string_values(
    values: np.ndarray,
    *,
    expected_rows: int,
    column_name: str,
) -> list[str]:
    if values.shape[0] != expected_rows:
        message = (
            f"{column_name} must contain one value per forecast row: "
            f"expected {expected_rows}, found {values.shape[0]}."
        )
        raise ValueError(message)
    normalized_values: list[str] = []
    for value in values:
        normalized = value.item() if isinstance(value, np.generic) else value
        if not isinstance(normalized, str) or not normalized:
            message = f"{column_name} values must be non-empty strings."
            raise TypeError(message)
        normalized_values.append(normalized)
    return normalized_values


def write_forecasts(
    output_path: Path,
    model_name: str,
    quote_dates: np.ndarray,
    target_dates: np.ndarray,
    split_ids: np.ndarray,
    decision_timestamps: np.ndarray,
    target_decision_timestamps: np.ndarray,
    predictions: np.ndarray,
    grid: SurfaceGrid,
    surface_config_hash: str,
    model_config_hash: str,
    training_run_id: str,
) -> None:
    """Write long-form forecast artifacts."""

    surface_cell_count = len(grid.maturity_days) * len(grid.moneyness_points)
    if quote_dates.shape[0] != target_dates.shape[0]:
        message = "quote_dates and target_dates must contain the same number of forecast rows."
        raise ValueError(message)
    if predictions.ndim != 2:
        message = f"predictions must be rank-2, found ndim={predictions.ndim}."
        raise ValueError(message)
    expected_shape = (quote_dates.shape[0], surface_cell_count)
    if predictions.shape != expected_shape:
        message = (
            "predictions shape does not match the forecast-date count and grid size: "
            f"expected {expected_shape}, found {predictions.shape}."
        )
        raise ValueError(message)
    for column_name, value in (
        ("surface_config_hash", surface_config_hash),
        ("model_config_hash", model_config_hash),
        ("training_run_id", training_run_id),
    ):
        if not isinstance(value, str) or not value:
            message = f"{column_name} must be a non-empty string."
            raise TypeError(message)

    flat_predictions = validate_total_variance_array(
        predictions.reshape(-1),
        context=f"Forecast artifact for model {model_name}",
        allow_zero=True,
    )
    normalized_quote_dates = _normalize_python_dates(quote_dates)
    normalized_target_dates = _normalize_python_dates(target_dates)
    normalized_split_ids = _normalize_string_values(
        split_ids,
        expected_rows=quote_dates.shape[0],
        column_name="split_id",
    )
    normalized_decision_timestamps = _normalize_string_values(
        decision_timestamps,
        expected_rows=quote_dates.shape[0],
        column_name=DECISION_TIMESTAMP_COLUMN,
    )
    normalized_target_decision_timestamps = _normalize_string_values(
        target_decision_timestamps,
        expected_rows=quote_dates.shape[0],
        column_name=TARGET_DECISION_TIMESTAMP_COLUMN,
    )

    per_surface_maturity_index = np.repeat(
        np.arange(len(grid.maturity_days), dtype=np.int64),
        len(grid.moneyness_points),
    )
    per_surface_maturity_days = np.repeat(
        np.asarray(grid.maturity_days, dtype=np.int64),
        len(grid.moneyness_points),
    )
    per_surface_moneyness_index = np.tile(
        np.arange(len(grid.moneyness_points), dtype=np.int64),
        len(grid.maturity_days),
    )
    per_surface_moneyness_points = np.tile(
        np.asarray(grid.moneyness_points, dtype=np.float64),
        len(grid.maturity_days),
    )

    repeated_quote_dates = np.repeat(
        np.asarray(normalized_quote_dates, dtype=object),
        surface_cell_count,
    ).tolist()
    repeated_target_dates = np.repeat(
        np.asarray(normalized_target_dates, dtype=object),
        surface_cell_count,
    ).tolist()
    repeated_split_ids = np.repeat(
        np.asarray(normalized_split_ids, dtype=object),
        surface_cell_count,
    )
    repeated_decision_timestamps = np.repeat(
        np.asarray(normalized_decision_timestamps, dtype=object),
        surface_cell_count,
    )
    repeated_target_decision_timestamps = np.repeat(
        np.asarray(normalized_target_decision_timestamps, dtype=object),
        surface_cell_count,
    )
    repeated_model_names = np.full(
        flat_predictions.shape[0],
        model_name,
        dtype=object,
    )
    frame = pl.DataFrame(
        {
            "model_name": repeated_model_names,
            "quote_date": pl.Series("quote_date", repeated_quote_dates, dtype=pl.Date),
            "target_date": pl.Series("target_date", repeated_target_dates, dtype=pl.Date),
            "split_id": repeated_split_ids,
            DECISION_TIMESTAMP_COLUMN: repeated_decision_timestamps,
            TARGET_DECISION_TIMESTAMP_COLUMN: repeated_target_decision_timestamps,
            "maturity_index": np.tile(per_surface_maturity_index, quote_dates.shape[0]),
            "maturity_days": np.tile(per_surface_maturity_days, quote_dates.shape[0]),
            "moneyness_index": np.tile(per_surface_moneyness_index, quote_dates.shape[0]),
            "moneyness_point": np.tile(per_surface_moneyness_points, quote_dates.shape[0]),
            "surface_grid_schema_version": np.full(
                flat_predictions.shape[0],
                SURFACE_GRID_SCHEMA_VERSION,
                dtype=object,
            ),
            "surface_grid_hash": np.full(
                flat_predictions.shape[0],
                grid.grid_hash,
                dtype=object,
            ),
            "maturity_coordinate": np.full(
                flat_predictions.shape[0],
                MATURITY_COORDINATE,
                dtype=object,
            ),
            "moneyness_coordinate": np.full(
                flat_predictions.shape[0],
                MONEYNESS_COORDINATE,
                dtype=object,
            ),
            "target_surface_version": np.full(
                flat_predictions.shape[0],
                COMPLETED_SURFACE_SCHEMA_VERSION,
                dtype=object,
            ),
            "surface_config_hash": np.full(
                flat_predictions.shape[0],
                surface_config_hash,
                dtype=object,
            ),
            "model_config_hash": np.full(
                flat_predictions.shape[0],
                model_config_hash,
                dtype=object,
            ),
            "training_run_id": np.full(
                flat_predictions.shape[0],
                training_run_id,
                dtype=object,
            ),
            "predicted_total_variance": flat_predictions,
        },
    )
    frame = frame.cast(
        {
            "model_name": pl.String,
            "quote_date": pl.Date,
            "target_date": pl.Date,
            "split_id": pl.String,
            DECISION_TIMESTAMP_COLUMN: pl.String,
            TARGET_DECISION_TIMESTAMP_COLUMN: pl.String,
            "maturity_index": pl.Int64,
            "maturity_days": pl.Int64,
            "moneyness_index": pl.Int64,
            "moneyness_point": pl.Float64,
            "surface_grid_schema_version": pl.String,
            "surface_grid_hash": pl.String,
            "maturity_coordinate": pl.String,
            "moneyness_coordinate": pl.String,
            "target_surface_version": pl.String,
            "surface_config_hash": pl.String,
            "model_config_hash": pl.String,
            "training_run_id": pl.String,
            "predicted_total_variance": pl.Float64,
        }
    )
    write_parquet_frame(frame, output_path)
