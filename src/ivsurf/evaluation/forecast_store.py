"""Persist model forecasts to parquet."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import validate_total_variance_array
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.surfaces.grid import SurfaceGrid


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


def write_forecasts(
    output_path: Path,
    model_name: str,
    quote_dates: np.ndarray,
    target_dates: np.ndarray,
    predictions: np.ndarray,
    grid: SurfaceGrid,
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

    flat_predictions = validate_total_variance_array(
        predictions.reshape(-1),
        context=f"Forecast artifact for model {model_name}",
        allow_zero=True,
    )
    normalized_quote_dates = _normalize_python_dates(quote_dates)
    normalized_target_dates = _normalize_python_dates(target_dates)

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
            "maturity_index": np.tile(per_surface_maturity_index, quote_dates.shape[0]),
            "maturity_days": np.tile(per_surface_maturity_days, quote_dates.shape[0]),
            "moneyness_index": np.tile(per_surface_moneyness_index, quote_dates.shape[0]),
            "moneyness_point": np.tile(per_surface_moneyness_points, quote_dates.shape[0]),
            "predicted_total_variance": flat_predictions,
        },
    )
    frame = frame.cast(
        {
            "model_name": pl.String,
            "quote_date": pl.Date,
            "target_date": pl.Date,
            "maturity_index": pl.Int64,
            "maturity_days": pl.Int64,
            "moneyness_index": pl.Int64,
            "moneyness_point": pl.Float64,
            "predicted_total_variance": pl.Float64,
        }
    )
    write_parquet_frame(frame, output_path)
