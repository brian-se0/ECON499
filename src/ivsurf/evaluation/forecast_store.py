"""Persist model forecasts to parquet."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

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


def write_forecasts(
    output_path: Path,
    model_name: str,
    quote_dates: np.ndarray,
    target_dates: np.ndarray,
    predictions: np.ndarray,
    grid: SurfaceGrid,
) -> None:
    """Write long-form forecast artifacts."""

    rows: list[dict[str, object]] = []
    pairs = zip(quote_dates, target_dates, strict=True)
    for row_index, (quote_date, target_date) in enumerate(pairs):
        normalized_quote_date = _normalize_python_date(quote_date)
        normalized_target_date = _normalize_python_date(target_date)
        surface = predictions[row_index].reshape(grid.shape)
        for maturity_index, maturity_day in enumerate(grid.maturity_days):
            for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": normalized_quote_date,
                        "target_date": normalized_target_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_day,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "predicted_total_variance": float(surface[maturity_index, moneyness_index]),
                    }
                )
    frame = pl.DataFrame(
        rows,
        schema={
            "model_name": pl.String,
            "quote_date": pl.Date,
            "target_date": pl.Date,
            "maturity_index": pl.Int64,
            "maturity_days": pl.Int64,
            "moneyness_index": pl.Int64,
            "moneyness_point": pl.Float64,
            "predicted_total_variance": pl.Float64,
        },
    )
    write_parquet_frame(frame, output_path)
