"""Helpers for reshaping long-form surface artifacts into dense grids."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ivsurf.surfaces.grid import SurfaceGrid


def _ordered_surface_frame(frame: pl.DataFrame, grid: SurfaceGrid) -> pl.DataFrame:
    ordered = frame.sort(["maturity_index", "moneyness_index"])
    expected_rows = grid.shape[0] * grid.shape[1]
    if ordered.height != expected_rows:
        message = (
            f"Expected exactly {expected_rows} rows for one dense surface, "
            f"found {ordered.height}."
        )
        raise ValueError(message)
    return ordered


def reshape_surface_column(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    column_name: str,
    *,
    null_fill: float | None = None,
) -> np.ndarray:
    """Reshape one long-form numeric surface column into the fixed grid."""

    ordered = _ordered_surface_frame(frame, grid)
    series = ordered[column_name]
    if series.null_count() > 0:
        if null_fill is None:
            message = f"Column {column_name} contains nulls but no null_fill was provided."
            raise ValueError(message)
        values = series.fill_null(null_fill).to_numpy()
    else:
        values = series.to_numpy()
    return np.asarray(values, dtype=np.float64).reshape(grid.shape)


def reshape_mask_column(frame: pl.DataFrame, grid: SurfaceGrid, column_name: str) -> np.ndarray:
    """Reshape one long-form boolean surface column into the fixed grid."""

    ordered = _ordered_surface_frame(frame, grid)
    values = ordered[column_name].to_numpy()
    return np.asarray(values, dtype=bool).reshape(grid.shape)


def dense_surface_rows(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    *,
    column_name: str,
    null_fill: float | None = None,
) -> list[dict[str, Any]]:
    """Return a dense surface column as row dictionaries for debugging or reporting."""

    surface = reshape_surface_column(frame, grid, column_name, null_fill=null_fill)
    rows: list[dict[str, Any]] = []
    for maturity_index, maturity_days in enumerate(grid.maturity_days):
        for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
            rows.append(
                {
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    column_name: float(surface[maturity_index, moneyness_index]),
                }
            )
    return rows
