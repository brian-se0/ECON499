"""Lagged-surface feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import polars as pl

from ivsurf.surfaces.grid import SurfaceGrid


@dataclass(frozen=True, slots=True)
class DailySurfaceArrays:
    """Dense daily arrays keyed by quote date."""

    quote_dates: list[date]
    completed_surfaces: np.ndarray
    observed_masks: np.ndarray
    observed_surfaces: np.ndarray
    vega_weights: np.ndarray


def pivot_surface_arrays(surface_frame: pl.DataFrame, grid: SurfaceGrid) -> DailySurfaceArrays:
    """Convert long daily surface rows into dense arrays."""

    sorted_frame = surface_frame.sort(["quote_date", "maturity_index", "moneyness_index"])
    raw_dates = (
        sorted_frame.select(pl.col("quote_date").unique(maintain_order=True)).to_series().to_list()
    )
    if any(not isinstance(value, date) for value in raw_dates):
        message = "Daily surface arrays require quote_date values to be Polars Date values."
        raise TypeError(message)
    dates = [cast(date, value) for value in raw_dates]
    rows_per_day = grid.shape[0] * grid.shape[1]
    observed_values = (
        sorted_frame["observed_total_variance"]
        .fill_null(np.nan)
        .to_numpy()
        .reshape(len(dates), rows_per_day)
    )
    completed_values = (
        sorted_frame["completed_total_variance"].to_numpy().reshape(len(dates), rows_per_day)
    )
    observed_masks = (
        sorted_frame["observed_mask"]
        .to_numpy()
        .reshape(len(dates), rows_per_day)
        .astype(np.float64)
    )
    vega_weights = sorted_frame["vega_sum"].to_numpy().reshape(len(dates), rows_per_day)
    return DailySurfaceArrays(
        quote_dates=dates,
        completed_surfaces=completed_values,
        observed_masks=observed_masks,
        observed_surfaces=observed_values,
        vega_weights=vega_weights,
    )


def summarize_lag_window(array: np.ndarray, index: int, window: int) -> np.ndarray:
    """Compute an equal-weight mean over the previous `window` observations."""

    start_index = index - window + 1
    if start_index < 0:
        message = f"Window {window} is not available at position {index}."
        raise ValueError(message)
    return np.asarray(array[start_index : index + 1].mean(axis=0), dtype=np.float64)
