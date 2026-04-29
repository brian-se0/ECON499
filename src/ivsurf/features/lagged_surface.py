"""Lagged-surface feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import polars as pl

from ivsurf.cleaning.derived_fields import DECISION_TIMESTAMP_COLUMN
from ivsurf.surfaces.grid import (
    SurfaceGrid,
    require_complete_unique_surface_grid,
    require_surface_grid_metadata,
)


@dataclass(frozen=True, slots=True)
class DailySurfaceArrays:
    """Dense daily arrays keyed by quote date."""

    quote_dates: list[date]
    completed_surfaces: np.ndarray
    observed_masks: np.ndarray
    observed_surfaces: np.ndarray
    vega_weights: np.ndarray
    decision_timestamps: list[str]
    surface_config_hashes: list[str]


def _per_date_string_values(
    sorted_frame: pl.DataFrame,
    *,
    dates: list[date],
    column_name: str,
) -> list[str]:
    grouped = (
        sorted_frame.group_by("quote_date")
        .agg(
            pl.col(column_name).n_unique().alias("unique_count"),
            pl.col(column_name).first().alias(column_name),
        )
        .sort("quote_date")
    )
    if grouped["quote_date"].to_list() != dates:
        message = f"{column_name} date keys do not match sorted surface dates."
        raise ValueError(message)
    invalid = grouped.filter(
        (pl.col("unique_count") != 1) | pl.col(column_name).is_null() | (pl.col(column_name) == "")
    )
    if invalid.height > 0:
        message = f"Surface frame must contain exactly one non-empty {column_name} per quote_date."
        raise ValueError(message)
    return [str(value) for value in grouped[column_name].to_list()]


def pivot_surface_arrays(surface_frame: pl.DataFrame, grid: SurfaceGrid) -> DailySurfaceArrays:
    """Convert long daily surface rows into dense arrays."""

    require_surface_grid_metadata(surface_frame, grid, dataset_name="Daily surface frame")
    require_complete_unique_surface_grid(
        surface_frame,
        grid,
        dataset_name="Daily surface frame",
        group_columns=("quote_date",),
    )
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
    decision_timestamps = _per_date_string_values(
        sorted_frame,
        dates=dates,
        column_name=DECISION_TIMESTAMP_COLUMN,
    )
    surface_config_hashes = _per_date_string_values(
        sorted_frame,
        dates=dates,
        column_name="surface_config_hash",
    )
    if len(set(surface_config_hashes)) != 1:
        message = "Daily surface frame contains multiple surface_config_hash values."
        raise ValueError(message)
    return DailySurfaceArrays(
        quote_dates=dates,
        completed_surfaces=completed_values,
        observed_masks=observed_masks,
        observed_surfaces=observed_values,
        vega_weights=vega_weights,
        decision_timestamps=decision_timestamps,
        surface_config_hashes=surface_config_hashes,
    )


def summarize_lag_window(array: np.ndarray, index: int, window: int) -> np.ndarray:
    """Compute an equal-weight mean over the previous `window` observations."""

    start_index = index - window + 1
    if start_index < 0:
        message = f"Window {window} is not available at position {index}."
        raise ValueError(message)
    return np.asarray(array[start_index : index + 1].mean(axis=0), dtype=np.float64)
