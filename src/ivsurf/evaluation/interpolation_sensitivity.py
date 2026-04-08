"""Interpolation-order sensitivity checks using saved observed surfaces only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.surfaces.interpolation import complete_surface
from ivsurf.surfaces.masks import reshape_mask_column, reshape_surface_column


@dataclass(frozen=True, slots=True)
class InterpolationSensitivityRow:
    """Difference between canonical and alternate interpolation orders."""

    quote_date: date
    observed_cell_count: int
    total_cell_count: int
    mean_abs_diff: float
    rmse_diff: float
    max_abs_diff: float


def build_interpolation_sensitivity_frame(
    actual_surface_frame: pl.DataFrame,
    *,
    grid: SurfaceGrid,
    surface_config: SurfaceGridConfig,
    alternate_order: tuple[str, ...],
    interpolation_cycles: int | None = None,
) -> pl.DataFrame:
    """Compare stored completed surfaces against a fixed alternate interpolation order."""

    cycles = (
        surface_config.interpolation_cycles
        if interpolation_cycles is None
        else interpolation_cycles
    )
    rows: list[InterpolationSensitivityRow] = []
    for group in actual_surface_frame.partition_by("quote_date", maintain_order=True):
        observed_surface = reshape_surface_column(
            group,
            grid,
            "observed_total_variance",
            null_fill=float("nan"),
        )
        observed_mask = reshape_mask_column(group, grid, "observed_mask")
        reference_surface = reshape_surface_column(group, grid, "completed_total_variance")
        alternate_surface = complete_surface(
            observed_total_variance=observed_surface,
            maturity_coordinates=grid.maturity_years,
            moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
            interpolation_order=alternate_order,
            interpolation_cycles=cycles,
            total_variance_floor=surface_config.total_variance_floor,
        ).completed_total_variance
        diff = alternate_surface - reference_surface
        rows.append(
            InterpolationSensitivityRow(
                quote_date=group["quote_date"][0],
                observed_cell_count=int(observed_mask.sum()),
                total_cell_count=int(observed_mask.size),
                mean_abs_diff=float(np.mean(np.abs(diff))),
                rmse_diff=float(np.sqrt(np.mean(np.square(diff)))),
                max_abs_diff=float(np.max(np.abs(diff))),
            )
        )
    return pl.DataFrame(rows).sort("quote_date")


def summarize_interpolation_sensitivity(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate interpolation sensitivity to one-row summary statistics."""

    mean_mean_abs_diff = float(cast(float, frame["mean_abs_diff"].mean()))
    mean_rmse_diff = float(cast(float, frame["rmse_diff"].mean()))
    max_max_abs_diff = float(cast(float, frame["max_abs_diff"].max()))
    return pl.DataFrame(
        {
            "mean_mean_abs_diff": [mean_mean_abs_diff],
            "mean_rmse_diff": [mean_rmse_diff],
            "max_max_abs_diff": [max_max_abs_diff],
            "n_quote_dates": [frame.height],
        }
    )
