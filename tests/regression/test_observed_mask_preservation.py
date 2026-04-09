from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.aggregation import aggregate_daily_surface
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.surfaces.interpolation import complete_surface


def test_thresholded_observed_mask_survives_surface_completion() -> None:
    surface_config = SurfaceGridConfig(
        moneyness_points=(0.0, 0.1),
        maturity_days=(30, 60),
        observed_cell_min_count=2,
    )
    grid = SurfaceGrid.from_config(surface_config)
    quote_date = date(2021, 1, 4)
    rows = [
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "moneyness_index": 0,
            "total_variance": 0.010,
            "implied_volatility_1545": 0.20,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "moneyness_index": 0,
            "total_variance": 0.012,
            "implied_volatility_1545": 0.21,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "moneyness_index": 1,
            "total_variance": 0.013,
            "implied_volatility_1545": 0.22,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 0,
            "total_variance": 0.014,
            "implied_volatility_1545": 0.23,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 0,
            "total_variance": 0.015,
            "implied_volatility_1545": 0.24,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 1,
            "total_variance": 0.016,
            "implied_volatility_1545": 0.25,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 1,
            "total_variance": 0.017,
            "implied_volatility_1545": 0.26,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
    ]
    observed = aggregate_daily_surface(pl.DataFrame(rows), grid, surface_config).sort(
        ["quote_date", "maturity_index", "moneyness_index"]
    )
    observed_matrix = (
        observed["observed_total_variance"]
        .fill_null(np.nan)
        .to_numpy()
        .reshape(grid.shape)
    )
    observed_mask = observed["observed_mask"].to_numpy().reshape(grid.shape)

    completed = complete_surface(
        observed_total_variance=observed_matrix,
        observed_mask=observed_mask,
        maturity_coordinates=grid.maturity_years,
        moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
        interpolation_order=surface_config.interpolation_order,
        interpolation_cycles=surface_config.interpolation_cycles,
        total_variance_floor=surface_config.total_variance_floor,
    )

    assert np.isfinite(completed.completed_total_variance[0, 1])
    assert bool(completed.observed_mask[0, 1]) is False
