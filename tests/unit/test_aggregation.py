from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.aggregation import aggregate_daily_surface
from ivsurf.surfaces.grid import SurfaceGrid


def test_vega_weighted_aggregation_is_correct() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "maturity_index": [0, 0],
            "moneyness_index": [0, 0],
            "total_variance": [0.04, 0.09],
            "implied_volatility_1545": [0.20, 0.30],
            "spread_1545": [0.10, 0.20],
            "vega_1545": [1.0, 3.0],
        }
    )
    grid = SurfaceGrid(maturity_days=(7,), moneyness_points=(0.0,))
    surface = aggregate_daily_surface(frame=frame, grid=grid, config=SurfaceGridConfig(
        moneyness_points=(0.0,),
        maturity_days=(7,),
    ))
    assert abs(surface["observed_total_variance"][0] - 0.0775) < 1.0e-12

