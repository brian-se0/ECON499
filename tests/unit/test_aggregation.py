from __future__ import annotations

from datetime import date

import polars as pl

from ivcast.config import SurfaceGridConfig
from ivcast.surfaces.aggregation import aggregate_daily_surface
from ivcast.surfaces.grid import SurfaceGrid


def test_vega_weighted_aggregation_is_correct() -> None:
    node_years = 7.0 / 365.0
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "maturity_index": [0, 0],
            "moneyness_index": [0, 0],
            "tau_years": [node_years, node_years],
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


def test_aggregation_expresses_total_variance_at_the_node_maturity() -> None:
    # Two contracts with the same 20% implied volatility at 22 and 44 days both bin to
    # the 30-day node; the node value is 20% vol at 30 days, not their raw-w average.
    sigma = 0.20
    taus = (22.0 / 365.0, 44.0 / 365.0)
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "maturity_index": [0, 0],
            "moneyness_index": [0, 0],
            "tau_years": list(taus),
            "total_variance": [sigma * sigma * tau for tau in taus],
            "implied_volatility_1545": [sigma, sigma],
            "spread_1545": [0.10, 0.10],
            "vega_1545": [1.0, 2.0],
        }
    )
    grid = SurfaceGrid(maturity_days=(30,), moneyness_points=(0.0,))
    surface = aggregate_daily_surface(frame=frame, grid=grid, config=SurfaceGridConfig(
        moneyness_points=(0.0,),
        maturity_days=(30,),
    ))
    assert abs(surface["observed_total_variance"][0] - sigma * sigma * 30.0 / 365.0) < 1e-15
    assert abs(surface["observed_iv"][0] - sigma) < 1e-15

