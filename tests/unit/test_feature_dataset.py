from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ivsurf.config import FeatureConfig, MarketCalendarConfig, SurfaceGridConfig
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
from ivsurf.surfaces.grid import SurfaceGrid


def test_build_daily_feature_dataset_requires_next_day_target() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    surface_frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4)],
            "maturity_index": [0],
            "moneyness_index": [0],
            "observed_total_variance": [0.04],
            "completed_total_variance": [0.04],
            "observed_mask": [True],
            "vega_sum": [1.0],
            "observation_count": [1],
            "weighted_spread_1545": [0.01],
        }
    )

    with pytest.raises(ValueError, match="Not enough daily surfaces"):
        build_daily_feature_dataset(
            surface_frame=surface_frame,
            grid=grid,
            feature_config=FeatureConfig(lag_windows=(1,)),
            calendar_config=MarketCalendarConfig(),
        )
