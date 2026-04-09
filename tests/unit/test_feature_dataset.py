from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ivsurf.config import FeatureConfig, MarketCalendarConfig, SurfaceGridConfig
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
from ivsurf.surfaces.grid import SurfaceGrid


def _surface_frame(quote_dates: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, quote_date in enumerate(quote_dates, start=1):
        rows.append(
            {
                "quote_date": quote_date,
                "maturity_index": 0,
                "maturity_days": 30,
                "moneyness_index": 0,
                "moneyness_point": 0.0,
                "observed_total_variance": 0.01 * offset,
                "completed_total_variance": 0.01 * offset,
                "observed_mask": True,
                "vega_sum": 1.0,
                "observation_count": 1,
                "weighted_spread_1545": 0.01,
            }
        )
    return pl.DataFrame(rows)


def test_build_daily_feature_dataset_requires_minimum_history() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    surface_frame = _surface_frame([date(2021, 1, 4)])

    with pytest.raises(ValueError, match="Not enough daily surfaces"):
        build_daily_feature_dataset(
            surface_frame=surface_frame,
            grid=grid,
            feature_config=FeatureConfig(lag_windows=(1,)),
            calendar_config=MarketCalendarConfig(),
        )


def test_build_daily_feature_dataset_uses_next_observed_date_and_records_gap_sessions() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    feature_frame = build_daily_feature_dataset(
        surface_frame=_surface_frame(
            [
                date(2021, 1, 4),
                date(2021, 1, 6),
                date(2021, 1, 7),
            ]
        ),
        grid=grid,
        feature_config=FeatureConfig(
            lag_windows=(1,),
            include_daily_change=False,
            include_liquidity=False,
        ),
        calendar_config=MarketCalendarConfig(),
    ).feature_frame

    assert feature_frame.select("quote_date", "target_date", "target_gap_sessions").to_dict(
        as_series=False
    ) == {
        "quote_date": [date(2021, 1, 4), date(2021, 1, 6)],
        "target_date": [date(2021, 1, 6), date(2021, 1, 7)],
        "target_gap_sessions": [1, 0],
    }


def test_build_daily_feature_dataset_records_two_consecutive_skipped_sessions() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    feature_frame = build_daily_feature_dataset(
        surface_frame=_surface_frame(
            [
                date(2021, 1, 4),
                date(2021, 1, 5),
                date(2021, 1, 8),
            ]
        ),
        grid=grid,
        feature_config=FeatureConfig(
            lag_windows=(1,),
            include_daily_change=False,
            include_liquidity=False,
        ),
        calendar_config=MarketCalendarConfig(),
    ).feature_frame

    assert feature_frame["target_gap_sessions"].to_list() == [0, 2]
    assert all(
        target_date > quote_date
        for quote_date, target_date in zip(
            feature_frame["quote_date"].to_list(),
            feature_frame["target_date"].to_list(),
            strict=True,
        )
    )


def test_build_daily_feature_dataset_does_not_wrap_daily_change_to_the_last_surface() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    feature_frame = build_daily_feature_dataset(
        surface_frame=_surface_frame(
            [
                date(2021, 1, 4),
                date(2021, 1, 5),
                date(2021, 1, 6),
            ]
        ),
        grid=grid,
        feature_config=FeatureConfig(lag_windows=(1,), include_liquidity=False),
        calendar_config=MarketCalendarConfig(),
    ).feature_frame

    assert feature_frame["quote_date"].to_list() == [date(2021, 1, 5)]
    assert feature_frame["target_date"].to_list() == [date(2021, 1, 6)]
