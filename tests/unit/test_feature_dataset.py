from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ivsurf.config import FeatureConfig, MarketCalendarConfig, SurfaceGridConfig
from ivsurf.features.availability import build_feature_availability_manifest
from ivsurf.features.tabular_dataset import (
    build_daily_feature_dataset,
    build_target_training_weights,
)
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
)


def _surface_frame(quote_dates: list[date]) -> pl.DataFrame:
    grid = SurfaceGrid(maturity_days=(30,), moneyness_points=(0.0,))
    rows: list[dict[str, object]] = []
    for offset, quote_date in enumerate(quote_dates, start=1):
        rows.append(
            {
                "quote_date": quote_date,
                "effective_decision_timestamp": f"{quote_date.isoformat()}T15:45:00-05:00",
                "maturity_index": 0,
                "maturity_days": 30,
                "moneyness_index": 0,
                "moneyness_point": 0.0,
                "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
                "surface_grid_hash": grid.grid_hash,
                "surface_config_hash": "surface-hash",
                "maturity_coordinate": MATURITY_COORDINATE,
                "moneyness_coordinate": MONEYNESS_COORDINATE,
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
    assert feature_frame["effective_decision_timestamp"].to_list() == [
        "2021-01-04T15:45:00-05:00",
        "2021-01-06T15:45:00-05:00",
    ]
    assert feature_frame["target_effective_decision_timestamp"].to_list() == [
        "2021-01-06T15:45:00-05:00",
        "2021-01-07T15:45:00-05:00",
    ]


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


def test_build_target_training_weights_keeps_completed_only_cells_supervised() -> None:
    training_weights = build_target_training_weights(
        observed_mask=np.asarray([1.0, 0.0], dtype=np.float64),
        vega_weights=np.asarray([2.0, 0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(training_weights, np.asarray([2.0, 1.0], dtype=np.float64))


def test_build_target_training_weights_rejects_negative_vega_weights() -> None:
    with pytest.raises(ValueError, match="vega_weights must be nonnegative"):
        build_target_training_weights(
            observed_mask=np.asarray([1.0, 0.0], dtype=np.float64),
            vega_weights=np.asarray([2.0, -0.1], dtype=np.float64),
        )


def test_build_target_training_weights_rejects_nonbinary_masks() -> None:
    with pytest.raises(ValueError, match="binary 0/1"):
        build_target_training_weights(
            observed_mask=np.asarray([1.0, 0.5], dtype=np.float64),
            vega_weights=np.asarray([2.0, 0.0], dtype=np.float64),
        )


def test_feature_availability_manifest_rejects_undeclared_columns() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4)],
            "effective_decision_timestamp": ["2021-01-04T15:45:00-05:00"],
            "unknown_feature_contract_column": [1.0],
        }
    )

    with pytest.raises(ValueError, match="without declared availability metadata"):
        build_feature_availability_manifest(frame)
