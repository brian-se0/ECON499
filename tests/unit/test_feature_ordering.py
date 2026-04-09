from __future__ import annotations

import polars as pl
import pytest

from ivsurf.config import FeatureConfig
from ivsurf.models.base import ordered_feature_columns
from ivsurf.models.no_change import validate_no_change_feature_layout


def test_ordered_feature_columns_follow_available_lag_windows() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [None],
            "target_date": [None],
            "target_gap_sessions": [0],
            "feature_mask_mean_10_0000": [0.0],
            "feature_surface_mean_10_0000": [0.30],
            "feature_surface_mean_01_0000": [0.10],
            "feature_mask_mean_03_0000": [0.0],
            "feature_surface_change_01_0000": [0.02],
            "feature_daily_option_count": [100],
            "feature_surface_mean_03_0000": [0.20],
            "feature_mask_mean_01_0000": [1.0],
            "target_total_variance_0000": [0.11],
            "target_observed_mask_0000": [1.0],
            "target_vega_weight_0000": [1.0],
        }
    )

    feature_columns = ordered_feature_columns(frame)

    assert feature_columns == (
        "feature_surface_mean_01_0000",
        "feature_surface_mean_03_0000",
        "feature_surface_mean_10_0000",
        "feature_surface_change_01_0000",
        "feature_mask_mean_01_0000",
        "feature_mask_mean_03_0000",
        "feature_mask_mean_10_0000",
        "feature_daily_option_count",
    )
    validate_no_change_feature_layout(
        feature_columns=feature_columns,
        target_columns=("target_total_variance_0000",),
    )


def test_feature_config_requires_lag_one_for_no_change_benchmark() -> None:
    with pytest.raises(ValueError, match="must include 1"):
        FeatureConfig.model_validate(
            {
                "lag_windows": [5, 22],
                "include_daily_change": True,
                "include_mask": True,
                "include_liquidity": True,
            }
        )
