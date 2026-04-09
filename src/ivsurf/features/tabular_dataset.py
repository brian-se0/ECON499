"""Model-ready daily feature and target dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from ivsurf.calendar import MarketCalendar
from ivsurf.config import FeatureConfig, MarketCalendarConfig
from ivsurf.features.lagged_surface import pivot_surface_arrays, summarize_lag_window
from ivsurf.features.liquidity import build_daily_liquidity_features
from ivsurf.qc.timing_checks import (
    assert_next_observed_target_date,
    assert_strictly_increasing_unique_dates,
)
from ivsurf.surfaces.grid import SurfaceGrid


@dataclass(frozen=True, slots=True)
class DailyDatasetBuildResult:
    """Feature/target table plus metadata."""

    feature_frame: pl.DataFrame


def _vector_columns(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{index:04d}": float(value) for index, value in enumerate(values)}


def _count_intervening_trading_sessions(
    calendar: MarketCalendar,
    *,
    quote_date: date,
    target_date: date,
) -> int:
    """Count trading sessions strictly between quote_date and target_date."""

    if target_date <= quote_date:
        message = "target_date must be after quote_date when counting target-gap sessions."
        raise ValueError(message)

    intervening_sessions = 0
    current_date = quote_date
    while True:
        current_date = calendar.next_trading_session(current_date)
        if current_date == target_date:
            return intervening_sessions
        if current_date > target_date:
            message = (
                "target_date must be a trading session reachable from quote_date: "
                f"quote_date={quote_date.isoformat()} "
                f"target_date={target_date.isoformat()}."
            )
            raise ValueError(message)
        intervening_sessions += 1


def build_daily_feature_dataset(
    surface_frame: pl.DataFrame,
    grid: SurfaceGrid,
    feature_config: FeatureConfig,
    calendar_config: MarketCalendarConfig,
) -> DailyDatasetBuildResult:
    """Build one model-ready row per quote date."""

    surface_arrays = pivot_surface_arrays(surface_frame=surface_frame, grid=grid)
    liquidity_frame = build_daily_liquidity_features(surface_frame)
    liquidity_by_date = {row["quote_date"]: row for row in liquidity_frame.iter_rows(named=True)}
    calendar = MarketCalendar(
        calendar_name=calendar_config.calendar_name,
        timezone=calendar_config.timezone,
        decision_time=calendar_config.decision_time,
        decision_snapshot_minutes_before_close=calendar_config.decision_snapshot_minutes_before_close,
        am_settled_roots=calendar_config.am_settled_roots,
    )

    max_window = max(feature_config.lag_windows)
    required_surface_dates = max_window + 1
    if feature_config.include_daily_change:
        required_surface_dates = max(required_surface_dates, 3)
    if len(surface_arrays.quote_dates) < required_surface_dates:
        message = (
            "Not enough daily surfaces to build a supervised feature dataset. "
            f"Need at least {required_surface_dates} decision-time-aligned dates, "
            f"received {len(surface_arrays.quote_dates)}."
        )
        raise ValueError(message)

    assert_strictly_increasing_unique_dates(
        surface_arrays.quote_dates,
        context="Observed gold-surface quote dates",
    )

    rows: list[dict[str, object]] = []
    start_position = max_window - 1
    if feature_config.include_daily_change:
        start_position = max(start_position, 1)
    for position in range(start_position, len(surface_arrays.quote_dates) - 1):
        quote_date = surface_arrays.quote_dates[position]
        target_date = surface_arrays.quote_dates[position + 1]
        assert_next_observed_target_date(
            surface_arrays.quote_dates,
            position=position,
            quote_date=quote_date,
            target_date=target_date,
        )

        row: dict[str, object] = {
            "quote_date": quote_date,
            "target_date": target_date,
            "target_gap_sessions": _count_intervening_trading_sessions(
                calendar,
                quote_date=quote_date,
                target_date=target_date,
            ),
        }
        for window in feature_config.lag_windows:
            lag_surface = summarize_lag_window(surface_arrays.completed_surfaces, position, window)
            row.update(_vector_columns(f"feature_surface_mean_{window:02d}", lag_surface))
            if feature_config.include_mask:
                lag_mask = summarize_lag_window(surface_arrays.observed_masks, position, window)
                row.update(_vector_columns(f"feature_mask_mean_{window:02d}", lag_mask))

        if feature_config.include_daily_change:
            change_vector = (
                surface_arrays.completed_surfaces[position]
                - surface_arrays.completed_surfaces[position - 1]
            )
            row.update(_vector_columns("feature_surface_change_01", change_vector))

        if feature_config.include_liquidity:
            if quote_date not in liquidity_by_date:
                message = f"Missing liquidity features for quote_date={quote_date.isoformat()}."
                raise KeyError(message)
            daily_liquidity = liquidity_by_date[quote_date]
            row["feature_coverage_ratio"] = float(daily_liquidity["coverage_ratio"])
            row["feature_daily_vega_sum"] = float(daily_liquidity["daily_vega_sum"])
            row["feature_daily_option_count"] = int(daily_liquidity["daily_option_count"])
            row["feature_daily_weighted_spread_1545"] = float(
                daily_liquidity["daily_weighted_spread_1545"]
            )

        target_surface = surface_arrays.completed_surfaces[position + 1]
        target_mask = surface_arrays.observed_masks[position + 1]
        target_vega = surface_arrays.vega_weights[position + 1]
        row.update(_vector_columns("target_total_variance", target_surface))
        row.update(_vector_columns("target_observed_mask", target_mask))
        row.update(_vector_columns("target_vega_weight", target_vega))
        rows.append(row)

    return DailyDatasetBuildResult(feature_frame=pl.DataFrame(rows).sort("quote_date"))
