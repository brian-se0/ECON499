"""Fail-fast hedging evaluation domain validation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from math import isclose, isfinite, log
from typing import cast

import polars as pl

from ivsurf.config import HedgingConfig
from ivsurf.hedging.pnl import summarize_hedging_results
from ivsurf.surfaces.grid import SurfaceGrid

HEDGING_COVERAGE_KEY_COLUMNS = ("model_name", "quote_date", "target_date")
HEDGING_SUMMARY_COLUMNS = (
    "model_name",
    "mean_abs_revaluation_error",
    "mean_squared_revaluation_error",
    "mean_abs_hedged_pnl",
    "mean_squared_hedged_pnl",
    "hedged_pnl_variance",
    "n_trades",
)
HEDGING_SUMMARY_FLOAT_TOLERANCE = 1.0e-12
HEDGING_DOMAIN_ABSOLUTE_TOLERANCE = 1.0e-12


def _require_columns(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        message = f"{artifact_name} is missing required columns: {', '.join(missing)}."
        raise ValueError(message)


def _require_non_null_columns(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    _require_columns(frame, columns=columns, artifact_name=artifact_name)
    for column in columns:
        if frame[column].null_count() > 0:
            message = f"{artifact_name} contains nulls in required column {column}."
            raise ValueError(message)


def _format_key_preview(frame: pl.DataFrame, *, key_columns: tuple[str, ...]) -> str:
    rows = []
    for row in frame.select(key_columns).head(5).iter_rows(named=True):
        rows.append(", ".join(f"{column}={row[column]!r}" for column in key_columns))
    return "; ".join(rows)


def _require_unique_keys(
    frame: pl.DataFrame,
    *,
    key_columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    _require_columns(frame, columns=key_columns, artifact_name=artifact_name)
    duplicate_keys = (
        frame.group_by(key_columns)
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
        .sort(key_columns)
    )
    if duplicate_keys.is_empty():
        return
    message = (
        f"{artifact_name} contains duplicate rows for key columns {key_columns!r}; "
        f"duplicate key count={duplicate_keys.height}. First duplicates: "
        f"{_format_key_preview(duplicate_keys, key_columns=key_columns)}."
    )
    raise ValueError(message)


def _values_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_is_numeric = isinstance(left, int | float) and not isinstance(left, bool)
    right_is_numeric = isinstance(right, int | float) and not isinstance(right, bool)
    if left_is_numeric or right_is_numeric:
        if not (left_is_numeric and right_is_numeric):
            return False
        left_float = float(cast(int | float, left))
        right_float = float(cast(int | float, right))
        if not isfinite(left_float) or not isfinite(right_float):
            return False
        return isclose(
            left_float,
            right_float,
            rel_tol=HEDGING_SUMMARY_FLOAT_TOLERANCE,
            abs_tol=HEDGING_SUMMARY_FLOAT_TOLERANCE,
        )
    return left == right


def require_hedging_results_match_forecast_coverage(
    hedging_results: pl.DataFrame,
    forecast_frame: pl.DataFrame,
) -> None:
    """Fail unless hedging results cover the exact clean forecast model/date universe."""

    _require_non_null_columns(
        forecast_frame,
        columns=HEDGING_COVERAGE_KEY_COLUMNS,
        artifact_name="Forecast artifacts",
    )
    _require_non_null_columns(
        hedging_results,
        columns=HEDGING_COVERAGE_KEY_COLUMNS,
        artifact_name="Hedging results",
    )
    _require_unique_keys(
        hedging_results,
        key_columns=HEDGING_COVERAGE_KEY_COLUMNS,
        artifact_name="Hedging results",
    )
    forecast_keys = (
        forecast_frame.select(HEDGING_COVERAGE_KEY_COLUMNS)
        .unique()
        .sort(HEDGING_COVERAGE_KEY_COLUMNS)
    )
    hedging_keys = (
        hedging_results.select(HEDGING_COVERAGE_KEY_COLUMNS)
        .unique()
        .sort(HEDGING_COVERAGE_KEY_COLUMNS)
    )
    if forecast_keys.is_empty():
        message = "Forecast artifacts must contain at least one model/date pair for hedging."
        raise ValueError(message)
    if hedging_keys.is_empty():
        message = "Hedging results must contain at least one model/date pair."
        raise ValueError(message)
    missing = forecast_keys.join(hedging_keys, on=HEDGING_COVERAGE_KEY_COLUMNS, how="anti")
    extra = hedging_keys.join(forecast_keys, on=HEDGING_COVERAGE_KEY_COLUMNS, how="anti")
    if missing.is_empty() and extra.is_empty():
        return
    message = (
        "Hedging results must match the clean forecast model/date coverage exactly; "
        f"missing={missing.height}, extra={extra.height}. First missing: "
        f"{_format_key_preview(missing, key_columns=HEDGING_COVERAGE_KEY_COLUMNS)}. "
        f"First extra: {_format_key_preview(extra, key_columns=HEDGING_COVERAGE_KEY_COLUMNS)}."
    )
    raise ValueError(message)


def require_hedging_summary_matches_results(
    hedging_summary: pl.DataFrame,
    hedging_results: pl.DataFrame,
) -> None:
    """Fail unless the hedging summary is an exact model-level aggregate of results."""

    _require_non_null_columns(
        hedging_summary,
        columns=("model_name",),
        artifact_name="Hedging summary",
    )
    _require_columns(
        hedging_summary,
        columns=HEDGING_SUMMARY_COLUMNS,
        artifact_name="Hedging summary",
    )
    _require_non_null_columns(
        hedging_results,
        columns=HEDGING_COVERAGE_KEY_COLUMNS,
        artifact_name="Hedging results",
    )
    _require_unique_keys(
        hedging_summary,
        key_columns=("model_name",),
        artifact_name="Hedging summary",
    )
    expected = summarize_hedging_results(hedging_results).select(HEDGING_SUMMARY_COLUMNS).sort(
        "model_name"
    )
    observed = hedging_summary.select(HEDGING_SUMMARY_COLUMNS).sort("model_name")
    missing = expected.select("model_name").join(observed, on="model_name", how="anti")
    extra = observed.select("model_name").join(expected, on="model_name", how="anti")
    mismatch_rows: list[str] = []
    if missing.is_empty() and extra.is_empty():
        expected_by_model = {
            str(row["model_name"]): row for row in expected.iter_rows(named=True)
        }
        for observed_row in observed.iter_rows(named=True):
            model_name = str(observed_row["model_name"])
            expected_row = expected_by_model[model_name]
            mismatched_columns = [
                column
                for column in HEDGING_SUMMARY_COLUMNS[1:]
                if not _values_equal(observed_row[column], expected_row[column])
            ]
            if mismatched_columns:
                mismatch_rows.append(
                    f"model_name={model_name!r}, mismatched_columns={mismatched_columns!r}"
                )
    if missing.is_empty() and extra.is_empty() and not mismatch_rows:
        return
    message = (
        "Hedging summary must exactly match the recomputed hedging result aggregate; "
        f"missing_models={missing.height}, extra_models={extra.height}, "
        f"mismatched_models={len(mismatch_rows)}. "
        f"First missing models: {_format_key_preview(missing, key_columns=('model_name',))}. "
        f"First extra models: {_format_key_preview(extra, key_columns=('model_name',))}. "
        f"First mismatches: {'; '.join(mismatch_rows[:5])}."
    )
    raise ValueError(message)


def _require_value_in_bounds(
    *,
    value: float,
    lower: float,
    upper: float,
    field_name: str,
) -> None:
    if lower <= value <= upper:
        return
    if isclose(value, lower, rel_tol=0.0, abs_tol=HEDGING_DOMAIN_ABSOLUTE_TOLERANCE):
        return
    if isclose(value, upper, rel_tol=0.0, abs_tol=HEDGING_DOMAIN_ABSOLUTE_TOLERANCE):
        return
    message = (
        f"Hedging config {field_name}={value!r} is outside the surface grid domain "
        f"[{lower!r}, {upper!r}]."
    )
    raise ValueError(message)


def _format_date_for_message(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return repr(value)


def _require_positive_spot_value(
    value: object,
    *,
    field_name: str,
    date_value: object,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = (
            f"Hedging spot path validation requires numeric {field_name} for "
            f"{_format_date_for_message(date_value)}; got {value!r}."
        )
        raise ValueError(message)
    spot = float(value)
    if isfinite(spot) and spot > 0.0:
        return spot
    message = (
        f"Hedging spot path validation requires finite positive {field_name} for "
        f"{_format_date_for_message(date_value)}; got {spot!r}."
    )
    raise ValueError(message)


def require_hedging_config_in_surface_domain(
    hedging_config: HedgingConfig,
    grid: SurfaceGrid,
    *,
    max_target_gap_days: int,
) -> None:
    """Fail before official hedging artifacts when config cannot fit the grid."""

    if max_target_gap_days <= 0:
        message = f"max_target_gap_days must be positive; got {max_target_gap_days!r}."
        raise ValueError(message)
    maturity_min = float(min(grid.maturity_days))
    maturity_max = float(max(grid.maturity_days))
    moneyness_min = float(min(grid.moneyness_points))
    moneyness_max = float(max(grid.moneyness_points))
    for field_name, maturity_days in (
        ("short_maturity_days", hedging_config.short_maturity_days),
        ("long_maturity_days", hedging_config.long_maturity_days),
        ("hedge_maturity_days", hedging_config.hedge_maturity_days),
    ):
        _require_value_in_bounds(
            value=float(maturity_days),
            lower=maturity_min,
            upper=maturity_max,
            field_name=field_name,
        )
        target_remaining_days = float(maturity_days - max_target_gap_days)
        _require_value_in_bounds(
            value=target_remaining_days,
            lower=maturity_min,
            upper=maturity_max,
            field_name=f"{field_name}_after_max_target_gap",
        )
    for field_name, moneyness in (
        ("skew_moneyness_abs", hedging_config.skew_moneyness_abs),
        ("negative_skew_moneyness_abs", -hedging_config.skew_moneyness_abs),
        ("hedge_straddle_moneyness", hedging_config.hedge_straddle_moneyness),
    ):
        _require_value_in_bounds(
            value=float(moneyness),
            lower=moneyness_min,
            upper=moneyness_max,
            field_name=field_name,
        )


def require_hedging_spot_paths_in_surface_domain(
    hedging_config: HedgingConfig,
    grid: SurfaceGrid,
    *,
    forecast_frame: pl.DataFrame,
    spot_lookup: Mapping[object, float],
) -> None:
    """Fail if carried hedging-book strikes can leave the surface moneyness domain."""

    _require_non_null_columns(
        forecast_frame,
        columns=("quote_date", "target_date"),
        artifact_name="Forecast artifacts",
    )
    forecast_paths = forecast_frame.select(["quote_date", "target_date"]).unique().sort(
        ["quote_date", "target_date"]
    )
    if forecast_paths.is_empty():
        message = "Forecast artifacts must contain at least one quote/target path."
        raise ValueError(message)
    moneyness_min = float(min(grid.moneyness_points))
    moneyness_max = float(max(grid.moneyness_points))
    for row in forecast_paths.iter_rows(named=True):
        quote_date = row["quote_date"]
        target_date = row["target_date"]
        if quote_date not in spot_lookup or target_date not in spot_lookup:
            message = (
                "Hedging spot path validation is missing spot state for "
                f"quote_date={_format_date_for_message(quote_date)} or "
                f"target_date={_format_date_for_message(target_date)}."
            )
            raise ValueError(message)
        quote_spot = _require_positive_spot_value(
            spot_lookup[quote_date],
            field_name="quote_spot",
            date_value=quote_date,
        )
        target_spot = _require_positive_spot_value(
            spot_lookup[target_date],
            field_name="target_spot",
            date_value=target_date,
        )
        spot_log_shift = log(quote_spot / target_spot)
        target_moneyness_values = (
            ("atm_target_log_moneyness", spot_log_shift),
            (
                "rr_put_target_log_moneyness",
                spot_log_shift - hedging_config.skew_moneyness_abs,
            ),
            (
                "rr_call_target_log_moneyness",
                spot_log_shift + hedging_config.skew_moneyness_abs,
            ),
            (
                "hedge_straddle_target_log_moneyness",
                spot_log_shift + hedging_config.hedge_straddle_moneyness,
            ),
        )
        for field_name, value in target_moneyness_values:
            if (
                moneyness_min <= value <= moneyness_max
                or isclose(
                    value,
                    moneyness_min,
                    rel_tol=0.0,
                    abs_tol=HEDGING_DOMAIN_ABSOLUTE_TOLERANCE,
                )
                or isclose(
                    value,
                    moneyness_max,
                    rel_tol=0.0,
                    abs_tol=HEDGING_DOMAIN_ABSOLUTE_TOLERANCE,
                )
            ):
                continue
            message = (
                "Hedging spot path moves a carried instrument outside the surface "
                "moneyness domain: "
                f"{field_name}={value!r}, moneyness_bounds=[{moneyness_min!r}, "
                f"{moneyness_max!r}], quote_date={_format_date_for_message(quote_date)}, "
                f"target_date={_format_date_for_message(target_date)}, "
                f"quote_spot={quote_spot!r}, target_spot={target_spot!r}."
            )
            raise ValueError(message)
