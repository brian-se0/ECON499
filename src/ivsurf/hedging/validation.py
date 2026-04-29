"""Fail-fast hedging evaluation domain validation."""

from __future__ import annotations

from math import isclose, isfinite

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
    if isinstance(left, float) or isinstance(right, float):
        left_float = float(left)
        right_float = float(right)
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
    message = (
        f"Hedging config {field_name}={value!r} is outside the surface grid domain "
        f"[{lower!r}, {upper!r}]."
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
