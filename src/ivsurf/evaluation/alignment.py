"""Forecast-artifact alignment against realized surfaces and spot states."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from ivsurf.cleaning.derived_fields import DECISION_TIMESTAMP_COLUMN
from ivsurf.evaluation.metric_contracts import (
    require_binary_mask_array,
    require_nonnegative_weights,
    require_observed_weight_contract,
)
from ivsurf.evaluation.metrics import total_variance_to_iv, validate_total_variance_array
from ivsurf.features.availability import TARGET_DECISION_TIMESTAMP_COLUMN
from ivsurf.io.parquet import scan_parquet_files
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.surfaces.grid import (
    SurfaceGrid,
    require_complete_unique_surface_grid,
    require_surface_grid_metadata,
)
from ivsurf.surfaces.interpolation import (
    COMPLETION_STATUS_EXTRAPOLATED,
    COMPLETION_STATUS_INTERPOLATED,
)

FORECAST_DUPLICATE_KEY_COLUMNS = (
    "model_name",
    "quote_date",
    "target_date",
    "maturity_index",
    "moneyness_index",
)
FORECAST_COVERAGE_KEY_COLUMNS = (
    "quote_date",
    "target_date",
    "split_id",
    "maturity_index",
    "moneyness_index",
)


def _require_files(paths: list[Path], description: str) -> None:
    if not paths:
        message = f"No {description} files found."
        raise FileNotFoundError(message)


def _require_non_null_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        message = (
            "Aligned evaluation panel is missing required columns: "
            f"{', '.join(missing_columns)}."
        )
        raise ValueError(message)
    for column in columns:
        if frame[column].null_count() > 0:
            message = f"Aligned evaluation panel contains nulls in required column {column}."
            raise ValueError(message)


def _require_unique_keys(
    frame: pl.DataFrame,
    *,
    key_columns: tuple[str, ...],
    context: str,
) -> None:
    missing_columns = [column for column in key_columns if column not in frame.columns]
    if missing_columns:
        message = f"{context} is missing key columns: {', '.join(missing_columns)}."
        raise ValueError(message)
    duplicate_keys = (
        frame.group_by(key_columns)
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
        .sort(key_columns)
    )
    if duplicate_keys.is_empty():
        return
    preview = _format_key_preview(duplicate_keys, key_columns=key_columns)
    message = (
        f"{context} contains duplicate rows for key columns {key_columns!r}; "
        f"duplicate key count={duplicate_keys.height}. First duplicates: {preview}."
    )
    raise ValueError(message)


def _format_key_preview(frame: pl.DataFrame, *, key_columns: tuple[str, ...]) -> str:
    rows = []
    for row in frame.select(key_columns).head(5).iter_rows(named=True):
        rows.append(", ".join(f"{column}={row[column]!r}" for column in key_columns))
    return "; ".join(rows)


def assert_equal_forecast_model_coverage(forecast_frame: pl.DataFrame) -> None:
    """Fail fast unless every model forecasts the identical date-cell universe."""

    _require_unique_keys(
        forecast_frame,
        key_columns=FORECAST_DUPLICATE_KEY_COLUMNS,
        context="Forecast artifacts",
    )
    _require_non_null_columns(
        forecast_frame,
        columns=("model_name", *FORECAST_COVERAGE_KEY_COLUMNS),
    )
    model_names = tuple(sorted(str(value) for value in forecast_frame["model_name"].unique()))
    if not model_names:
        message = "Forecast artifacts must contain at least one model."
        raise ValueError(message)
    reference_model = model_names[0]
    reference_keys = (
        forecast_frame.filter(pl.col("model_name") == reference_model)
        .select(FORECAST_COVERAGE_KEY_COLUMNS)
        .unique()
        .sort(FORECAST_COVERAGE_KEY_COLUMNS)
    )
    for model_name in model_names[1:]:
        model_keys = (
            forecast_frame.filter(pl.col("model_name") == model_name)
            .select(FORECAST_COVERAGE_KEY_COLUMNS)
            .unique()
            .sort(FORECAST_COVERAGE_KEY_COLUMNS)
        )
        missing = reference_keys.join(model_keys, on=FORECAST_COVERAGE_KEY_COLUMNS, how="anti")
        extra = model_keys.join(reference_keys, on=FORECAST_COVERAGE_KEY_COLUMNS, how="anti")
        if missing.is_empty() and extra.is_empty():
            continue
        message = (
            "Forecast artifacts must have identical coverage across models. "
            f"Model {model_name!r} differs from reference model {reference_model!r}: "
            f"missing={missing.height}, extra={extra.height}. "
            "First missing: "
            f"{_format_key_preview(missing, key_columns=FORECAST_COVERAGE_KEY_COLUMNS)}. "
            f"First extra: {_format_key_preview(extra, key_columns=FORECAST_COVERAGE_KEY_COLUMNS)}."
        )
        raise ValueError(message)


def require_forecast_surface_grid(
    forecast_frame: pl.DataFrame,
    grid: SurfaceGrid,
    *,
    dataset_name: str = "Forecast artifacts",
) -> None:
    """Require forecast artifacts to match the configured fixed surface grid."""

    require_surface_grid_metadata(forecast_frame, grid, dataset_name=dataset_name)
    require_complete_unique_surface_grid(
        forecast_frame,
        grid,
        dataset_name=dataset_name,
        group_columns=("model_name", "quote_date", "target_date", "split_id"),
    )


def _require_equal_columns(
    frame: pl.DataFrame,
    *,
    left_column: str,
    right_column: str,
    context: str,
) -> None:
    mismatches = frame.filter(pl.col(left_column) != pl.col(right_column))
    if mismatches.is_empty():
        return
    message = (
        f"Aligned evaluation panel contains {mismatches.height} {context} mismatches "
        f"between {left_column} and {right_column}."
    )
    raise ValueError(message)


def _require_finite_float_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            message = f"Aligned evaluation panel contains non-finite values in column {column}."
            raise ValueError(message)


def _require_non_negative_float_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy().astype(np.float64, copy=False)
        validate_total_variance_array(
            values,
            context=f"Aligned evaluation panel column {column}",
            allow_zero=True,
        )


def _require_actual_vega_contract(frame: pl.DataFrame) -> None:
    _require_non_null_columns(
        frame,
        columns=("actual_observed_mask", "actual_vega_sum"),
    )
    mask = require_binary_mask_array(
        frame["actual_observed_mask"].to_numpy(),
        context="Aligned evaluation panel actual_observed_mask",
    )
    actual_vega_sum = require_nonnegative_weights(
        frame["actual_vega_sum"].to_numpy(),
        context="Aligned evaluation panel actual_vega_sum",
    )
    observed_nonpositive = mask & (actual_vega_sum <= 0.0)
    if observed_nonpositive.any():
        message = (
            "Aligned evaluation panel observed actual cells must have strictly positive "
            f"actual_vega_sum; invalid_count={int(observed_nonpositive.sum())}."
        )
        raise ValueError(message)


def _format_spot_contract_violations(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
) -> str:
    violations: list[str] = []
    for row in frame.select("quote_date", *columns).iter_rows(named=True):
        quote_date = cast(date, row["quote_date"])
        details = ", ".join(f"{column}={row[column]!r}" for column in columns)
        violations.append(f"{quote_date.isoformat()} ({details})")
    return "; ".join(violations)


def load_actual_surface_frame(gold_dir: Path, grid: SurfaceGrid) -> pl.DataFrame:
    """Load persisted daily surface artifacts."""

    gold_files = sorted_artifact_files(gold_dir, "year=*/*.parquet")
    _require_files(gold_files, "gold surface")
    frame = (
        scan_parquet_files(gold_files)
        .select(
            "quote_date",
            "maturity_index",
            "maturity_days",
            "moneyness_index",
            "moneyness_point",
            "observed_total_variance",
            "observed_iv",
            "completed_total_variance",
            "completed_iv",
            "observed_mask",
            "completion_status",
            DECISION_TIMESTAMP_COLUMN,
            "surface_grid_schema_version",
            "surface_grid_hash",
            "maturity_coordinate",
            "moneyness_coordinate",
            "target_surface_version",
            "surface_config_hash",
            "vega_sum",
        )
        .collect(engine="streaming")
        .sort(["quote_date", "maturity_index", "moneyness_index"])
    )
    _require_unique_keys(
        frame,
        key_columns=("quote_date", "maturity_index", "moneyness_index"),
        context="Gold surface artifacts",
    )
    require_surface_grid_metadata(frame, grid, dataset_name="Gold surface artifacts")
    require_complete_unique_surface_grid(
        frame,
        grid,
        dataset_name="Gold surface artifacts",
        group_columns=("quote_date",),
    )
    return frame


def load_forecast_frame(forecast_dir: Path, grid: SurfaceGrid) -> pl.DataFrame:
    """Load persisted forecast artifacts."""

    forecast_files = sorted_artifact_files(forecast_dir, "*.parquet")
    _require_files(forecast_files, "forecast")
    frame = (
        scan_parquet_files(forecast_files)
        .collect(engine="streaming")
        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
    )
    assert_equal_forecast_model_coverage(frame)
    require_forecast_surface_grid(frame, grid)
    return frame


def load_daily_spot_frame(silver_dir: Path) -> pl.DataFrame:
    """Load the official stage-08 daily spot from valid active_underlying_price_1545 rows."""

    silver_files = sorted_artifact_files(silver_dir, "year=*/*.parquet")
    _require_files(silver_files, "silver")
    lazy_frame = scan_parquet_files(silver_files)
    spot_frame = (
        lazy_frame.select("quote_date", "active_underlying_price_1545", "is_valid_observation")
        .filter(pl.col("is_valid_observation"))
        .group_by("quote_date")
        .agg(
            pl.len().alias("valid_spot_row_count"),
            pl.col("active_underlying_price_1545").n_unique().alias("active_spot_n_unique"),
            pl.col("active_underlying_price_1545").median().alias("spot_1545"),
            pl.col("active_underlying_price_1545").min().alias("active_spot_min"),
            pl.col("active_underlying_price_1545").max().alias("active_spot_max"),
        )
        .collect(engine="streaming")
        .sort("quote_date")
    )
    if spot_frame.height == 0:
        message = "No valid silver rows available to derive stage-08 daily spot states."
        raise ValueError(message)
    invalid_spot_frames = [
        spot_frame.filter(pl.col("valid_spot_row_count") <= 0),
        spot_frame.filter(
            pl.col("spot_1545").is_null()
            | (~pl.col("spot_1545").is_finite())
            | (pl.col("spot_1545") <= 0.0)
        ),
    ]
    invalid_spots = pl.concat(invalid_spot_frames).unique(subset=["quote_date"]).sort("quote_date")
    if invalid_spots.height > 0:
        message = (
            "Expected strictly positive finite stage-08 daily spot values derived from the "
            "median active_underlying_price_1545 across valid silver rows. Violations: "
            f"{_format_spot_contract_violations(invalid_spots, columns=('spot_1545',))}."
        )
        raise ValueError(message)
    return spot_frame.select("quote_date", "spot_1545")


def assert_forecast_origins_after_hpo_boundary(
    forecast_frame: pl.DataFrame,
    *,
    max_hpo_validation_date: date,
) -> None:
    """Fail fast if forecast artifacts include HPO-contaminated origin dates."""

    contaminated = forecast_frame.filter(pl.col("quote_date") <= pl.lit(max_hpo_validation_date))
    if contaminated.is_empty():
        return
    earliest_quote_date = cast(date, contaminated["quote_date"].min())
    latest_quote_date = cast(date, contaminated["quote_date"].max())
    earliest_target_date = cast(date, contaminated["target_date"].min())
    latest_target_date = cast(date, contaminated["target_date"].max())
    message = (
        "Forecast artifacts include quote_date values that were inside the HPO-used validation "
        f"sample. Found {contaminated.height} contaminated rows with quote_date range "
        f"[{earliest_quote_date.isoformat()}, {latest_quote_date.isoformat()}], "
        f"target_date range "
        f"[{earliest_target_date.isoformat()}, {latest_target_date.isoformat()}], and boundary "
        f"{max_hpo_validation_date.isoformat()}."
    )
    raise ValueError(message)


def build_forecast_realization_panel(
    actual_surface_frame: pl.DataFrame,
    forecast_frame: pl.DataFrame,
    *,
    total_variance_floor: float,
) -> pl.DataFrame:
    """Align forecast artifacts with realized target-day surfaces and origin-day references."""

    _require_unique_keys(
        actual_surface_frame,
        key_columns=("quote_date", "maturity_index", "moneyness_index"),
        context="Actual surface frame",
    )
    assert_equal_forecast_model_coverage(forecast_frame)
    actual_target = actual_surface_frame.rename(
        {
            "quote_date": "target_date",
            "observed_total_variance": "actual_observed_total_variance",
            "observed_iv": "actual_observed_iv",
            "completed_total_variance": "actual_completed_total_variance",
            "completed_iv": "actual_completed_iv",
            "observed_mask": "actual_observed_mask",
            "completion_status": "actual_completion_status",
            DECISION_TIMESTAMP_COLUMN: f"actual_{DECISION_TIMESTAMP_COLUMN}",
            "surface_grid_schema_version": "actual_surface_grid_schema_version",
            "surface_grid_hash": "actual_surface_grid_hash",
            "maturity_coordinate": "actual_maturity_coordinate",
            "moneyness_coordinate": "actual_moneyness_coordinate",
            "target_surface_version": "actual_target_surface_version",
            "surface_config_hash": "actual_surface_config_hash",
            "vega_sum": "actual_vega_sum",
        }
    )
    origin_surface = actual_surface_frame.rename(
        {
            "completed_total_variance": "origin_completed_total_variance",
            "completed_iv": "origin_completed_iv",
            "completion_status": "origin_completion_status",
            DECISION_TIMESTAMP_COLUMN: f"origin_{DECISION_TIMESTAMP_COLUMN}",
            "surface_grid_schema_version": "origin_surface_grid_schema_version",
            "surface_grid_hash": "origin_surface_grid_hash",
            "maturity_coordinate": "origin_maturity_coordinate",
            "moneyness_coordinate": "origin_moneyness_coordinate",
            "target_surface_version": "origin_target_surface_version",
            "surface_config_hash": "origin_surface_config_hash",
        }
    ).select(
        "quote_date",
        "maturity_index",
        "moneyness_index",
        "origin_completed_total_variance",
        "origin_completed_iv",
        "origin_completion_status",
        f"origin_{DECISION_TIMESTAMP_COLUMN}",
        "origin_surface_grid_schema_version",
        "origin_surface_grid_hash",
        "origin_maturity_coordinate",
        "origin_moneyness_coordinate",
        "origin_target_surface_version",
        "origin_surface_config_hash",
    )

    joined_panel = (
        forecast_frame.join(
            actual_target,
            on=["target_date", "maturity_index", "moneyness_index"],
            how="left",
            validate="m:1",
        )
        .join(
            origin_surface,
            on=["quote_date", "maturity_index", "moneyness_index"],
            how="left",
            validate="m:1",
        )
    )

    _require_non_null_columns(
        joined_panel,
        columns=(
            "actual_completed_total_variance",
            "actual_completed_iv",
            "origin_completed_iv",
            "actual_completion_status",
            "origin_completion_status",
            DECISION_TIMESTAMP_COLUMN,
            TARGET_DECISION_TIMESTAMP_COLUMN,
            f"actual_{DECISION_TIMESTAMP_COLUMN}",
            f"origin_{DECISION_TIMESTAMP_COLUMN}",
            "surface_grid_schema_version",
            "surface_grid_hash",
            "maturity_coordinate",
            "moneyness_coordinate",
            "target_surface_version",
            "split_id",
            "surface_config_hash",
            "model_config_hash",
            "training_run_id",
            "actual_surface_grid_schema_version",
            "actual_surface_grid_hash",
            "actual_maturity_coordinate",
            "actual_moneyness_coordinate",
            "actual_target_surface_version",
            "actual_surface_config_hash",
            "origin_surface_grid_schema_version",
            "origin_surface_grid_hash",
            "origin_maturity_coordinate",
            "origin_moneyness_coordinate",
            "origin_target_surface_version",
            "origin_surface_config_hash",
            "predicted_total_variance",
        ),
    )
    for left_column, right_column, context in (
        (
            TARGET_DECISION_TIMESTAMP_COLUMN,
            f"actual_{DECISION_TIMESTAMP_COLUMN}",
            "target decision timestamp",
        ),
        (
            DECISION_TIMESTAMP_COLUMN,
            f"origin_{DECISION_TIMESTAMP_COLUMN}",
            "origin decision timestamp",
        ),
        ("surface_grid_schema_version", "actual_surface_grid_schema_version", "target grid"),
        ("surface_grid_hash", "actual_surface_grid_hash", "target grid"),
        ("maturity_coordinate", "actual_maturity_coordinate", "target maturity coordinate"),
        ("moneyness_coordinate", "actual_moneyness_coordinate", "target moneyness coordinate"),
        ("target_surface_version", "actual_target_surface_version", "target surface version"),
        ("surface_config_hash", "actual_surface_config_hash", "target surface config"),
        ("surface_grid_schema_version", "origin_surface_grid_schema_version", "origin grid"),
        ("surface_grid_hash", "origin_surface_grid_hash", "origin grid"),
        ("maturity_coordinate", "origin_maturity_coordinate", "origin maturity coordinate"),
        ("moneyness_coordinate", "origin_moneyness_coordinate", "origin moneyness coordinate"),
        ("target_surface_version", "origin_target_surface_version", "origin surface version"),
        ("surface_config_hash", "origin_surface_config_hash", "origin surface config"),
    ):
        _require_equal_columns(
            joined_panel,
            left_column=left_column,
            right_column=right_column,
            context=context,
        )
    _require_finite_float_columns(joined_panel, columns=("predicted_total_variance",))
    _require_non_negative_float_columns(joined_panel, columns=("predicted_total_variance",))
    _require_actual_vega_contract(joined_panel)

    panel = (
        panel_with_completed_iv(
            joined_panel,
            maturity_days_column="maturity_days",
            total_variance_column="predicted_total_variance",
            output_iv_column="predicted_iv",
            total_variance_floor=total_variance_floor,
        )
        .with_columns(
            (
                pl.col("actual_completed_iv") - pl.col("origin_completed_iv")
            ).alias("actual_iv_change"),
            (
                pl.col("predicted_iv") - pl.col("origin_completed_iv")
            ).alias("predicted_iv_change"),
            # Completed surfaces are the forecast target, while observed-mask x vega defines
            # the official observed-cell evaluation slice used for headline metrics.
            pl.when(pl.col("actual_observed_mask"))
            .then(pl.col("actual_vega_sum"))
            .otherwise(0.0)
            .alias("observed_weight"),
            pl.when(pl.col("actual_completion_status") == COMPLETION_STATUS_INTERPOLATED)
            .then(1.0)
            .otherwise(0.0)
            .alias("interpolated_weight"),
            pl.when(pl.col("actual_completion_status") == COMPLETION_STATUS_EXTRAPOLATED)
            .then(1.0)
            .otherwise(0.0)
            .alias("extrapolated_weight"),
            pl.lit(1.0).alias("full_grid_weight"),
        )
        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
    )

    _require_non_null_columns(
        panel,
        columns=(
            "predicted_iv",
            "predicted_iv_change",
        ),
    )
    _require_finite_float_columns(
        panel,
        columns=("predicted_total_variance", "predicted_iv", "predicted_iv_change"),
    )
    require_observed_weight_contract(
        observed_mask=panel["actual_observed_mask"].to_numpy(),
        observed_weight=panel["observed_weight"].to_numpy(),
        context="Aligned evaluation panel",
    )
    return panel


def panel_with_completed_iv(
    frame: pl.DataFrame,
    maturity_days_column: str,
    total_variance_column: str,
    output_iv_column: str,
    *,
    total_variance_floor: float,
) -> pl.DataFrame:
    """Add an IV column derived from total variance and maturity days."""

    maturity_years = (
        frame[maturity_days_column].to_numpy().astype(np.float64) / 365.0
    ).reshape(-1, 1)
    total_variance = validate_total_variance_array(
        frame[total_variance_column].to_numpy().astype(np.float64),
        context=f"Aligned panel column {total_variance_column}",
        allow_zero=True,
    ).reshape(-1, 1)
    iv = total_variance_to_iv(
        total_variance=total_variance,
        maturity_years=maturity_years,
        total_variance_floor=total_variance_floor,
    ).reshape(-1)
    return frame.with_columns(pl.Series(output_iv_column, iv))
