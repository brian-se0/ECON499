"""Forecast-artifact alignment against realized surfaces and spot states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import total_variance_to_iv, validate_total_variance_array
from ivsurf.io.parquet import scan_parquet_files


def _require_files(paths: list[Path], description: str) -> None:
    if not paths:
        message = f"No {description} files found."
        raise FileNotFoundError(message)


def _require_non_null_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        if frame[column].null_count() > 0:
            message = f"Aligned evaluation panel contains nulls in required column {column}."
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


def load_actual_surface_frame(gold_dir: Path) -> pl.DataFrame:
    """Load persisted daily surface artifacts."""

    gold_files = sorted(gold_dir.glob("year=*/*.parquet"))
    _require_files(gold_files, "gold surface")
    return (
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
            "vega_sum",
        )
        .collect(engine="streaming")
        .sort(["quote_date", "maturity_index", "moneyness_index"])
    )


def load_forecast_frame(forecast_dir: Path) -> pl.DataFrame:
    """Load persisted forecast artifacts."""

    forecast_files = sorted(forecast_dir.glob("*.parquet"))
    _require_files(forecast_files, "forecast")
    return (
        scan_parquet_files(forecast_files)
        .collect(engine="streaming")
        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
    )


def load_daily_spot_frame(silver_dir: Path) -> pl.DataFrame:
    """Load one explicit decision-snapshot spot per date from the silver option panel."""

    silver_files = sorted(silver_dir.glob("year=*/*.parquet"))
    _require_files(silver_files, "silver")
    lazy_frame = scan_parquet_files(silver_files)
    spot_frame = (
        lazy_frame.select("quote_date", "active_underlying_price_1545")
        .group_by("quote_date")
        .agg(
            pl.col("active_underlying_price_1545").n_unique().alias("spot_n_unique"),
            pl.col("active_underlying_price_1545").first().alias("spot_1545"),
        )
        .collect(engine="streaming")
        .sort("quote_date")
    )
    if spot_frame.filter(pl.col("spot_n_unique") != 1).height > 0:
        message = "Expected exactly one active_underlying_price_1545 per quote_date in silver data."
        raise ValueError(message)
    return spot_frame.drop("spot_n_unique")


def build_forecast_realization_panel(
    actual_surface_frame: pl.DataFrame,
    forecast_frame: pl.DataFrame,
    *,
    total_variance_floor: float,
) -> pl.DataFrame:
    """Align forecast artifacts with realized target-day surfaces and origin-day references."""

    actual_target = actual_surface_frame.rename(
        {
            "quote_date": "target_date",
            "observed_total_variance": "actual_observed_total_variance",
            "observed_iv": "actual_observed_iv",
            "completed_total_variance": "actual_completed_total_variance",
            "completed_iv": "actual_completed_iv",
            "observed_mask": "actual_observed_mask",
            "vega_sum": "actual_vega_sum",
        }
    )
    origin_surface = actual_surface_frame.rename(
        {
            "completed_total_variance": "origin_completed_total_variance",
            "completed_iv": "origin_completed_iv",
        }
    ).select(
        "quote_date",
        "maturity_index",
        "moneyness_index",
        "origin_completed_total_variance",
        "origin_completed_iv",
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
            "predicted_total_variance",
        ),
    )
    _require_finite_float_columns(joined_panel, columns=("predicted_total_variance",))
    _require_non_negative_float_columns(joined_panel, columns=("predicted_total_variance",))

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
            pl.when(pl.col("actual_observed_mask"))
            .then(pl.col("actual_vega_sum"))
            .otherwise(0.0)
            .alias("observed_weight"),
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
