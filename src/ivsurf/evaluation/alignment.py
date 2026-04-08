"""Forecast-artifact alignment against realized surfaces and spot states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import total_variance_to_iv


def _require_files(paths: list[Path], description: str) -> None:
    if not paths:
        message = f"No {description} files found."
        raise FileNotFoundError(message)


def load_actual_surface_frame(gold_dir: Path) -> pl.DataFrame:
    """Load persisted daily surface artifacts."""

    gold_files = sorted(gold_dir.glob("year=*/*.parquet"))
    _require_files(gold_files, "gold surface")
    return pl.concat(
        [
            pl.read_parquet(path).select(
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
            for path in gold_files
        ]
    ).sort(["quote_date", "maturity_index", "moneyness_index"])


def load_forecast_frame(forecast_dir: Path) -> pl.DataFrame:
    """Load persisted forecast artifacts."""

    forecast_files = sorted(forecast_dir.glob("*.parquet"))
    _require_files(forecast_files, "forecast")
    return pl.concat([pl.read_parquet(path) for path in forecast_files]).sort(
        ["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"]
    )


def load_daily_spot_frame(silver_dir: Path) -> pl.DataFrame:
    """Load one explicit 15:45 spot per date from the silver option panel."""

    silver_files = sorted(silver_dir.glob("year=*/*.parquet"))
    _require_files(silver_files, "silver")
    lazy_frame = pl.scan_parquet([str(path) for path in silver_files])
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

    panel = (
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
        .with_columns(
            (
                pl.col("predicted_total_variance")
                / (pl.col("maturity_days").cast(pl.Float64) / 365.0)
            )
            .sqrt()
            .alias("predicted_iv")
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

    required_columns = (
        "actual_completed_total_variance",
        "actual_completed_iv",
        "origin_completed_iv",
    )
    for column in required_columns:
        if panel[column].null_count() > 0:
            message = f"Aligned evaluation panel contains nulls in required column {column}."
            raise ValueError(message)
    return panel


def panel_with_completed_iv(
    frame: pl.DataFrame,
    maturity_days_column: str,
    total_variance_column: str,
    output_iv_column: str,
) -> pl.DataFrame:
    """Add an IV column derived from total variance and maturity days."""

    maturity_years = (
        frame[maturity_days_column].to_numpy().astype(np.float64) / 365.0
    ).reshape(-1, 1)
    total_variance = frame[total_variance_column].to_numpy().astype(np.float64).reshape(-1, 1)
    iv = total_variance_to_iv(
        total_variance=total_variance,
        maturity_years=maturity_years,
    ).reshape(-1)
    return frame.with_columns(pl.Series(output_iv_column, iv))

