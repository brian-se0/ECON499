"""Explicit thesis sample-window validation helpers."""

from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.config import RawDataConfig
from ivsurf.exceptions import DataValidationError


def sample_window_label(config: RawDataConfig) -> str:
    """Render the configured inclusive sample window."""

    return (
        f"{config.sample_start_date.isoformat()} to {config.sample_end_date.isoformat()}"
    )


def quote_date_in_sample_window(quote_date: date, config: RawDataConfig) -> bool:
    """Return whether a quote date is inside the configured inclusive sample window."""

    return config.sample_start_date <= quote_date <= config.sample_end_date


def require_quote_date_in_sample_window(
    quote_date: date,
    config: RawDataConfig,
    *,
    context: str,
) -> None:
    """Fail fast when one quote date violates the configured thesis sample window."""

    if quote_date_in_sample_window(quote_date, config):
        return
    message = (
        f"{context} quote_date {quote_date.isoformat()} is outside the configured "
        f"sample window {sample_window_label(config)}."
    )
    raise DataValidationError(message)


def sample_window_expr(
    config: RawDataConfig,
    *,
    column: str = "quote_date",
) -> pl.Expr:
    """Return a Polars expression for the configured inclusive sample window."""

    return (pl.col(column) >= pl.lit(config.sample_start_date)) & (
        pl.col(column) <= pl.lit(config.sample_end_date)
    )


def assert_frame_dates_in_sample_window(
    frame: pl.DataFrame,
    config: RawDataConfig,
    *,
    column: str = "quote_date",
    context: str,
) -> None:
    """Require all distinct dates in one frame to stay inside the configured sample window."""

    out_of_window = (
        frame.select(column)
        .unique()
        .filter(~sample_window_expr(config, column=column))
        .sort(column)
    )
    if out_of_window.is_empty():
        return
    offending_dates = ", ".join(
        value.isoformat() for value in out_of_window[column].to_list() if isinstance(value, date)
    )
    message = (
        f"{context} contains out-of-window {column} values: {offending_dates}. "
        f"Configured sample window: {sample_window_label(config)}."
    )
    raise DataValidationError(message)
