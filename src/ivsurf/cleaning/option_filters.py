"""Explicit option-level cleaning rules and reason codes."""

from __future__ import annotations

import polars as pl

from ivsurf.config import CleaningConfig


def apply_option_quality_flags(frame: pl.DataFrame, config: CleaningConfig) -> pl.DataFrame:
    """Flag invalid rows with explicit reason codes instead of silently dropping them."""

    invalid_reason = (
        pl.when(~pl.col("option_type").is_in(config.allowed_option_types))
        .then(pl.lit("INVALID_OPTION_TYPE"))
        .when(pl.col("bid_1545") <= config.min_valid_bid_exclusive)
        .then(pl.lit("NON_POSITIVE_BID"))
        .when(pl.col("ask_1545") <= config.min_valid_ask_exclusive)
        .then(pl.lit("NON_POSITIVE_ASK"))
        .when(config.require_ask_ge_bid & (pl.col("ask_1545") < pl.col("bid_1545")))
        .then(pl.lit("ASK_LT_BID"))
        .when(config.require_positive_iv & (pl.col("implied_volatility_1545") <= 0.0))
        .then(pl.lit("NON_POSITIVE_IV"))
        .when(config.require_positive_vega & (pl.col("vega_1545") <= 0.0))
        .then(pl.lit("NON_POSITIVE_VEGA"))
        .when(
            config.require_positive_underlying_price
            & (pl.col("active_underlying_price_1545") <= 0.0)
        )
        .then(pl.lit("NON_POSITIVE_UNDERLYING_PRICE"))
        .when(pl.col("mid_1545") <= config.min_valid_mid_price_exclusive)
        .then(pl.lit("NON_POSITIVE_MID"))
        .when(pl.col("tau_years") < config.min_tau_years)
        .then(pl.lit("TAU_TOO_SHORT"))
        .when(pl.col("tau_years") > config.max_tau_years)
        .then(pl.lit("TAU_TOO_LONG"))
        .when(pl.col("log_moneyness").abs() > config.max_abs_log_moneyness)
        .then(pl.lit("OUTSIDE_MONEYNESS_RANGE"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )

    return (
        frame.with_columns(invalid_reason.alias("invalid_reason"))
        .with_columns(pl.col("invalid_reason").is_null().alias("is_valid_observation"))
    )


def valid_option_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Return only valid rows."""

    return frame.filter(pl.col("is_valid_observation"))
