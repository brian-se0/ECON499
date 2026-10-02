"""Explicit option-level cleaning rules and reason codes."""

from __future__ import annotations

import polars as pl

from ivcast.config import CleaningConfig


def _placeholder_bid_ask_expr(config: CleaningConfig) -> pl.Expr:
    expression = pl.lit(False)
    for bid_value, ask_value in config.placeholder_bid_ask_pairs:
        expression = expression | (
            (pl.col("bid_1545") == bid_value) & (pl.col("ask_1545") == ask_value)
        )
    return expression


def _placeholder_iv_expr(config: CleaningConfig) -> pl.Expr:
    if not config.placeholder_implied_volatilities:
        return pl.lit(False)
    return pl.col("implied_volatility_1545").is_in(list(config.placeholder_implied_volatilities))


def apply_option_quality_flags(frame: pl.DataFrame, config: CleaningConfig) -> pl.DataFrame:
    """Flag invalid rows with explicit reason codes instead of silently dropping them.

    Vendor placeholder values are not market observations: a configured bid/ask pair
    (for example 998/999) marks a missing market, and configured exact implied
    volatilities (for example 0.02 and 0.001) are solver sentinels written when the
    vendor could not solve an implied volatility.
    """

    invalid_reason = (
        pl.when(pl.col("option_type").is_null())
        .then(pl.lit("MISSING_OPTION_TYPE"))
        .when(pl.col("bid_1545").is_null())
        .then(pl.lit("MISSING_BID_1545"))
        .when(pl.col("ask_1545").is_null())
        .then(pl.lit("MISSING_ASK_1545"))
        .when(pl.col("implied_volatility_1545").is_null())
        .then(pl.lit("MISSING_IMPLIED_VOLATILITY_1545"))
        .when(pl.col("vega_1545").is_null())
        .then(pl.lit("MISSING_VEGA_1545"))
        .when(pl.col("active_underlying_price_1545").is_null())
        .then(pl.lit("MISSING_ACTIVE_UNDERLYING_PRICE_1545"))
        .when(pl.col("strike").is_null())
        .then(pl.lit("MISSING_STRIKE"))
        .when(pl.col("mid_1545").is_null())
        .then(pl.lit("MISSING_MID_1545"))
        .when(pl.col("tau_years").is_null())
        .then(pl.lit("MISSING_TAU_YEARS"))
        .when(pl.col("total_variance").is_null())
        .then(pl.lit("MISSING_TOTAL_VARIANCE"))
        .when(~pl.col("bid_1545").is_finite())
        .then(pl.lit("NONFINITE_BID_1545"))
        .when(~pl.col("ask_1545").is_finite())
        .then(pl.lit("NONFINITE_ASK_1545"))
        .when(~pl.col("implied_volatility_1545").is_finite())
        .then(pl.lit("NONFINITE_IV_1545"))
        .when(~pl.col("vega_1545").is_finite())
        .then(pl.lit("NONFINITE_VEGA_1545"))
        .when(~pl.col("active_underlying_price_1545").is_finite())
        .then(pl.lit("NONFINITE_UNDERLYING_1545"))
        .when(~pl.col("strike").is_finite())
        .then(pl.lit("NONFINITE_STRIKE"))
        .when(pl.col("strike") <= 0.0)
        .then(pl.lit("NON_POSITIVE_STRIKE"))
        .when(~pl.col("mid_1545").is_finite())
        .then(pl.lit("NONFINITE_MID_1545"))
        .when(~pl.col("tau_years").is_finite())
        .then(pl.lit("NONFINITE_TAU_YEARS"))
        .when(~pl.col("total_variance").is_finite())
        .then(pl.lit("NONFINITE_TOTAL_VARIANCE"))
        .when(~pl.col("option_type").is_in(config.allowed_option_types))
        .then(pl.lit("INVALID_OPTION_TYPE"))
        .when(_placeholder_bid_ask_expr(config))
        .then(pl.lit("PLACEHOLDER_BID_ASK_QUOTE"))
        .when(_placeholder_iv_expr(config))
        .then(pl.lit("PLACEHOLDER_IMPLIED_VOLATILITY"))
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
        .when(pl.col("total_variance") <= 0.0)
        .then(pl.lit("NON_POSITIVE_TOTAL_VARIANCE"))
        .when(pl.col("log_moneyness").is_null())
        .then(pl.lit("MISSING_LOG_MONEYNESS"))
        .when(~pl.col("log_moneyness").is_finite())
        .then(pl.lit("NONFINITE_LOG_MONEYNESS"))
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
