"""Derived-field construction for decision-snapshot SPX option rows."""

from __future__ import annotations

from datetime import date

import polars as pl

from ivcast.calendar import MarketCalendar
from ivcast.config import MarketCalendarConfig

DECISION_TIMESTAMP_COLUMN = "effective_decision_timestamp"
DAILY_SPOT_COLUMN = "daily_spot_1545"


def _market_calendar(config: MarketCalendarConfig) -> MarketCalendar:
    return MarketCalendar(
        calendar_name=config.calendar_name,
        timezone=config.timezone,
        decision_time=config.decision_time,
        decision_snapshot_minutes_before_close=config.decision_snapshot_minutes_before_close,
        am_settled_roots=config.am_settled_roots,
        am_settlement_time=config.am_settlement_time,
        vendor_am_iv_roots=config.vendor_am_iv_roots,
        vendor_am_iv_previous_session_before=config.vendor_am_iv_previous_session_before,
    )


def _single_quote_date(frame: pl.DataFrame) -> date:
    quote_dates = frame.select(pl.col("quote_date").unique()).to_series().to_list()
    if len(quote_dates) != 1:
        message = "Derived-field construction expects a single quote_date per frame."
        raise ValueError(message)
    quote_date = quote_dates[0]
    if not isinstance(quote_date, date):
        message = "quote_date must be parsed as a Polars Date before derived-field construction."
        raise TypeError(message)
    return quote_date


def build_tau_lookup(
    frame: pl.DataFrame,
    calendar_config: MarketCalendarConfig,
) -> pl.DataFrame:
    """Build a per-root and per-expiration maturity lookup for one quote date.

    `tau_years` runs from the effective decision snapshot to the economic settlement
    instant and is the maturity coordinate. `tau_vendor_years` runs to the maturity
    instant embedded in the vendor implied volatility and is used only to convert the
    vendor implied volatility into price-consistent total variance.
    """

    quote_date = _single_quote_date(frame)
    market_calendar = _market_calendar(calendar_config)

    keys = frame.select("root", "expiration").unique().sort(["root", "expiration"])
    rows: list[dict[str, object]] = []
    for row in keys.iter_rows(named=True):
        expiration = row["expiration"]
        root = row["root"]
        if not isinstance(expiration, date) or not isinstance(root, str):
            message = "root/expiration lookup contains invalid types."
            raise TypeError(message)
        tau_years = market_calendar.compute_tau_years(
            quote_date=quote_date,
            expiration=expiration,
            root=root,
        )
        tau_vendor_years = market_calendar.compute_vendor_iv_tau_years(
            quote_date=quote_date,
            expiration=expiration,
            root=root,
        )
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "tau_years": tau_years,
                "tau_vendor_years": tau_vendor_years,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "root": pl.String,
            "expiration": pl.Date,
            "tau_years": pl.Float64,
            "tau_vendor_years": pl.Float64,
        },
    )


def daily_spot_expr() -> pl.Expr:
    """Return one 15:45 spot per quote date.

    The vendor `active_underlying_price_1545` is a single daily value on all but five
    sessions (2020-08-10 to 2020-08-14), where it is reported per expiration as a
    carry-adjusted price. The nearest expiration carries the spot on those days, so the
    daily spot is the median positive vendor price of the nearest expiration.
    """

    positive = pl.col("active_underlying_price_1545") > 0.0
    nearest_expiration = pl.col("expiration").filter(positive).min()
    return (
        pl.col("active_underlying_price_1545")
        .filter(positive & (pl.col("expiration") == nearest_expiration))
        .median()
    )


def add_effective_decision_timestamp(
    frame: pl.DataFrame,
    calendar_config: MarketCalendarConfig,
) -> pl.DataFrame:
    """Persist the effective vendor decision timestamp for one daily option artifact."""

    quote_date = _single_quote_date(frame)
    market_calendar = _market_calendar(calendar_config)
    decision_timestamp = market_calendar.effective_decision_datetime(quote_date).isoformat()
    return frame.with_columns(pl.lit(decision_timestamp).alias(DECISION_TIMESTAMP_COLUMN))


def add_derived_fields(frame: pl.DataFrame, tau_lookup: pl.DataFrame) -> pl.DataFrame:
    """Join maturity and compute option-level derived fields.

    Total variance is the vendor implied volatility squared times the vendor IV
    maturity, which reproduces the quoted option price regardless of the
    settlement-time convention. Log moneyness is measured against one daily spot.
    """

    _single_quote_date(frame)
    enriched = frame.join(tau_lookup, on=["root", "expiration"], how="left", validate="m:1")
    enriched = enriched.with_columns(daily_spot_expr().alias(DAILY_SPOT_COLUMN))
    return enriched.with_columns(
        ((pl.col("bid_1545") + pl.col("ask_1545")) / 2.0).alias("mid_1545"),
        (pl.col("ask_1545") - pl.col("bid_1545")).alias("spread_1545"),
        (pl.col("strike").log() - pl.col(DAILY_SPOT_COLUMN).log()).alias("log_moneyness"),
        (
            (pl.col("implied_volatility_1545") ** 2.0) * pl.col("tau_vendor_years")
        ).alias("total_variance"),
    )
