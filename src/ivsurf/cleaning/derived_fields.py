"""Derived-field construction for decision-snapshot SPX option rows."""

from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.calendar import MarketCalendar
from ivsurf.config import MarketCalendarConfig

DECISION_TIMESTAMP_COLUMN = "effective_decision_timestamp"


def _market_calendar(config: MarketCalendarConfig) -> MarketCalendar:
    return MarketCalendar(
        calendar_name=config.calendar_name,
        timezone=config.timezone,
        decision_time=config.decision_time,
        decision_snapshot_minutes_before_close=config.decision_snapshot_minutes_before_close,
        am_settled_roots=config.am_settled_roots,
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
    """Build a per-root and per-expiration tau lookup for one quote date."""

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
        rows.append({"root": root, "expiration": expiration, "tau_years": tau_years})
    return pl.DataFrame(
        rows,
        schema={"root": pl.String, "expiration": pl.Date, "tau_years": pl.Float64},
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
    """Join maturity and compute option-level derived fields."""

    enriched = frame.join(tau_lookup, on=["root", "expiration"], how="left", validate="m:1")
    return enriched.with_columns(
        ((pl.col("bid_1545") + pl.col("ask_1545")) / 2.0).alias("mid_1545"),
        (pl.col("ask_1545") - pl.col("bid_1545")).alias("spread_1545"),
        (
            pl.col("strike").log() - pl.col("active_underlying_price_1545").log()
        ).alias("log_moneyness"),
        ((pl.col("implied_volatility_1545") ** 2.0) * pl.col("tau_years")).alias("total_variance"),
    )
