"""Explicit schemas for raw and derived data."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl
import pyarrow as pa

from ivsurf.exceptions import SchemaDriftError

RAW_COLUMNS: tuple[str, ...] = (
    "underlying_symbol",
    "quote_date",
    "root",
    "expiration",
    "strike",
    "option_type",
    "trade_volume",
    "bid_size_1545",
    "bid_1545",
    "ask_size_1545",
    "ask_1545",
    "underlying_bid_1545",
    "underlying_ask_1545",
    "implied_underlying_price_1545",
    "active_underlying_price_1545",
    "implied_volatility_1545",
    "delta_1545",
    "gamma_1545",
    "theta_1545",
    "vega_1545",
    "rho_1545",
    "open_interest",
)

RAW_POLARS_SCHEMA: dict[str, object] = {
    "underlying_symbol": pl.String,
    "quote_date": pl.String,
    "root": pl.String,
    "expiration": pl.String,
    "strike": pl.Float64,
    "option_type": pl.String,
    "trade_volume": pl.Int64,
    "bid_size_1545": pl.Int64,
    "bid_1545": pl.Float64,
    "ask_size_1545": pl.Int64,
    "ask_1545": pl.Float64,
    "underlying_bid_1545": pl.Float64,
    "underlying_ask_1545": pl.Float64,
    "implied_underlying_price_1545": pl.Float64,
    "active_underlying_price_1545": pl.Float64,
    "implied_volatility_1545": pl.Float64,
    "delta_1545": pl.Float64,
    "gamma_1545": pl.Float64,
    "theta_1545": pl.Float64,
    "vega_1545": pl.Float64,
    "rho_1545": pl.Float64,
    "open_interest": pl.Int64,
}

RAW_ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("quote_date", pa.date32(), nullable=False),
        pa.field("root", pa.string(), nullable=False),
        pa.field("expiration", pa.date32(), nullable=False),
        pa.field("strike", pa.float64(), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),
        pa.field("trade_volume", pa.int64(), nullable=False),
        pa.field("bid_size_1545", pa.int64(), nullable=False),
        pa.field("bid_1545", pa.float64(), nullable=False),
        pa.field("ask_size_1545", pa.int64(), nullable=False),
        pa.field("ask_1545", pa.float64(), nullable=False),
        pa.field("underlying_bid_1545", pa.float64(), nullable=False),
        pa.field("underlying_ask_1545", pa.float64(), nullable=False),
        pa.field("implied_underlying_price_1545", pa.float64(), nullable=False),
        pa.field("active_underlying_price_1545", pa.float64(), nullable=False),
        pa.field("implied_volatility_1545", pa.float64(), nullable=False),
        pa.field("delta_1545", pa.float64(), nullable=False),
        pa.field("gamma_1545", pa.float64(), nullable=False),
        pa.field("theta_1545", pa.float64(), nullable=False),
        pa.field("vega_1545", pa.float64(), nullable=False),
        pa.field("rho_1545", pa.float64(), nullable=False),
        pa.field("open_interest", pa.int64(), nullable=True),
        pa.field("source_zip", pa.string(), nullable=False),
    ]
)


def validate_raw_columns(columns: Iterable[str]) -> None:
    """Fail fast if a raw file drifts from the expected schema."""

    actual = tuple(columns)
    missing = [name for name in RAW_COLUMNS if name not in actual]
    unexpected = [name for name in actual if name not in RAW_COLUMNS]
    if missing or unexpected:
        message = (
            "Raw schema drift detected. "
            f"missing={missing if missing else '[]'} "
            f"unexpected={unexpected if unexpected else '[]'}"
        )
        raise SchemaDriftError(message)
