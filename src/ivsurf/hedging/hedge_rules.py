"""Delta-vega hedge sizing rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from ivsurf.hedging.book import BookInstrument
from ivsurf.hedging.revaluation import (
    SurfaceInterpolator,
    require_positive_spot,
    value_book,
    value_instrument,
)


@dataclass(frozen=True, slots=True)
class HedgePortfolio:
    """Underlying plus one straddle hedge."""

    underlying_quantity: float
    straddle_quantity: float
    hedge_instrument_call: BookInstrument
    hedge_instrument_put: BookInstrument
    predicted_net_delta: float
    predicted_net_vega: float


def compute_delta_vega_hedge(
    book_instruments: list[BookInstrument],
    trade_date: date,
    trade_spot: float,
    target_date: date,
    predicted_surface: SurfaceInterpolator,
    rate: float,
    hedge_maturity_days: int,
    hedge_straddle_moneyness: float,
) -> HedgePortfolio:
    """Size hedges under the explicit no-change spot assumption for next-day sizing."""

    validated_trade_spot = require_positive_spot(
        trade_spot,
        context="Delta-vega hedge sizing trade_spot",
        valuation_date=trade_date,
    )
    hedge_strike = validated_trade_spot * math.exp(hedge_straddle_moneyness)
    hedge_call = BookInstrument(
        label="hedge_call",
        option_type="C",
        quantity=1.0,
        strike=hedge_strike,
        initial_maturity_days=hedge_maturity_days,
        trade_date=trade_date,
    )
    hedge_put = BookInstrument(
        label="hedge_put",
        option_type="P",
        quantity=1.0,
        strike=hedge_strike,
        initial_maturity_days=hedge_maturity_days,
        trade_date=trade_date,
    )

    predicted_book = value_book(
        instruments=book_instruments,
        valuation_date=target_date,
        spot=validated_trade_spot,
        surface=predicted_surface,
        rate=rate,
    )
    predicted_hedge_call = value_instrument(
        instrument=hedge_call,
        valuation_date=target_date,
        spot=validated_trade_spot,
        surface=predicted_surface,
        rate=rate,
    )
    predicted_hedge_put = value_instrument(
        instrument=hedge_put,
        valuation_date=target_date,
        spot=validated_trade_spot,
        surface=predicted_surface,
        rate=rate,
    )
    hedge_delta = predicted_hedge_call.delta + predicted_hedge_put.delta
    hedge_vega = predicted_hedge_call.vega + predicted_hedge_put.vega
    if hedge_vega == 0.0:
        underlying_quantity = -predicted_book.total_delta
        return HedgePortfolio(
            underlying_quantity=float(underlying_quantity),
            straddle_quantity=0.0,
            hedge_instrument_call=hedge_call,
            hedge_instrument_put=hedge_put,
            predicted_net_delta=0.0,
            predicted_net_vega=float(predicted_book.total_vega),
        )

    straddle_quantity = -predicted_book.total_vega / hedge_vega
    underlying_quantity = -(predicted_book.total_delta + (straddle_quantity * hedge_delta))
    predicted_net_delta = (
        predicted_book.total_delta
        + underlying_quantity
        + (straddle_quantity * hedge_delta)
    )
    predicted_net_vega = predicted_book.total_vega + (straddle_quantity * hedge_vega)
    return HedgePortfolio(
        underlying_quantity=float(underlying_quantity),
        straddle_quantity=float(straddle_quantity),
        hedge_instrument_call=hedge_call,
        hedge_instrument_put=hedge_put,
        predicted_net_delta=float(predicted_net_delta),
        predicted_net_vega=float(predicted_net_vega),
    )
