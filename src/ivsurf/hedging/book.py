"""Standardized synthetic book definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class BookInstrument:
    """One option position in the standardized book."""

    label: str
    option_type: str
    quantity: float
    strike: float
    initial_maturity_days: int
    trade_date: date


def build_standard_book(
    trade_date: date,
    spot: float,
    level_notional: float,
    skew_notional: float,
    calendar_notional: float,
    skew_moneyness_abs: float,
    short_maturity_days: int,
    long_maturity_days: int,
) -> list[BookInstrument]:
    """Build the standardized ATM, skew, and calendar exposure book."""

    atm_strike = spot
    put_strike = spot * math.exp(-skew_moneyness_abs)
    call_strike = spot * math.exp(skew_moneyness_abs)
    return [
        BookInstrument(
            "atm_call_short",
            "C",
            level_notional,
            atm_strike,
            short_maturity_days,
            trade_date,
        ),
        BookInstrument(
            "atm_put_short",
            "P",
            level_notional,
            atm_strike,
            short_maturity_days,
            trade_date,
        ),
        BookInstrument(
            "rr_put_short",
            "P",
            skew_notional,
            put_strike,
            short_maturity_days,
            trade_date,
        ),
        BookInstrument(
            "rr_call_short",
            "C",
            -skew_notional,
            call_strike,
            short_maturity_days,
            trade_date,
        ),
        BookInstrument(
            "cal_call_long",
            "C",
            calendar_notional,
            atm_strike,
            long_maturity_days,
            trade_date,
        ),
        BookInstrument(
            "cal_call_short",
            "C",
            -calendar_notional,
            atm_strike,
            short_maturity_days,
            trade_date,
        ),
        BookInstrument(
            "cal_put_long",
            "P",
            calendar_notional,
            atm_strike,
            long_maturity_days,
            trade_date,
        ),
        BookInstrument(
            "cal_put_short",
            "P",
            -calendar_notional,
            atm_strike,
            short_maturity_days,
            trade_date,
        ),
    ]

