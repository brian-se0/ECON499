"""Book revaluation and hedge PnL evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from ivsurf.hedging.book import build_standard_book
from ivsurf.hedging.hedge_rules import compute_delta_vega_hedge
from ivsurf.hedging.revaluation import (
    SurfaceInterpolator,
    require_positive_spot,
    value_book,
    value_instrument,
)


@dataclass(frozen=True, slots=True)
class HedgingEvaluationRow:
    """Per-model, per-date hedging evaluation row."""

    model_name: str
    quote_date: date
    target_date: date
    spot_t: float
    spot_t1: float
    book_value_t0: float
    predicted_book_value_t1: float
    actual_book_value_t1: float
    revaluation_error: float
    abs_revaluation_error: float
    squared_revaluation_error: float
    underlying_quantity: float
    straddle_quantity: float
    predicted_net_delta: float
    predicted_net_vega: float
    unhedged_pnl: float
    hedged_pnl: float
    abs_hedged_pnl: float
    squared_hedged_pnl: float


def evaluate_model_hedging(
    model_name: str,
    quote_date: date,
    target_date: date,
    trade_spot: float,
    target_spot: float,
    actual_surface_t: SurfaceInterpolator,
    actual_surface_t1: SurfaceInterpolator,
    predicted_surface_t1: SurfaceInterpolator,
    rate: float,
    level_notional: float,
    skew_notional: float,
    calendar_notional: float,
    skew_moneyness_abs: float,
    short_maturity_days: int,
    long_maturity_days: int,
    hedge_maturity_days: int,
    hedge_straddle_moneyness: float,
) -> HedgingEvaluationRow:
    """Evaluate next-day revaluation and hedged PnL for one model/date."""

    validated_trade_spot = require_positive_spot(
        trade_spot,
        context="Hedging evaluation trade_spot",
        valuation_date=quote_date,
    )
    validated_target_spot = require_positive_spot(
        target_spot,
        context="Hedging evaluation target_spot",
        valuation_date=target_date,
    )
    book_instruments = build_standard_book(
        trade_date=quote_date,
        spot=validated_trade_spot,
        level_notional=level_notional,
        skew_notional=skew_notional,
        calendar_notional=calendar_notional,
        skew_moneyness_abs=skew_moneyness_abs,
        short_maturity_days=short_maturity_days,
        long_maturity_days=long_maturity_days,
    )
    book_t0 = value_book(
        instruments=book_instruments,
        valuation_date=quote_date,
        spot=validated_trade_spot,
        surface=actual_surface_t,
        rate=rate,
    )
    predicted_book_t1 = value_book(
        instruments=book_instruments,
        valuation_date=target_date,
        spot=validated_target_spot,
        surface=predicted_surface_t1,
        rate=rate,
    )
    actual_book_t1 = value_book(
        instruments=book_instruments,
        valuation_date=target_date,
        spot=validated_target_spot,
        surface=actual_surface_t1,
        rate=rate,
    )

    hedge = compute_delta_vega_hedge(
        book_instruments=book_instruments,
        trade_date=quote_date,
        trade_spot=validated_trade_spot,
        target_date=target_date,
        predicted_surface=predicted_surface_t1,
        rate=rate,
        hedge_maturity_days=hedge_maturity_days,
        hedge_straddle_moneyness=hedge_straddle_moneyness,
    )
    hedge_call_t0 = value_instrument(
        instrument=hedge.hedge_instrument_call,
        valuation_date=quote_date,
        spot=validated_trade_spot,
        surface=actual_surface_t,
        rate=rate,
    )
    hedge_put_t0 = value_instrument(
        instrument=hedge.hedge_instrument_put,
        valuation_date=quote_date,
        spot=validated_trade_spot,
        surface=actual_surface_t,
        rate=rate,
    )
    hedge_call_t1 = value_instrument(
        instrument=hedge.hedge_instrument_call,
        valuation_date=target_date,
        spot=validated_target_spot,
        surface=actual_surface_t1,
        rate=rate,
    )
    hedge_put_t1 = value_instrument(
        instrument=hedge.hedge_instrument_put,
        valuation_date=target_date,
        spot=validated_target_spot,
        surface=actual_surface_t1,
        rate=rate,
    )

    revaluation_error = predicted_book_t1.total_value - actual_book_t1.total_value
    unhedged_pnl = actual_book_t1.total_value - book_t0.total_value
    hedge_option_pnl = hedge.straddle_quantity * (
        (hedge_call_t1.price + hedge_put_t1.price) - (hedge_call_t0.price + hedge_put_t0.price)
    )
    hedge_underlying_pnl = hedge.underlying_quantity * (
        validated_target_spot - validated_trade_spot
    )
    hedged_pnl = unhedged_pnl + hedge_option_pnl + hedge_underlying_pnl

    return HedgingEvaluationRow(
        model_name=model_name,
        quote_date=quote_date,
        target_date=target_date,
        spot_t=validated_trade_spot,
        spot_t1=validated_target_spot,
        book_value_t0=book_t0.total_value,
        predicted_book_value_t1=predicted_book_t1.total_value,
        actual_book_value_t1=actual_book_t1.total_value,
        revaluation_error=revaluation_error,
        abs_revaluation_error=abs(revaluation_error),
        squared_revaluation_error=revaluation_error * revaluation_error,
        underlying_quantity=hedge.underlying_quantity,
        straddle_quantity=hedge.straddle_quantity,
        predicted_net_delta=hedge.predicted_net_delta,
        predicted_net_vega=hedge.predicted_net_vega,
        unhedged_pnl=unhedged_pnl,
        hedged_pnl=hedged_pnl,
        abs_hedged_pnl=abs(hedged_pnl),
        squared_hedged_pnl=hedged_pnl * hedged_pnl,
    )


def summarize_hedging_results(results: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-trade hedging results to model summaries."""

    return (
        results.group_by("model_name")
        .agg(
            pl.col("abs_revaluation_error").mean().alias("mean_abs_revaluation_error"),
            pl.col("squared_revaluation_error").mean().alias("mean_squared_revaluation_error"),
            pl.col("abs_hedged_pnl").mean().alias("mean_abs_hedged_pnl"),
            pl.col("squared_hedged_pnl").mean().alias("mean_squared_hedged_pnl"),
            pl.col("hedged_pnl").var(ddof=1).alias("hedged_pnl_variance"),
            pl.len().alias("n_trades"),
        )
        .sort("mean_abs_revaluation_error")
    )
