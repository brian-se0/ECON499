from __future__ import annotations

from datetime import date

import numpy as np

from ivsurf.hedging.book import build_standard_book
from ivsurf.hedging.hedge_rules import compute_delta_vega_hedge
from ivsurf.hedging.pnl import evaluate_model_hedging
from ivsurf.hedging.revaluation import SurfaceInterpolator


def _flat_surface_interpolator(sigma: float) -> SurfaceInterpolator:
    maturity_days = np.asarray([30.0, 90.0], dtype=np.float64)
    moneyness = np.asarray([-0.1, 0.0, 0.1], dtype=np.float64)
    total_variance = np.asarray(
        [[sigma * sigma * (days / 365.0) for _money in moneyness] for days in maturity_days],
        dtype=np.float64,
    )
    return SurfaceInterpolator(
        maturity_days=maturity_days,
        moneyness_points=moneyness,
        total_variance_grid=total_variance,
    )


def test_delta_vega_hedge_neutralizes_predicted_exposures() -> None:
    quote_date = date(2021, 1, 4)
    target_date = date(2021, 1, 5)
    surface = _flat_surface_interpolator(0.2)
    book = build_standard_book(
        trade_date=quote_date,
        spot=100.0,
        level_notional=1.0,
        skew_notional=1.0,
        calendar_notional=0.5,
        skew_moneyness_abs=0.1,
        short_maturity_days=30,
        long_maturity_days=90,
    )
    hedge = compute_delta_vega_hedge(
        book_instruments=book,
        trade_date=quote_date,
        trade_spot=100.0,
        target_date=target_date,
        predicted_surface=surface,
        rate=0.0,
        hedge_maturity_days=30,
        hedge_straddle_moneyness=0.0,
    )
    assert abs(hedge.predicted_net_delta) < 1.0e-8
    assert abs(hedge.predicted_net_vega) < 1.0e-8


def test_revaluation_error_is_zero_when_forecast_matches_actual_surface() -> None:
    quote_date = date(2021, 1, 4)
    target_date = date(2021, 1, 5)
    surface_t = _flat_surface_interpolator(0.2)
    surface_t1 = _flat_surface_interpolator(0.22)
    result = evaluate_model_hedging(
        model_name="perfect",
        quote_date=quote_date,
        target_date=target_date,
        trade_spot=100.0,
        target_spot=101.0,
        actual_surface_t=surface_t,
        actual_surface_t1=surface_t1,
        predicted_surface_t1=surface_t1,
        rate=0.0,
        level_notional=1.0,
        skew_notional=1.0,
        calendar_notional=0.5,
        skew_moneyness_abs=0.1,
        short_maturity_days=30,
        long_maturity_days=90,
        hedge_maturity_days=30,
        hedge_straddle_moneyness=0.0,
    )
    assert abs(result.revaluation_error) < 1.0e-10


def test_zero_vega_surface_uses_delta_only_hedge_when_book_vega_is_already_zero() -> None:
    quote_date = date(2021, 1, 4)
    target_date = date(2021, 1, 5)
    zero_surface = _flat_surface_interpolator(0.0)
    book = build_standard_book(
        trade_date=quote_date,
        spot=100.0,
        level_notional=1.0,
        skew_notional=1.0,
        calendar_notional=0.5,
        skew_moneyness_abs=0.1,
        short_maturity_days=30,
        long_maturity_days=90,
    )

    hedge = compute_delta_vega_hedge(
        book_instruments=book,
        trade_date=quote_date,
        trade_spot=100.0,
        target_date=target_date,
        predicted_surface=zero_surface,
        rate=0.0,
        hedge_maturity_days=30,
        hedge_straddle_moneyness=0.0,
    )

    assert hedge.straddle_quantity == 0.0
    assert abs(hedge.predicted_net_delta) < 1.0e-8
    assert hedge.predicted_net_vega == 0.0


def test_standard_book_labels_match_position_signs() -> None:
    book = build_standard_book(
        trade_date=date(2021, 1, 4),
        spot=100.0,
        level_notional=1.0,
        skew_notional=1.0,
        calendar_notional=0.5,
        skew_moneyness_abs=0.1,
        short_maturity_days=30,
        long_maturity_days=90,
    )

    for instrument in book:
        if instrument.label.endswith("_short"):
            assert instrument.quantity < 0.0
        if instrument.label.endswith("_long"):
            assert instrument.quantity > 0.0
