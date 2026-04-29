from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ivsurf.config import HedgingConfig
from ivsurf.hedging.book import BookInstrument, build_standard_book
from ivsurf.hedging.hedge_rules import InfeasibleHedgeError, compute_delta_vega_hedge
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import (
    SurfaceDomainError,
    SurfaceInterpolator,
    surface_interpolator_from_frame,
    value_instrument,
)
from ivsurf.hedging.validation import (
    require_hedging_config_in_surface_domain,
    require_hedging_summary_matches_results,
)
from ivsurf.surfaces.grid import SurfaceGrid

GRID = SurfaceGrid(maturity_days=(30, 90), moneyness_points=(-0.1, 0.0))


def _flat_surface_interpolator(sigma: float) -> SurfaceInterpolator:
    maturity_days = np.asarray([1.0, 30.0, 90.0], dtype=np.float64)
    moneyness = np.asarray([-0.3, -0.1, 0.0, 0.1, 0.3], dtype=np.float64)
    total_variance = np.asarray(
        [[sigma * sigma * (days / 365.0) for _money in moneyness] for days in maturity_days],
        dtype=np.float64,
    )
    return SurfaceInterpolator(
        maturity_days=maturity_days,
        moneyness_points=moneyness,
        total_variance_grid=total_variance,
    )


def _surface_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            rows.append(
                {
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "completed_total_variance": 0.01 + (0.001 * maturity_index),
                }
            )
    return pl.DataFrame(rows)


def test_surface_interpolator_from_frame_rejects_incomplete_grid() -> None:
    frame = _surface_frame().filter(
        ~((pl.col("maturity_index") == 1) & (pl.col("moneyness_index") == 1))
    )

    with pytest.raises(ValueError, match="coordinates do not match"):
        surface_interpolator_from_frame(frame, "completed_total_variance", GRID)


def test_surface_interpolator_from_frame_rejects_shrunken_inferred_grid() -> None:
    frame = _surface_frame().filter(pl.col("maturity_index") == 0)

    with pytest.raises(ValueError, match="coordinates do not match"):
        surface_interpolator_from_frame(frame, "completed_total_variance", GRID)


def test_surface_interpolator_from_frame_rejects_configured_coordinate_mismatch() -> None:
    frame = _surface_frame().with_columns(
        pl.when(pl.col("maturity_index") == 1)
        .then(91)
        .otherwise(pl.col("maturity_days"))
        .alias("maturity_days")
    )

    with pytest.raises(ValueError, match="coordinates do not match"):
        surface_interpolator_from_frame(frame, "completed_total_variance", GRID)


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


def test_surface_interpolator_rejects_out_of_domain_queries() -> None:
    surface = _flat_surface_interpolator(0.2)

    with pytest.raises(SurfaceDomainError, match=r"remaining_days=0\.0.*unit_book"):
        surface.sigma(
            0,
            0.0,
            instrument_label="unit_book",
            valuation_date=date(2021, 1, 5),
            model_name="unit_model",
        )
    with pytest.raises(SurfaceDomainError, match=r"log_moneyness=0\.5"):
        surface.sigma(
            30,
            0.5,
            instrument_label="unit_book",
            valuation_date=date(2021, 1, 5),
            model_name="unit_model",
        )


def test_value_instrument_rejects_expired_maturity_before_surface_lookup() -> None:
    trade_date = date(2021, 1, 4)
    target_date = trade_date + timedelta(days=30)
    instrument = BookInstrument(
        label="unit_expired",
        option_type="C",
        quantity=1.0,
        strike=100.0,
        initial_maturity_days=30,
        trade_date=trade_date,
    )

    with pytest.raises(SurfaceDomainError, match=r"remaining_days=0\.0.*unit_expired"):
        value_instrument(
            instrument=instrument,
            valuation_date=target_date,
            spot=100.0,
            surface=_flat_surface_interpolator(0.2),
            rate=0.0,
            model_name="unit_model",
        )


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


def test_evaluate_model_hedging_rejects_expired_book_maturity_before_results() -> None:
    quote_date = date(2021, 1, 4)
    target_date = quote_date + timedelta(days=30)
    surface_t = _flat_surface_interpolator(0.2)
    surface_t1 = _flat_surface_interpolator(0.22)

    with pytest.raises(SurfaceDomainError, match=r"remaining_days=0\.0.*atm_call_long"):
        evaluate_model_hedging(
            model_name="expired_book",
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
            hedge_maturity_days=60,
            hedge_straddle_moneyness=0.0,
        )


def test_evaluate_model_hedging_rejects_nonpositive_target_spot_before_valuation_math() -> None:
    quote_date = date(2021, 1, 4)
    target_date = date(2021, 1, 5)
    surface_t = _flat_surface_interpolator(0.2)
    surface_t1 = _flat_surface_interpolator(0.22)

    with pytest.raises(ValueError, match=r"target_spot.*2021-01-05.*0\.0"):
        evaluate_model_hedging(
            model_name="invalid_spot",
            quote_date=quote_date,
            target_date=target_date,
            trade_spot=100.0,
            target_spot=0.0,
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


def test_zero_vega_surface_fails_instead_of_delta_only_fallback() -> None:
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

    with pytest.raises(InfeasibleHedgeError, match=r"hedge straddle vega.*unit_model"):
        compute_delta_vega_hedge(
            book_instruments=book,
            trade_date=quote_date,
            trade_spot=100.0,
            target_date=target_date,
            predicted_surface=zero_surface,
            rate=0.0,
            hedge_maturity_days=30,
            hedge_straddle_moneyness=0.0,
            model_name="unit_model",
        )


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


def test_zero_vega_hedge_instrument_fails_with_residual_book_vega() -> None:
    quote_date = date(2021, 1, 4)
    target_date = date(2021, 1, 5)
    predicted_surface = SurfaceInterpolator(
        maturity_days=np.asarray([1.0, 30.0, 90.0], dtype=np.float64),
        moneyness_points=np.asarray([-0.3, -0.1, 0.0, 0.1, 0.3], dtype=np.float64),
        total_variance_grid=np.asarray(
            [
                [0.012, 0.012, 0.0, 0.012, 0.012],
                [0.012, 0.012, 0.0, 0.012, 0.012],
                [0.030, 0.030, 0.024, 0.030, 0.030],
            ],
            dtype=np.float64,
        ),
    )
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

    with pytest.raises(InfeasibleHedgeError, match="hedge_maturity_days=30"):
        compute_delta_vega_hedge(
            book_instruments=book,
            trade_date=quote_date,
            trade_spot=100.0,
            target_date=target_date,
            predicted_surface=predicted_surface,
            rate=0.0,
            hedge_maturity_days=30,
            hedge_straddle_moneyness=0.0,
        )


def test_near_zero_hedge_vega_fails_against_configured_floor() -> None:
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

    with pytest.raises(InfeasibleHedgeError, match=r"hedge_vega_floor=1000000000\.0"):
        compute_delta_vega_hedge(
            book_instruments=book,
            trade_date=quote_date,
            trade_spot=100.0,
            target_date=target_date,
            predicted_surface=surface,
            rate=0.0,
            hedge_maturity_days=30,
            hedge_straddle_moneyness=0.0,
            hedge_vega_floor=1.0e9,
        )


def test_stage08_hedging_config_prevalidation_rejects_out_of_domain_config() -> None:
    grid = SurfaceGrid(maturity_days=(1, 30, 90), moneyness_points=(-0.1, 0.0, 0.1))
    config = HedgingConfig.model_validate(
        {
            "risk_free_rate": 0.0,
            "level_notional": 1.0,
            "skew_notional": 1.0,
            "calendar_notional": 0.5,
            "skew_moneyness_abs": 0.2,
            "short_maturity_days": 30,
            "long_maturity_days": 90,
            "hedge_maturity_days": 30,
            "hedge_straddle_moneyness": 0.0,
            "hedge_vega_floor": 1.0e-12,
        }
    )

    with pytest.raises(ValueError, match="skew_moneyness_abs"):
        require_hedging_config_in_surface_domain(
            config,
            grid,
            max_target_gap_days=1,
        )


def test_stage08_hedging_config_prevalidation_rejects_target_gap_below_grid() -> None:
    grid = SurfaceGrid(maturity_days=(30, 90), moneyness_points=(-0.3, 0.0, 0.3))
    config = HedgingConfig.model_validate(
        {
            "risk_free_rate": 0.0,
            "level_notional": 1.0,
            "skew_notional": 1.0,
            "calendar_notional": 0.5,
            "skew_moneyness_abs": 0.1,
            "short_maturity_days": 30,
            "long_maturity_days": 90,
            "hedge_maturity_days": 30,
            "hedge_straddle_moneyness": 0.0,
            "hedge_vega_floor": 1.0e-12,
        }
    )

    with pytest.raises(ValueError, match="short_maturity_days_after_max_target_gap"):
        require_hedging_config_in_surface_domain(
            config,
            grid,
            max_target_gap_days=3,
        )


def test_stage08_hedging_config_prevalidation_rejects_exhausted_short_maturity() -> None:
    grid = SurfaceGrid(maturity_days=(1, 30, 60, 90), moneyness_points=(-0.3, 0.0, 0.3))
    config = HedgingConfig.model_validate(
        {
            "risk_free_rate": 0.0,
            "level_notional": 1.0,
            "skew_notional": 1.0,
            "calendar_notional": 0.5,
            "skew_moneyness_abs": 0.1,
            "short_maturity_days": 30,
            "long_maturity_days": 90,
            "hedge_maturity_days": 60,
            "hedge_straddle_moneyness": 0.0,
            "hedge_vega_floor": 1.0e-12,
        }
    )

    with pytest.raises(ValueError, match="short_maturity_days_after_max_target_gap"):
        require_hedging_config_in_surface_domain(
            config,
            grid,
            max_target_gap_days=30,
        )


def test_stage08_hedging_config_prevalidation_rejects_exhausted_hedge_maturity() -> None:
    grid = SurfaceGrid(maturity_days=(1, 30, 60, 90), moneyness_points=(-0.3, 0.0, 0.3))
    config = HedgingConfig.model_validate(
        {
            "risk_free_rate": 0.0,
            "level_notional": 1.0,
            "skew_notional": 1.0,
            "calendar_notional": 0.5,
            "skew_moneyness_abs": 0.1,
            "short_maturity_days": 60,
            "long_maturity_days": 90,
            "hedge_maturity_days": 30,
            "hedge_straddle_moneyness": 0.0,
            "hedge_vega_floor": 1.0e-12,
        }
    )

    with pytest.raises(ValueError, match="hedge_maturity_days_after_max_target_gap"):
        require_hedging_config_in_surface_domain(
            config,
            grid,
            max_target_gap_days=30,
        )


def test_hedging_summary_validation_rejects_stale_metric_values() -> None:
    results = pl.DataFrame(
        [
            {
                "model_name": "ridge",
                "quote_date": date(2021, 1, 4),
                "target_date": date(2021, 1, 5),
                "abs_revaluation_error": 0.1,
                "squared_revaluation_error": 0.01,
                "abs_hedged_pnl": 0.2,
                "squared_hedged_pnl": 0.04,
                "hedged_pnl": 0.2,
            },
            {
                "model_name": "ridge",
                "quote_date": date(2021, 1, 5),
                "target_date": date(2021, 1, 6),
                "abs_revaluation_error": 0.3,
                "squared_revaluation_error": 0.09,
                "abs_hedged_pnl": 0.4,
                "squared_hedged_pnl": 0.16,
                "hedged_pnl": -0.4,
            },
        ]
    )
    stale_summary = summarize_hedging_results(results).with_columns(
        (pl.col("mean_abs_revaluation_error") + 0.01).alias("mean_abs_revaluation_error")
    )

    with pytest.raises(ValueError, match="recomputed hedging result aggregate"):
        require_hedging_summary_matches_results(stale_summary, results)
