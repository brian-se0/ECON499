from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from ivsurf.evaluation.alignment import (
    build_forecast_realization_panel,
    load_actual_surface_frame,
    load_daily_spot_frame,
    load_forecast_frame,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import surface_interpolator_from_frame


def _write_day(
    path: str,
    quote_date: date,
    sigma: float,
) -> None:
    rows: list[dict[str, object]] = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            total_variance = sigma * (maturity_days / 365.0)
            iv = sigma**0.5
            rows.append(
                {
                    "quote_date": quote_date,
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "observed_total_variance": total_variance,
                    "observed_iv": iv,
                    "completed_total_variance": total_variance,
                    "completed_iv": iv,
                    "observed_mask": True,
                    "vega_sum": 1.0,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_stats_and_hedging_slice_smoke(tmp_path: Path) -> None:
    root = tmp_path
    gold_year = root / "gold" / "year=2021"
    gold_year.mkdir(parents=True)
    forecast_dir = root / "gold" / "forecasts"
    forecast_dir.mkdir(parents=True)
    silver_year = root / "silver" / "year=2021"
    silver_year.mkdir(parents=True)

    _write_day(str(gold_year / "2021-01-04.parquet"), date(2021, 1, 4), 0.04)
    _write_day(str(gold_year / "2021-01-05.parquet"), date(2021, 1, 5), 0.05)

    forecast_rows = []
    for model_name, sigma in (("good", 0.05), ("bad", 0.07)):
        for maturity_index, maturity_days in enumerate((30, 90)):
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                forecast_rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": date(2021, 1, 4),
                        "target_date": date(2021, 1, 5),
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "predicted_total_variance": sigma * (maturity_days / 365.0),
                    }
                )
    pl.DataFrame(forecast_rows).write_parquet(forecast_dir / "models.parquet")

    silver_rows = []
    for quote_date, spot in ((date(2021, 1, 4), 100.0), (date(2021, 1, 5), 101.0)):
        silver_rows.append(
            {
                "quote_date": quote_date,
                "active_underlying_price_1545": spot,
            }
        )
    pl.DataFrame(silver_rows).write_parquet(silver_year / "spots.parquet")

    actual = load_actual_surface_frame(root / "gold")
    forecasts = load_forecast_frame(forecast_dir)
    spots = load_daily_spot_frame(root / "silver")
    assert spots.height == 2

    panel = build_forecast_realization_panel(
        actual_surface_frame=actual,
        forecast_frame=forecasts,
    )
    loss_frame = build_daily_loss_frame(
        panel=panel,
        positive_floor=1.0e-8,
        full_grid_weighting="uniform",
    )
    assert loss_frame.height == 2

    actual_groups = actual.partition_by("quote_date", as_dict=True)
    lookup = {key[0]: value for key, value in actual_groups.items()}
    spot_lookup = {
        row["quote_date"]: float(row["spot_1545"])
        for row in spots.iter_rows(named=True)
    }
    results = []
    for group in forecasts.partition_by(
        ["model_name", "quote_date", "target_date"],
        maintain_order=True,
    ):
        results.append(
            evaluate_model_hedging(
                model_name=str(group["model_name"][0]),
                quote_date=group["quote_date"][0],
                target_date=group["target_date"][0],
                trade_spot=spot_lookup[group["quote_date"][0]],
                target_spot=spot_lookup[group["target_date"][0]],
                actual_surface_t=surface_interpolator_from_frame(
                    lookup[group["quote_date"][0]],
                    total_variance_column="completed_total_variance",
                ),
                actual_surface_t1=surface_interpolator_from_frame(
                    lookup[group["target_date"][0]],
                    total_variance_column="completed_total_variance",
                ),
                predicted_surface_t1=surface_interpolator_from_frame(
                    group,
                    total_variance_column="predicted_total_variance",
                ),
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
        )
    summary = summarize_hedging_results(pl.DataFrame(results))
    assert summary.height == 2
