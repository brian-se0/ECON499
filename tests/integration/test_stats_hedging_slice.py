from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ivsurf.cleaning.derived_fields import DECISION_TIMESTAMP_COLUMN
from ivsurf.evaluation.alignment import (
    build_forecast_realization_panel,
    load_actual_surface_frame,
    load_daily_spot_frame,
    load_forecast_frame,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.features.availability import TARGET_DECISION_TIMESTAMP_COLUMN
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import surface_interpolator_from_frame
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
)
from ivsurf.surfaces.interpolation import (
    COMPLETED_SURFACE_SCHEMA_VERSION,
    COMPLETION_STATUS_OBSERVED,
)

MATURITY_DAYS = (1, 30, 90)
MONEYNESS_POINTS = (-0.3, -0.1, 0.0, 0.1, 0.3)
GRID = SurfaceGrid(maturity_days=MATURITY_DAYS, moneyness_points=MONEYNESS_POINTS)
SURFACE_CONFIG_HASH = "surface-hash"


def _write_day(
    path: str,
    quote_date: date,
    sigma: float,
) -> None:
    rows: list[dict[str, object]] = []
    for maturity_index, maturity_days in enumerate(MATURITY_DAYS):
        for moneyness_index, moneyness_point in enumerate(MONEYNESS_POINTS):
            total_variance = sigma * (maturity_days / 365.0)
            iv = sigma**0.5
            rows.append(
                {
                    "quote_date": quote_date,
                    DECISION_TIMESTAMP_COLUMN: f"{quote_date.isoformat()}T15:45:00-05:00",
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "observed_total_variance": total_variance,
                    "observed_iv": iv,
                    "completed_total_variance": total_variance,
                    "completed_iv": iv,
                    "observed_mask": True,
                    "completion_status": COMPLETION_STATUS_OBSERVED,
                    "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
                    "surface_grid_hash": GRID.grid_hash,
                    "maturity_coordinate": MATURITY_COORDINATE,
                    "moneyness_coordinate": MONEYNESS_COORDINATE,
                    "target_surface_version": COMPLETED_SURFACE_SCHEMA_VERSION,
                    "surface_config_hash": SURFACE_CONFIG_HASH,
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
        for maturity_index, maturity_days in enumerate(MATURITY_DAYS):
            for moneyness_index, moneyness_point in enumerate(MONEYNESS_POINTS):
                forecast_rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": date(2021, 1, 4),
                        "target_date": date(2021, 1, 5),
                        "split_id": "split_0000",
                        DECISION_TIMESTAMP_COLUMN: "2021-01-04T15:45:00-05:00",
                        TARGET_DECISION_TIMESTAMP_COLUMN: "2021-01-05T15:45:00-05:00",
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
                        "surface_grid_hash": GRID.grid_hash,
                        "maturity_coordinate": MATURITY_COORDINATE,
                        "moneyness_coordinate": MONEYNESS_COORDINATE,
                        "target_surface_version": COMPLETED_SURFACE_SCHEMA_VERSION,
                        "surface_config_hash": SURFACE_CONFIG_HASH,
                        "model_config_hash": f"{model_name}-model-hash",
                        "training_run_id": f"{model_name}-training-run",
                        "predicted_total_variance": sigma * (maturity_days / 365.0),
                    }
                )
    pl.DataFrame(forecast_rows).write_parquet(forecast_dir / "models.parquet")

    silver_rows = []
    for quote_date, spot in ((date(2021, 1, 4), 100.0), (date(2021, 1, 5), 101.0)):
        silver_rows.extend(
            [
                {
                    "quote_date": quote_date,
                    "underlying_bid_1545": 0.0,
                    "underlying_ask_1545": 0.0,
                    "active_underlying_price_1545": spot,
                    "is_valid_observation": True,
                },
                {
                    "quote_date": quote_date,
                    "underlying_bid_1545": 0.0,
                    "underlying_ask_1545": 0.0,
                    "active_underlying_price_1545": spot,
                    "is_valid_observation": True,
                },
            ]
        )
    pl.DataFrame(silver_rows).write_parquet(silver_year / "spots.parquet")

    actual = load_actual_surface_frame(root / "gold", GRID)
    forecasts = load_forecast_frame(forecast_dir, GRID)
    spots = load_daily_spot_frame(root / "silver")
    assert spots.height == 2
    assert spots["spot_1545"].to_list() == [100.0, 101.0]

    panel = build_forecast_realization_panel(
        actual_surface_frame=actual,
        forecast_frame=forecasts,
        total_variance_floor=1.0e-8,
    )
    loss_frame = build_daily_loss_frame(
        panel=panel,
        positive_floor=1.0e-8,
        full_grid_weighting="uniform",
    )
    assert loss_frame.height == 2
    assert loss_frame["observed_mse_total_variance"].is_not_null().all()

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
                    grid=GRID,
                ),
                actual_surface_t1=surface_interpolator_from_frame(
                    lookup[group["target_date"][0]],
                    total_variance_column="completed_total_variance",
                    grid=GRID,
                ),
                predicted_surface_t1=surface_interpolator_from_frame(
                    group,
                    total_variance_column="predicted_total_variance",
                    grid=GRID,
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
    results_frame = pl.DataFrame(results)
    assert np.isfinite(
        results_frame.select(pl.exclude("model_name", "quote_date", "target_date")).to_numpy()
    ).all()

    summary = summarize_hedging_results(results_frame)
    assert summary.height == 2
