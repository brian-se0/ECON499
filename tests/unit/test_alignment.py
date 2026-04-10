from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ivsurf.evaluation.alignment import build_forecast_realization_panel, load_daily_spot_frame


def _actual_surface_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for quote_date, sigma in ((date(2021, 1, 4), 0.20), (date(2021, 1, 5), 0.22)):
        for maturity_index, maturity_days in enumerate((30, 90)):
            maturity_years = maturity_days / 365.0
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                total_variance = (sigma * sigma) * maturity_years
                rows.append(
                    {
                        "quote_date": quote_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "observed_total_variance": total_variance,
                        "observed_iv": sigma,
                        "completed_total_variance": total_variance,
                        "completed_iv": sigma,
                        "observed_mask": True,
                        "vega_sum": 1.0,
                    }
                )
    return pl.DataFrame(rows)


def test_build_forecast_realization_panel_rejects_negative_predicted_total_variance() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            forecast_rows.append(
                {
                    "model_name": "ridge",
                    "quote_date": date(2021, 1, 4),
                    "target_date": date(2021, 1, 5),
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "predicted_total_variance": -1.0e-4 if maturity_index == 0 else 5.0e-3,
                }
            )
    with pytest.raises(ValueError, match="negative total variance"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_load_daily_spot_frame_uses_underlying_midpoint_when_active_price_varies(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 100.0,
                "underlying_ask_1545": 101.0,
                "active_underlying_price_1545": 97.0,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 100.0,
                "underlying_ask_1545": 101.0,
                "active_underlying_price_1545": 94.5,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 102.0,
                "underlying_ask_1545": 103.0,
                "active_underlying_price_1545": 99.0,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 102.0,
                "underlying_ask_1545": 103.0,
                "active_underlying_price_1545": 96.5,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    spots = load_daily_spot_frame(tmp_path / "silver")

    assert spots.to_dicts() == [
        {
            "quote_date": date(2020, 8, 10),
            "spot_bid_1545": 100.0,
            "spot_ask_1545": 101.0,
            "spot_1545": 100.5,
        },
        {
            "quote_date": date(2020, 8, 11),
            "spot_bid_1545": 102.0,
            "spot_ask_1545": 103.0,
            "spot_1545": 102.5,
        },
    ]


def test_load_daily_spot_frame_rejects_multiple_underlying_bid_ask_snapshots_per_date(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 100.0,
                "underlying_ask_1545": 101.0,
                "active_underlying_price_1545": 97.0,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 100.2,
                "underlying_ask_1545": 101.2,
                "active_underlying_price_1545": 97.1,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    with pytest.raises(ValueError, match="underlying_bid_1545 and underlying_ask_1545"):
        load_daily_spot_frame(tmp_path / "silver")
