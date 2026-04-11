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


def test_load_daily_spot_frame_uses_active_underlying_price_when_bid_ask_are_zero(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 99.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 99.0,
                "is_valid_observation": True,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    spots = load_daily_spot_frame(tmp_path / "silver")

    assert spots.to_dicts() == [
        {
            "quote_date": date(2020, 8, 10),
            "spot_1545": 97.0,
        },
        {
            "quote_date": date(2020, 8, 11),
            "spot_1545": 99.0,
        },
    ]


def test_load_daily_spot_frame_uses_valid_row_median_when_spot_varies_across_option_rows(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.1,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.2,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 140.0,
                "is_valid_observation": False,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    spots = load_daily_spot_frame(tmp_path / "silver")

    assert spots.to_dicts() == [
        {
            "quote_date": date(2020, 8, 10),
            "spot_1545": 97.1,
        }
    ]


def test_load_daily_spot_frame_rejects_nonpositive_active_underlying_price(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 0.0,
                "is_valid_observation": True,
            }
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    with pytest.raises(
        ValueError,
        match="strictly positive finite stage-08 daily spot values derived from the median",
    ):
        load_daily_spot_frame(tmp_path / "silver")
