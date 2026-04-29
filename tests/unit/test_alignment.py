from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ivsurf.evaluation.alignment import (
    build_forecast_realization_panel,
    load_daily_spot_frame,
    require_forecast_surface_grid,
)
from ivsurf.features.availability import TARGET_DECISION_TIMESTAMP_COLUMN
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
)
from ivsurf.surfaces.interpolation import (
    COMPLETED_SURFACE_SCHEMA_VERSION,
    COMPLETION_STATUS_EXTRAPOLATED,
    COMPLETION_STATUS_INTERPOLATED,
    COMPLETION_STATUS_OBSERVED,
)

GRID = SurfaceGrid(maturity_days=(30, 90), moneyness_points=(-0.1, 0.0))


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
                        "effective_decision_timestamp": f"{quote_date.isoformat()}T15:45:00-05:00",
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "observed_total_variance": total_variance,
                        "observed_iv": sigma,
                        "completed_total_variance": total_variance,
                        "completed_iv": sigma,
                        "observed_mask": True,
                        "completion_status": COMPLETION_STATUS_OBSERVED,
                        "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
                        "surface_grid_hash": GRID.grid_hash,
                        "maturity_coordinate": MATURITY_COORDINATE,
                        "moneyness_coordinate": MONEYNESS_COORDINATE,
                        "target_surface_version": COMPLETED_SURFACE_SCHEMA_VERSION,
                        "surface_config_hash": "surface-hash",
                        "vega_sum": 1.0,
                    }
                )
    return pl.DataFrame(rows)


def _forecast_row(
    *,
    maturity_index: int,
    maturity_days: int,
    moneyness_index: int,
    moneyness_point: float,
    predicted_total_variance: float,
) -> dict[str, object]:
    return {
        "model_name": "ridge",
        "quote_date": date(2021, 1, 4),
        "target_date": date(2021, 1, 5),
        "split_id": "split_001",
        "effective_decision_timestamp": "2021-01-04T15:45:00-05:00",
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
        "surface_config_hash": "surface-hash",
        "model_config_hash": "model-hash",
        "training_run_id": "training-run",
        "predicted_total_variance": predicted_total_variance,
    }


def test_build_forecast_realization_panel_rejects_negative_predicted_total_variance() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            forecast_rows.append(
                _forecast_row(
                    maturity_index=maturity_index,
                    maturity_days=maturity_days,
                    moneyness_index=moneyness_index,
                    moneyness_point=moneyness_point,
                    predicted_total_variance=-1.0e-4 if maturity_index == 0 else 5.0e-3,
                )
            )
    with pytest.raises(ValueError, match="negative total variance"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_build_forecast_realization_panel_preserves_completion_status_masks() -> None:
    actual_surface = _actual_surface_frame().with_columns(
        pl.when((pl.col("quote_date") == date(2021, 1, 5)) & (pl.col("maturity_index") == 0))
        .then(pl.lit(COMPLETION_STATUS_INTERPOLATED))
        .when((pl.col("quote_date") == date(2021, 1, 5)) & (pl.col("maturity_index") == 1))
        .then(pl.lit(COMPLETION_STATUS_EXTRAPOLATED))
        .otherwise(pl.col("completion_status"))
        .alias("completion_status")
    )
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            forecast_rows.append(
                _forecast_row(
                    maturity_index=maturity_index,
                    maturity_days=maturity_days,
                    moneyness_index=moneyness_index,
                    moneyness_point=moneyness_point,
                    predicted_total_variance=5.0e-3,
                )
            )

    panel = build_forecast_realization_panel(
        actual_surface_frame=actual_surface,
        forecast_frame=pl.DataFrame(forecast_rows),
        total_variance_floor=1.0e-8,
    )

    assert panel["actual_completion_status"].to_list() == [
        COMPLETION_STATUS_INTERPOLATED,
        COMPLETION_STATUS_INTERPOLATED,
        COMPLETION_STATUS_EXTRAPOLATED,
        COMPLETION_STATUS_EXTRAPOLATED,
    ]
    assert panel["origin_completion_status"].to_list() == [COMPLETION_STATUS_OBSERVED] * 4
    assert panel["interpolated_weight"].to_list() == [1.0, 1.0, 0.0, 0.0]
    assert panel["extrapolated_weight"].to_list() == [0.0, 0.0, 1.0, 1.0]


def test_build_forecast_realization_panel_rejects_grid_metadata_mismatch() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            row = _forecast_row(
                maturity_index=maturity_index,
                maturity_days=maturity_days,
                moneyness_index=moneyness_index,
                moneyness_point=moneyness_point,
                predicted_total_variance=5.0e-3,
            )
            row["surface_grid_hash"] = "wrong-grid-hash"
            forecast_rows.append(row)

    with pytest.raises(ValueError, match="target grid mismatches"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_build_forecast_realization_panel_rejects_surface_config_mismatch() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            row = _forecast_row(
                maturity_index=maturity_index,
                maturity_days=maturity_days,
                moneyness_index=moneyness_index,
                moneyness_point=moneyness_point,
                predicted_total_variance=5.0e-3,
            )
            row["surface_config_hash"] = "wrong-surface-config"
            forecast_rows.append(row)

    with pytest.raises(ValueError, match="target surface config mismatches"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_build_forecast_realization_panel_rejects_missing_surface_config_hash() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            forecast_rows.append(
                _forecast_row(
                    maturity_index=maturity_index,
                    maturity_days=maturity_days,
                    moneyness_index=moneyness_index,
                    moneyness_point=moneyness_point,
                    predicted_total_variance=5.0e-3,
                )
            )

    with pytest.raises(ValueError, match="missing required columns: surface_config_hash"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=pl.DataFrame(forecast_rows).drop("surface_config_hash"),
            total_variance_floor=1.0e-8,
        )


@pytest.mark.parametrize(
    ("bad_vega", "match"),
    [
        (-1.0, "actual_vega_sum must be nonnegative"),
        (None, "nulls in required column actual_vega_sum"),
        (np.nan, "actual_vega_sum must contain only finite values"),
        (np.inf, "actual_vega_sum must contain only finite values"),
    ],
)
def test_build_forecast_realization_panel_rejects_invalid_unobserved_actual_vega_sum(
    bad_vega: float | None,
    match: str,
) -> None:
    target_cell = (pl.col("quote_date") == date(2021, 1, 5)) & (
        pl.col("maturity_index") == 0
    ) & (pl.col("moneyness_index") == 0)
    actual_surface = _actual_surface_frame().with_columns(
        pl.when(target_cell)
        .then(False)
        .otherwise(pl.col("observed_mask"))
        .alias("observed_mask"),
        pl.when(target_cell)
        .then(pl.lit(bad_vega, dtype=pl.Float64))
        .otherwise(pl.col("vega_sum"))
        .alias("vega_sum"),
    )
    forecast_rows = [
        _forecast_row(
            maturity_index=maturity_index,
            maturity_days=maturity_days,
            moneyness_index=moneyness_index,
            moneyness_point=moneyness_point,
            predicted_total_variance=5.0e-3,
        )
        for maturity_index, maturity_days in enumerate((30, 90))
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0))
    ]

    with pytest.raises(ValueError, match=match):
        build_forecast_realization_panel(
            actual_surface_frame=actual_surface,
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_build_forecast_realization_panel_rejects_nonpositive_observed_actual_vega_sum() -> None:
    target_cell = (pl.col("quote_date") == date(2021, 1, 5)) & (
        pl.col("maturity_index") == 0
    ) & (pl.col("moneyness_index") == 0)
    actual_surface = _actual_surface_frame().with_columns(
        pl.when(target_cell).then(0.0).otherwise(pl.col("vega_sum")).alias("vega_sum")
    )
    forecast_rows = [
        _forecast_row(
            maturity_index=maturity_index,
            maturity_days=maturity_days,
            moneyness_index=moneyness_index,
            moneyness_point=moneyness_point,
            predicted_total_variance=5.0e-3,
        )
        for maturity_index, maturity_days in enumerate((30, 90))
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0))
    ]

    with pytest.raises(ValueError, match="strictly positive actual_vega_sum"):
        build_forecast_realization_panel(
            actual_surface_frame=actual_surface,
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_forecast_surface_grid_contract_rejects_full_maturity_band_omission() -> None:
    forecast_rows = [
        _forecast_row(
            maturity_index=0,
            maturity_days=30,
            moneyness_index=moneyness_index,
            moneyness_point=moneyness_point,
            predicted_total_variance=5.0e-3,
        )
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0))
    ]

    with pytest.raises(ValueError, match="coordinates do not match"):
        require_forecast_surface_grid(pl.DataFrame(forecast_rows), GRID)


def test_forecast_surface_grid_contract_rejects_full_moneyness_band_omission() -> None:
    forecast_rows = [
        _forecast_row(
            maturity_index=maturity_index,
            maturity_days=maturity_days,
            moneyness_index=0,
            moneyness_point=-0.1,
            predicted_total_variance=5.0e-3,
        )
        for maturity_index, maturity_days in enumerate((30, 90))
    ]

    with pytest.raises(ValueError, match="coordinates do not match"):
        require_forecast_surface_grid(pl.DataFrame(forecast_rows), GRID)


def test_forecast_surface_grid_contract_rejects_configured_coordinate_mismatch() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            row = _forecast_row(
                maturity_index=maturity_index,
                maturity_days=maturity_days,
                moneyness_index=moneyness_index,
                moneyness_point=moneyness_point,
                predicted_total_variance=5.0e-3,
            )
            if maturity_index == 1:
                row["maturity_days"] = 91
            forecast_rows.append(row)

    with pytest.raises(ValueError, match="coordinates do not match"):
        require_forecast_surface_grid(pl.DataFrame(forecast_rows), GRID)


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
