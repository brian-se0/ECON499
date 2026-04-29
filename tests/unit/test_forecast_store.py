from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ivsurf.evaluation.forecast_store import write_forecasts
from ivsurf.features.availability import TARGET_DECISION_TIMESTAMP_COLUMN
from ivsurf.surfaces.grid import MONEYNESS_COORDINATE, SurfaceGrid


def _grid() -> SurfaceGrid:
    return SurfaceGrid(maturity_days=(30, 90), moneyness_points=(-0.1, 0.0))


def test_write_forecasts_rejects_negative_total_variance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="negative total variance"):
        write_forecasts(
            output_path=tmp_path / "ridge.parquet",
            model_name="ridge",
            quote_dates=np.asarray([date(2021, 1, 4)], dtype=object),
            target_dates=np.asarray([date(2021, 1, 5)], dtype=object),
            split_ids=np.asarray(["split_001"], dtype=object),
            decision_timestamps=np.asarray(["2021-01-04T15:45:00-05:00"], dtype=object),
            target_decision_timestamps=np.asarray(["2021-01-05T15:45:00-05:00"], dtype=object),
            predictions=np.asarray([[-1.0e-4, 0.01, 0.02, 0.03]], dtype=np.float64),
            grid=_grid(),
            surface_config_hash="surface-hash",
            model_config_hash="model-hash",
            training_run_id="training-run",
        )


def test_write_forecasts_persists_only_finite_non_negative_values(tmp_path: Path) -> None:
    output_path = tmp_path / "ridge.parquet"
    write_forecasts(
        output_path=output_path,
        model_name="ridge",
        quote_dates=np.asarray([date(2021, 1, 4), date(2021, 1, 5)], dtype=object),
        target_dates=np.asarray([date(2021, 1, 5), date(2021, 1, 6)], dtype=object),
        split_ids=np.asarray(["split_001", "split_002"], dtype=object),
        decision_timestamps=np.asarray(
            ["2021-01-04T15:45:00-05:00", "2021-01-05T15:45:00-05:00"],
            dtype=object,
        ),
        target_decision_timestamps=np.asarray(
            ["2021-01-05T15:45:00-05:00", "2021-01-06T15:45:00-05:00"],
            dtype=object,
        ),
        predictions=np.asarray(
            [
                [0.001, 0.002, 0.003, 0.004],
                [0.005, 0.006, 0.007, 0.008],
            ],
            dtype=np.float64,
        ),
        grid=_grid(),
        surface_config_hash="surface-hash",
        model_config_hash="model-hash",
        training_run_id="training-run",
    )

    frame = pl.read_parquet(output_path)
    predicted = frame["predicted_total_variance"].to_numpy().astype(np.float64)

    assert frame.height == 8
    assert np.isfinite(predicted).all()
    assert (predicted >= 0.0).all()
    assert frame["moneyness_coordinate"].unique().to_list() == [MONEYNESS_COORDINATE]
    assert frame["surface_grid_hash"].unique().to_list() == [_grid().grid_hash]
    assert frame[TARGET_DECISION_TIMESTAMP_COLUMN].unique().sort().to_list() == [
        "2021-01-05T15:45:00-05:00",
        "2021-01-06T15:45:00-05:00",
    ]
    assert frame["split_id"].unique().sort().to_list() == ["split_001", "split_002"]
    assert frame["surface_config_hash"].unique().to_list() == ["surface-hash"]
    assert frame["model_config_hash"].unique().to_list() == ["model-hash"]
    assert frame["training_run_id"].unique().to_list() == ["training-run"]
