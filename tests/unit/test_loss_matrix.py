from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ivcast.evaluation.loss_matrix import daily_loss_matrix


def _loss_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_name": ["naive", "naive", "ridge", "ridge"],
            "target_date": ["2020-01-03", "2020-01-06", "2020-01-03", "2020-01-06"],
            "observed_mse_total_variance": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_daily_loss_matrix_pivots_models_by_sorted_dates() -> None:
    matrix, model_columns, target_dates = daily_loss_matrix(
        _loss_frame(),
        "observed_mse_total_variance",
    )
    assert target_dates == ("2020-01-03", "2020-01-06")
    assert set(model_columns) == {"naive", "ridge"}
    naive_index = model_columns.index("naive")
    ridge_index = model_columns.index("ridge")
    np.testing.assert_allclose(matrix[:, naive_index], [0.1, 0.2])
    np.testing.assert_allclose(matrix[:, ridge_index], [0.3, 0.4])


def test_daily_loss_matrix_rejects_duplicate_keys() -> None:
    frame = pl.concat([_loss_frame(), _loss_frame().head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        daily_loss_matrix(frame, "observed_mse_total_variance")


def test_daily_loss_matrix_rejects_missing_metric_column() -> None:
    with pytest.raises(ValueError, match="missing metric column"):
        daily_loss_matrix(_loss_frame(), "missing_metric")


def test_daily_loss_matrix_rejects_incomplete_coverage() -> None:
    frame = _loss_frame().head(3)
    with pytest.raises(ValueError, match="missing model/date coverage"):
        daily_loss_matrix(frame, "observed_mse_total_variance")


def test_daily_loss_matrix_rejects_non_finite_losses() -> None:
    frame = _loss_frame().with_columns(
        pl.when(pl.col("model_name") == "ridge")
        .then(pl.lit(float("inf")))
        .otherwise(pl.col("observed_mse_total_variance"))
        .alias("observed_mse_total_variance")
    )
    with pytest.raises(ValueError, match="non-finite"):
        daily_loss_matrix(frame, "observed_mse_total_variance")
