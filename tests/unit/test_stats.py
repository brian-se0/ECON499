from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ivsurf.evaluation.alignment import (
    assert_forecast_origins_after_hpo_boundary,
    build_forecast_realization_panel,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame, daily_loss_metric_values
from ivsurf.features.availability import TARGET_DECISION_TIMESTAMP_COLUMN
from ivsurf.stats.diebold_mariano import diebold_mariano_test
from ivsurf.stats.mcs import model_confidence_set
from ivsurf.stats.spa import superior_predictive_ability_test
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

GRID = SurfaceGrid(maturity_days=(30, 90), moneyness_points=(-0.1, 0.0))


def _actual_surface_frame() -> pl.DataFrame:
    dates = [date(2021, 1, 4), date(2021, 1, 5)]
    rows: list[dict[str, object]] = []
    for quote_date in dates:
        for maturity_index, maturity_days in enumerate((30, 90)):
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                total_variance = (
                    0.04 if quote_date == dates[0] else 0.05
                ) * (maturity_days / 365.0)
                iv = float(np.sqrt(total_variance / (maturity_days / 365.0)))
                rows.append(
                    {
                        "quote_date": quote_date,
                        "effective_decision_timestamp": f"{quote_date.isoformat()}T15:45:00-05:00",
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
                        "surface_config_hash": "surface-hash",
                        "vega_sum": 1.0 + maturity_index + moneyness_index,
                    }
                )
    return pl.DataFrame(rows)


def _forecast_frame() -> pl.DataFrame:
    target_date = date(2021, 1, 5)
    quote_date = date(2021, 1, 4)
    rows: list[dict[str, object]] = []
    for model_name, sigma in (("good", 0.05), ("bad", 0.07)):
        for maturity_index, maturity_days in enumerate((30, 90)):
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": quote_date,
                        "target_date": target_date,
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
                        "model_config_hash": f"{model_name}-model-hash",
                        "training_run_id": f"{model_name}-training-run",
                        "predicted_total_variance": sigma * (maturity_days / 365.0),
                    }
                )
    return pl.DataFrame(rows)


def test_alignment_and_loss_frame_rank_forecasts_correctly() -> None:
    panel = build_forecast_realization_panel(
        actual_surface_frame=_actual_surface_frame(),
        forecast_frame=_forecast_frame(),
        total_variance_floor=1.0e-8,
    )
    loss_frame = build_daily_loss_frame(
        panel=panel,
        positive_floor=1.0e-8,
        full_grid_weighting="uniform",
    )
    assert {
        "observed_mse_total_variance",
        "full_mse_total_variance",
        "observed_qlike_total_variance",
        "full_qlike_total_variance",
        "interpolated_cell_count",
        "extrapolated_cell_count",
    }.issubset(set(loss_frame.columns))
    assert loss_frame["interpolated_cell_count"].to_list() == [0, 0]
    assert loss_frame["extrapolated_cell_count"].to_list() == [0, 0]
    ranked = loss_frame.sort("observed_wrmse_total_variance")
    assert ranked["model_name"].to_list() == ["good", "bad"]


def test_alignment_rejects_duplicate_forecast_keys() -> None:
    duplicate_forecasts = pl.concat([_forecast_frame(), _forecast_frame().head(1)])

    with pytest.raises(ValueError, match="duplicate rows"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=duplicate_forecasts,
            total_variance_floor=1.0e-8,
        )


def test_alignment_rejects_unequal_model_forecast_coverage() -> None:
    incomplete_forecasts = _forecast_frame().filter(
        ~(
            (pl.col("model_name") == "bad")
            & (pl.col("maturity_index") == 1)
            & (pl.col("moneyness_index") == 1)
        )
    )

    with pytest.raises(ValueError, match="identical coverage"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=incomplete_forecasts,
            total_variance_floor=1.0e-8,
        )


def test_observed_metrics_fail_fast_when_observed_weight_is_empty() -> None:
    actual = _actual_surface_frame().with_columns(
        pl.when(pl.col("quote_date") == date(2021, 1, 5))
        .then(0.0)
        .otherwise(pl.col("vega_sum"))
        .alias("vega_sum")
    )
    with pytest.raises(ValueError, match="strictly positive actual_vega_sum"):
        build_forecast_realization_panel(
            actual_surface_frame=actual,
            forecast_frame=_forecast_frame(),
            total_variance_floor=1.0e-8,
        )


def test_validation_observed_metric_rejects_empty_observed_weight() -> None:
    with pytest.raises(ValueError, match="strictly positive vega_weights"):
        daily_loss_metric_values(
            metric_name="observed_mse_total_variance",
            y_true=np.asarray([[0.1, 0.2]], dtype=np.float64),
            y_pred=np.asarray([[0.1, 0.3]], dtype=np.float64),
            observed_masks=np.asarray([[1.0, 1.0]], dtype=np.float64),
            vega_weights=np.asarray([[0.0, 0.0]], dtype=np.float64),
            positive_floor=1.0e-8,
        )


def test_validation_daily_loss_rejects_negative_vega_weights() -> None:
    with pytest.raises(ValueError, match="vega_weights must be nonnegative"):
        daily_loss_metric_values(
            metric_name="observed_mse_total_variance",
            y_true=np.asarray([[0.1, 0.2]], dtype=np.float64),
            y_pred=np.asarray([[0.1, 0.3]], dtype=np.float64),
            observed_masks=np.asarray([[1.0, 0.0]], dtype=np.float64),
            vega_weights=np.asarray([[1.0, -0.5]], dtype=np.float64),
            positive_floor=1.0e-8,
        )


def test_validation_daily_loss_rejects_nonbinary_masks() -> None:
    with pytest.raises(ValueError, match="binary 0/1"):
        daily_loss_metric_values(
            metric_name="observed_mse_total_variance",
            y_true=np.asarray([[0.1, 0.2]], dtype=np.float64),
            y_pred=np.asarray([[0.1, 0.3]], dtype=np.float64),
            observed_masks=np.asarray([[1.0, 0.5]], dtype=np.float64),
            vega_weights=np.asarray([[1.0, 0.0]], dtype=np.float64),
            positive_floor=1.0e-8,
        )


def test_diebold_mariano_detects_better_model() -> None:
    result = diebold_mariano_test(
        loss_a=np.asarray([2.0, 2.0, 2.0, 2.0]),
        loss_b=np.asarray([1.0, 1.0, 1.0, 1.0]),
        model_a="benchmark",
        model_b="candidate",
        alternative="greater",
        max_lag=0,
    )
    assert result.mean_differential > 0.0
    assert result.p_value == 0.0


def test_spa_and_mcs_favor_better_models() -> None:
    benchmark = np.asarray([1.2, 1.1, 1.3, 1.25, 1.15, 1.2])
    candidates = np.column_stack(
        [
            np.asarray([0.9, 0.85, 0.95, 0.9, 0.88, 0.91]),
            np.asarray([1.0, 0.98, 1.02, 1.01, 0.99, 1.0]),
            np.asarray([1.4, 1.45, 1.35, 1.5, 1.42, 1.4]),
        ]
    )
    spa = superior_predictive_ability_test(
        benchmark_losses=benchmark,
        candidate_losses=candidates,
        benchmark_model="benchmark",
        candidate_models=("good", "okay", "bad"),
        alpha=0.10,
        block_size=2,
        bootstrap_reps=200,
        seed=7,
    )
    assert spa.observed_statistic > 0.0
    assert 0.0 <= spa.p_value <= 1.0
    assert spa.alpha == 0.10
    assert "good" in spa.superior_models_by_mean

    losses = np.column_stack([benchmark, candidates])
    mcs = model_confidence_set(
        losses=losses,
        model_names=("benchmark", "good", "okay", "bad"),
        alpha=0.10,
        block_size=2,
        bootstrap_reps=200,
        seed=7,
    )
    assert "bad" not in mcs.superior_models


def test_statistical_tests_reject_nonfinite_losses() -> None:
    with pytest.raises(ValueError, match="finite"):
        diebold_mariano_test(
            loss_a=np.asarray([1.0, np.nan]),
            loss_b=np.asarray([1.0, 1.1]),
            model_a="benchmark",
            model_b="candidate",
        )
    with pytest.raises(ValueError, match="finite"):
        superior_predictive_ability_test(
            benchmark_losses=np.asarray([1.0, 1.1]),
            candidate_losses=np.asarray([[0.9], [np.inf]]),
            benchmark_model="benchmark",
            candidate_models=("candidate",),
            alpha=0.10,
            block_size=1,
            bootstrap_reps=10,
            seed=7,
        )
    with pytest.raises(ValueError, match="finite"):
        model_confidence_set(
            losses=np.asarray([[1.0, 0.9], [1.1, np.nan]]),
            model_names=("benchmark", "candidate"),
            alpha=0.10,
            block_size=1,
            bootstrap_reps=10,
            seed=7,
        )


def test_forecast_origin_guard_rejects_hpo_contaminated_rows() -> None:
    forecast_frame = pl.DataFrame(
        {
            "model_name": ["ridge", "ridge"],
            "quote_date": [date(2021, 1, 4), date(2021, 1, 5)],
            "target_date": [date(2021, 1, 5), date(2021, 1, 6)],
            "maturity_index": [0, 0],
            "maturity_days": [30, 30],
            "moneyness_index": [0, 0],
            "moneyness_point": [0.0, 0.0],
            "predicted_total_variance": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="quote_date values"):
        assert_forecast_origins_after_hpo_boundary(
            forecast_frame,
            max_hpo_validation_date=date(2021, 1, 4),
        )
