from __future__ import annotations

import polars as pl
import pytest

from ivcast.config import StatsSensitivityConfig
from ivcast.evaluation.robust_aggregation import (
    EQUAL_CELL_METRIC,
    MATURITY_BALANCED_METRIC,
    build_weighting_robustness_daily_losses,
    summarize_weighting_robustness,
)


def _panel_row(
    *,
    model_name: str,
    maturity_days: int,
    observed: bool,
    weight: float,
    actual: float,
    predicted: float,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "quote_date": "2020-01-02",
        "target_date": "2020-01-03",
        "maturity_days": maturity_days,
        "actual_observed_mask": observed,
        "observed_weight": weight,
        "actual_completed_total_variance": actual,
        "predicted_total_variance": predicted,
    }


def _synthetic_panel() -> pl.DataFrame:
    # Model A: maturity 7 has cells with errors 0.1 (weight 3) and 0.2 (weight 1);
    # maturity 30 has one cell with error 0.4. Maturity 90 is unobserved.
    rows = [
        _panel_row(
            model_name="model_a",
            maturity_days=7,
            observed=True,
            weight=3.0,
            actual=1.0,
            predicted=1.1,
        ),
        _panel_row(
            model_name="model_a",
            maturity_days=7,
            observed=True,
            weight=1.0,
            actual=1.0,
            predicted=1.2,
        ),
        _panel_row(
            model_name="model_a",
            maturity_days=30,
            observed=True,
            weight=5.0,
            actual=2.0,
            predicted=2.4,
        ),
        _panel_row(
            model_name="model_a",
            maturity_days=90,
            observed=False,
            weight=0.0,
            actual=3.0,
            predicted=9.9,
        ),
        _panel_row(
            model_name="model_b",
            maturity_days=7,
            observed=True,
            weight=2.0,
            actual=1.0,
            predicted=1.0,
        ),
        _panel_row(
            model_name="model_b",
            maturity_days=30,
            observed=True,
            weight=2.0,
            actual=2.0,
            predicted=2.1,
        ),
    ]
    return pl.DataFrame(rows)


def test_maturity_balanced_and_equal_cell_losses_match_hand_computation() -> None:
    daily = build_weighting_robustness_daily_losses(_synthetic_panel())

    row_a = daily.filter(pl.col("model_name") == "model_a").row(0, named=True)
    # Maturity 7 slice: (3 * 0.1^2 + 1 * 0.2^2) / 4 = 0.0175; maturity 30 slice: 0.16.
    expected_maturity_balanced_a = (0.0175 + 0.16) / 2.0
    # Equal-cell over three observed cells: (0.01 + 0.04 + 0.16) / 3.
    expected_equal_cell_a = (0.01 + 0.04 + 0.16) / 3.0
    assert row_a[MATURITY_BALANCED_METRIC] == pytest.approx(expected_maturity_balanced_a)
    assert row_a[EQUAL_CELL_METRIC] == pytest.approx(expected_equal_cell_a)
    assert row_a["observed_maturity_slice_count"] == 2
    assert row_a["observed_cell_count"] == 3

    row_b = daily.filter(pl.col("model_name") == "model_b").row(0, named=True)
    assert row_b[MATURITY_BALANCED_METRIC] == pytest.approx((0.0 + 0.01) / 2.0)
    assert row_b[EQUAL_CELL_METRIC] == pytest.approx((0.0 + 0.01) / 2.0)


def test_unobserved_only_maturity_slices_are_excluded() -> None:
    daily = build_weighting_robustness_daily_losses(_synthetic_panel())
    row_a = daily.filter(pl.col("model_name") == "model_a").row(0, named=True)
    assert row_a["observed_maturity_slice_count"] == 2


def test_missing_required_column_raises() -> None:
    panel = _synthetic_panel().drop("observed_weight")
    with pytest.raises(ValueError, match="missing required columns"):
        build_weighting_robustness_daily_losses(panel)


def test_nonpositive_observed_weight_raises() -> None:
    panel = _synthetic_panel().with_columns(
        pl.when(pl.col("actual_observed_mask"))
        .then(pl.lit(0.0))
        .otherwise(pl.col("observed_weight"))
        .alias("observed_weight")
    )
    with pytest.raises(ValueError, match="strictly positive observed_weight"):
        build_weighting_robustness_daily_losses(panel)


def test_non_finite_observed_prediction_raises() -> None:
    panel = _synthetic_panel().with_columns(
        pl.when(pl.col("actual_observed_mask"))
        .then(pl.lit(float("nan")))
        .otherwise(pl.col("predicted_total_variance"))
        .alias("predicted_total_variance")
    )
    with pytest.raises(ValueError, match="finite"):
        build_weighting_robustness_daily_losses(panel)


def test_mismatched_model_date_coverage_raises() -> None:
    extra_row = _panel_row(
        model_name="model_b",
        maturity_days=7,
        observed=True,
        weight=1.0,
        actual=1.0,
        predicted=1.0,
    )
    extra_row["quote_date"] = "2020-01-03"
    extra_row["target_date"] = "2020-01-06"
    panel = pl.concat([_synthetic_panel(), pl.DataFrame([extra_row])])
    with pytest.raises(ValueError, match="same observed target-date coverage"):
        build_weighting_robustness_daily_losses(panel)


def test_summarize_ranks_models_under_each_weighting() -> None:
    daily = build_weighting_robustness_daily_losses(_synthetic_panel())
    official = pl.DataFrame(
        {
            "model_name": ["model_a", "model_b"],
            "observed_mse_total_variance": [0.5, 0.1],
        }
    )
    summary = summarize_weighting_robustness(daily, official)
    assert summary["model_name"].to_list() == ["model_b", "model_a"]
    assert summary["rank_official"].to_list() == [1, 2]
    assert summary["rank_maturity_balanced"].to_list() == [1, 2]
    assert summary["rank_equal_cell"].to_list() == [1, 2]
    assert summary["n_target_dates"].to_list() == [1, 1]


def test_summarize_rejects_model_set_mismatch() -> None:
    daily = build_weighting_robustness_daily_losses(_synthetic_panel())
    official = pl.DataFrame(
        {
            "model_name": ["model_a"],
            "observed_mse_total_variance": [0.5],
        }
    )
    with pytest.raises(ValueError, match="same model set"):
        summarize_weighting_robustness(daily, official)


def test_stats_sensitivity_config_defaults_are_valid() -> None:
    config = StatsSensitivityConfig()
    assert config.bootstrap_reps == 10_000
    assert config.block_sizes == (5, 10, 20)
    assert config.dm_max_lags == (0, 5, 10)


def test_stats_sensitivity_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        StatsSensitivityConfig(dm_max_lags=(0, -1))
    with pytest.raises(ValueError):
        StatsSensitivityConfig(block_sizes=())
    with pytest.raises(ValueError):
        StatsSensitivityConfig(loss_metrics=("a", "a"))
