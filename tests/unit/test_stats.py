from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from ivsurf.evaluation.alignment import build_forecast_realization_panel
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.stats.diebold_mariano import diebold_mariano_test
from ivsurf.stats.mcs import model_confidence_set
from ivsurf.stats.spa import superior_predictive_ability_test


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
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "observed_total_variance": total_variance,
                        "observed_iv": iv,
                        "completed_total_variance": total_variance,
                        "completed_iv": iv,
                        "observed_mask": True,
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
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
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
    }.issubset(set(loss_frame.columns))
    ranked = loss_frame.sort("observed_wrmse_total_variance")
    assert ranked["model_name"].to_list() == ["good", "bad"]


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
        block_size=2,
        bootstrap_reps=200,
        seed=7,
    )
    assert spa.observed_statistic > 0.0
    assert 0.0 <= spa.p_value <= 1.0
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
