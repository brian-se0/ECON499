from __future__ import annotations

import json
from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import optuna
import polars as pl
import pytest

from ivsurf.config import HpoProfileConfig
from ivsurf.evaluation.loss_panels import build_daily_loss_frame, mean_daily_loss_metric
from ivsurf.surfaces.interpolation import COMPLETION_STATUS_OBSERVED
from ivsurf.training.tuning import (
    TUNING_RESULT_SCHEMA_VERSION,
    TuningResult,
    load_required_tuning_results,
    load_tuning_result,
    require_consistent_clean_evaluation_policy,
    require_matching_primary_loss_metric,
    write_tuning_result,
)


def _load_stage05_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "05_tune_models.py"
    spec = spec_from_file_location("stage05_tune_models_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_required_tuning_results_fails_fast_when_profile_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_required_tuning_results(
            tmp_path,
            hpo_profile_name="hpo_30_trials",
            model_names=("ridge", "neural_surface"),
        )


def test_load_required_tuning_results_reads_profile_specific_manifests(tmp_path: Path) -> None:
    ridge_path = tmp_path / "tuning" / "hpo_30_trials" / "ridge.json"
    neural_path = tmp_path / "tuning" / "hpo_30_trials" / "neural_surface.json"
    write_tuning_result(
        TuningResult(
            schema_version=TUNING_RESULT_SCHEMA_VERSION,
            model_name="ridge",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
            primary_loss_metric="observed_mse_total_variance",
            best_value=0.1,
            best_params={"alpha": 1.0},
            n_trials_requested=30,
            n_trials_completed=30,
            n_trials_pruned=0,
            tuning_splits_count=3,
            max_hpo_validation_date=date(2021, 1, 29),
            first_clean_test_split_id="split_0002",
            seed=7,
            sampler="TPESampler",
            pruner="MedianPruner",
        ),
        ridge_path,
    )
    write_tuning_result(
        TuningResult(
            schema_version=TUNING_RESULT_SCHEMA_VERSION,
            model_name="neural_surface",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
            primary_loss_metric="observed_mse_total_variance",
            best_value=0.05,
            best_params={"hidden_width": 64, "depth": 2},
            n_trials_requested=30,
            n_trials_completed=18,
            n_trials_pruned=12,
            tuning_splits_count=3,
            max_hpo_validation_date=date(2021, 1, 29),
            first_clean_test_split_id="split_0002",
            seed=7,
            sampler="TPESampler",
            pruner="MedianPruner",
        ),
        neural_path,
    )

    loaded = load_required_tuning_results(
        tmp_path,
        hpo_profile_name="hpo_30_trials",
        model_names=("ridge", "neural_surface"),
    )

    assert tuple(sorted(loaded)) == ("neural_surface", "ridge")
    assert loaded["ridge"].best_params["alpha"] == 1.0
    assert loaded["ridge"].primary_loss_metric == "observed_mse_total_variance"
    assert loaded["neural_surface"].n_trials_pruned == 12
    policy = require_consistent_clean_evaluation_policy(loaded.values())
    assert policy.max_hpo_validation_date == date(2021, 1, 29)
    assert policy.first_clean_test_split_id == "split_0002"
    assert (
        require_matching_primary_loss_metric(
            loaded.values(),
            expected_primary_loss_metric="observed_mse_total_variance",
        )
        == "observed_mse_total_variance"
    )


def test_load_tuning_result_rejects_missing_schema_version(tmp_path: Path) -> None:
    stale_path = tmp_path / "ridge.json"
    stale_path.write_text(
        json.dumps(
            {
                "model_name": "ridge",
                "hpo_profile_name": "hpo_30_trials",
                "training_profile_name": "train_30_epochs",
                "primary_loss_metric": "observed_mse_total_variance",
                "best_value": 0.1,
                "best_params": {"alpha": 1.0},
                "n_trials_requested": 30,
                "n_trials_completed": 30,
                "n_trials_pruned": 0,
                "tuning_splits_count": 3,
                "max_hpo_validation_date": "2021-01-29",
                "first_clean_test_split_id": "split_0002",
                "seed": 7,
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema is stale or invalid"):
        load_tuning_result(stale_path)


def test_require_matching_primary_loss_metric_rejects_stale_metric(tmp_path: Path) -> None:
    ridge_path = tmp_path / "tuning" / "hpo_30_trials" / "ridge.json"
    write_tuning_result(
        TuningResult(
            schema_version=TUNING_RESULT_SCHEMA_VERSION,
            model_name="ridge",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
            primary_loss_metric="observed_qlike_total_variance",
            best_value=0.1,
            best_params={"alpha": 1.0},
            n_trials_requested=30,
            n_trials_completed=30,
            n_trials_pruned=0,
            tuning_splits_count=3,
            max_hpo_validation_date=date(2021, 1, 29),
            first_clean_test_split_id="split_0002",
            seed=7,
            sampler="TPESampler",
            pruner="MedianPruner",
        ),
        ridge_path,
    )

    loaded = load_required_tuning_results(
        tmp_path,
        hpo_profile_name="hpo_30_trials",
        model_names=("ridge",),
    )

    with pytest.raises(ValueError, match="different primary_loss_metric"):
        require_matching_primary_loss_metric(
            loaded.values(),
            expected_primary_loss_metric="observed_mse_total_variance",
        )


def test_stage05_uses_the_configured_optuna_sampler() -> None:
    stage05 = _load_stage05_module()
    sampler = stage05._make_sampler(  # pyright: ignore[reportAttributeAccessIssue]
        HpoProfileConfig(
            profile_name="hpo_30_trials",
            n_trials=1,
            tuning_splits_count=3,
            seed=7,
            sampler="tpe",
        )
    )

    assert isinstance(sampler, optuna.samplers.TPESampler)


def test_stage05_daily_loss_metric_matches_stage07_daily_aggregation() -> None:
    y_true = np.asarray(
        [
            [0.10, 0.20],
            [0.30, 0.40],
        ],
        dtype=np.float64,
    )
    y_pred = np.asarray(
        [
            [0.11, 0.18],
            [0.28, 0.42],
        ],
        dtype=np.float64,
    )
    observed_masks = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    vega_weights = np.asarray(
        [
            [2.0, 0.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )

    selected_metric = mean_daily_loss_metric(
        metric_name="observed_mse_total_variance",
        y_true=y_true,
        y_pred=y_pred,
        observed_masks=observed_masks,
        vega_weights=vega_weights,
        positive_floor=1.0e-8,
    )

    panel = pl.DataFrame(
        {
            "model_name": ["candidate"] * 4,
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 5)],
            "target_date": [date(2021, 1, 5), date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 6)],
            "actual_completed_total_variance": y_true.reshape(-1),
            "predicted_total_variance": y_pred.reshape(-1),
            "actual_completed_iv": np.sqrt(y_true.reshape(-1)),
            "predicted_iv": np.sqrt(y_pred.reshape(-1)),
            "actual_iv_change": np.zeros(4, dtype=np.float64),
            "predicted_iv_change": np.zeros(4, dtype=np.float64),
            "actual_observed_mask": observed_masks.reshape(-1) > 0.5,
            "actual_completion_status": [COMPLETION_STATUS_OBSERVED] * 4,
            "observed_weight": (observed_masks * vega_weights).reshape(-1),
            "full_grid_weight": np.ones(4, dtype=np.float64),
        }
    )
    daily_loss_frame = build_daily_loss_frame(
        panel=panel,
        positive_floor=1.0e-8,
        full_grid_weighting="uniform",
    )
    expected = cast(float, daily_loss_frame["observed_mse_total_variance"].mean())

    assert selected_metric == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
