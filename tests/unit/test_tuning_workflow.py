from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import optuna
import pytest

from ivsurf.config import HpoProfileConfig
from ivsurf.training.tuning import (
    TuningResult,
    load_required_tuning_results,
    require_consistent_clean_evaluation_policy,
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
            model_name="ridge",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
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
            model_name="neural_surface",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
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
    assert loaded["neural_surface"].n_trials_pruned == 12
    policy = require_consistent_clean_evaluation_policy(loaded.values())
    assert policy.max_hpo_validation_date == date(2021, 1, 29)
    assert policy.first_clean_test_split_id == "split_0002"


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
