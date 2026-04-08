from __future__ import annotations

from pathlib import Path

import pytest

from ivsurf.training.tuning import TuningResult, load_required_tuning_results, write_tuning_result


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
