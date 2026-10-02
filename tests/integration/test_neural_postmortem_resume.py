from __future__ import annotations

from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import orjson
import polars as pl
import pytest
import yaml

from ivcast.config import TrainingProfileConfig, WalkforwardConfig
from ivcast.models.base import DatasetMatrices
from ivcast.models.neural_residual import NeuralResidualSurfaceRegressor
from ivcast.models.neural_surface import NeuralSurfaceRegressor
from ivcast.reproducibility import sha256_file
from ivcast.splits.manifests import serialize_splits
from ivcast.splits.walkforward import build_walkforward_splits, clean_evaluation_boundary
from ivcast.training.model_factory import TUNABLE_MODEL_NAMES
from ivcast.training.tuning import (
    TUNING_RESULT_SCHEMA_VERSION,
    TuningResult,
    load_tuning_result,
    write_tuning_result,
)
from ivcast.workflow import tuning_manifest_path


@pytest.fixture
def postmortem_inputs(tmp_path: Path) -> tuple[ModuleType, dict[str, Path], Path]:
    script = Path(__file__).resolve().parents[2] / "scripts/11_neural_postmortem.py"
    spec = spec_from_file_location("postmortem_resume_test", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    gold_dir = tmp_path / "gold"
    manifests_dir = tmp_path / "manifests"
    gold_dir.mkdir()
    config_payloads: dict[str, dict[str, object]] = {
        "raw_config_path": {
            "raw_options_dir": str(tmp_path / "unavailable_raw"),
            "bronze_dir": str(tmp_path / "bronze"),
            "silver_dir": str(tmp_path / "silver"),
            "gold_dir": str(gold_dir), "manifests_dir": str(manifests_dir),
        },
        "surface_config_path": {"maturity_days": [1, 7, 30],
                                "moneyness_points": [-0.1, 0.0, 0.1]},
        "metrics_config_path": {"primary_loss_metric": "observed_mse_total_variance",
                                "positive_floor": 1.0e-8},
        "neural_config_path": {"model_name": "neural_surface", "device": "cpu",
                               "hidden_width": 16},
        "residual_config_path": {"model_name": "neural_surface_residual", "device": "cpu"},
        "hpo_profile_config_path": {"profile_name": "hpo_test", "n_trials": 1,
                                    "tuning_splits_count": 1},
        "training_profile_config_path": {"profile_name": "train_test", "epochs": 1},
    }
    config_paths = {}
    for name, payload in config_payloads.items():
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(payload))
        config_paths[name] = config_path

    dates = [date(2021, 1, 4) + timedelta(days=index) for index in range(7)]
    rows: list[dict[str, object]] = []
    for row_index in range(6):
        row: dict[str, object] = {
            "quote_date": dates[row_index], "target_date": dates[row_index + 1],
            "effective_decision_timestamp": f"{dates[row_index]}T15:45:00-05:00",
            "target_effective_decision_timestamp": f"{dates[row_index + 1]}T15:45:00-05:00",
        }
        for cell in range(9):
            level = 0.01 + 0.001 * row_index + 0.0001 * cell
            row.update({
                f"feature_surface_mean_01_{cell:04d}": level,
                f"target_total_variance_{cell:04d}": level + 0.001,
                f"target_observed_mask_{cell:04d}": 1.0,
                f"target_vega_weight_{cell:04d}": 1.0,
                f"target_training_weight_{cell:04d}": 1.0,
            })
        rows.append(row)
    features = pl.DataFrame(rows)
    feature_path = gold_dir / "daily_features.parquet"
    features.write_parquet(feature_path)
    split_config = WalkforwardConfig(train_size=2, validation_size=1, test_size=1, step_size=1)
    splits = build_walkforward_splits(dates[:-1], split_config)
    serialize_splits(
        splits, manifests_dir / "walkforward_splits.json",
        walkforward_config=split_config.model_dump(mode="json"),
        sample_window={"sample_start_date": str(dates[0]), "sample_end_date": str(dates[-1])},
        date_universe=dates[:-1], feature_dataset_hash=sha256_file(feature_path),
    )
    boundary = clean_evaluation_boundary(splits, tuning_splits_count=1)
    for model_name in TUNABLE_MODEL_NAMES:
        result = TuningResult(
            schema_version=TUNING_RESULT_SCHEMA_VERSION, model_name=model_name,
            hpo_profile_name="hpo_test", training_profile_name="train_test",
            primary_loss_metric="observed_mse_total_variance", best_value=0.1,
            best_params={"hidden_width": 16} if model_name == "neural_surface" else {},
            n_trials_requested=1, n_trials_completed=1, n_trials_pruned=0,
            tuning_splits_count=1, max_hpo_validation_date=boundary.max_hpo_validation_date,
            first_clean_test_split_id=boundary.first_clean_test_split_id, seed=7,
            sampler="TPESampler", pruner="MedianPruner",
        )
        write_tuning_result(result, tuning_manifest_path(manifests_dir, "hpo_test", model_name))
    return module, config_paths, manifests_dir


def test_postmortem_retrains_when_consumed_tuning_changes_and_records_provenance(
    postmortem_inputs: tuple[ModuleType, dict[str, Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config_paths, manifests_dir = postmortem_inputs
    fit_calls: list[tuple[str, int]] = []

    def fake_fit(
        model: NeuralSurfaceRegressor | NeuralResidualSurfaceRegressor,
        train_index: np.ndarray,
        validation_index: np.ndarray,
        predict_index: np.ndarray,
        matrices: DatasetMatrices,
        training_profile: TrainingProfileConfig,
        **kwargs: object,
    ) -> np.ndarray:
        fit_calls.append((model.config.model_name, model.config.hidden_width))
        model.best_epoch = 1
        return np.asarray(
            matrices.targets[predict_index] * (1.0 + model.config.hidden_width / 10_000),
            dtype=np.float64,
        )

    # Only the costly estimator fits are replaced. File loading, dependency hashing,
    # resumption, split validation, scoring, and manifest writing execute normally.
    monkeypatch.setattr(module, "_fit_and_predict_torch_variant", fake_fit)
    module.main(**config_paths, seeds="7")
    assert ("neural_surface", 16) in fit_calls
    baseline_path = (manifests_dir / "postmortem/hpo_test__train_test"
                     / "diagnostics_neural_surface_seed7.json")
    old_baseline_hash = sha256_file(baseline_path)
    fit_calls.clear()
    module.main(**config_paths, seeds="7")
    assert not fit_calls

    tuning_path = tuning_manifest_path(manifests_dir, "hpo_test", "neural_surface")
    changed = load_tuning_result(tuning_path).model_copy(
        update={"best_params": {"hidden_width": 32}},
    )
    write_tuning_result(changed, tuning_path)
    module.main(**config_paths, seeds="7")
    assert ("neural_surface", 32) in fit_calls
    assert ("neural_surface", 16) not in fit_calls
    assert sha256_file(baseline_path) != old_baseline_hash
    latest_manifest_path = max(
        (manifests_dir / "runs/11_neural_postmortem").glob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    manifest = orjson.loads(latest_manifest_path.read_bytes())
    recorded_inputs = {Path(item["path"]): item["sha256"] for item in manifest["input_artifacts"]}
    for model_name in TUNABLE_MODEL_NAMES:
        expected = tuning_manifest_path(manifests_dir, "hpo_test", model_name).resolve()
        assert recorded_inputs[expected] == sha256_file(expected)


def test_postmortem_rejects_different_official_training_profile(
    postmortem_inputs: tuple[ModuleType, dict[str, Path], Path],
) -> None:
    module, config_paths, manifests_dir = postmortem_inputs
    tuning_path = tuning_manifest_path(manifests_dir, "hpo_test", "neural_surface")
    changed = load_tuning_result(tuning_path).model_copy(
        update={"training_profile_name": "different_train_profile"},
    )
    write_tuning_result(changed, tuning_path)
    with pytest.raises(ValueError, match="does not match the selected"):
        module.main(**config_paths, seeds="7")
