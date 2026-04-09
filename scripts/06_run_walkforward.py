from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import typer

from ivsurf.config import (
    HpoProfileConfig,
    NeuralModelConfig,
    RawDataConfig,
    SurfaceGridConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.evaluation.forecast_store import write_forecasts
from ivsurf.models.base import dataset_to_matrices
from ivsurf.models.no_change import validate_no_change_feature_layout
from ivsurf.progress import create_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.splits.manifests import load_splits
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.training.fit_lightgbm import fit_and_predict_lightgbm
from ivsurf.training.fit_sklearn import fit_and_predict
from ivsurf.training.fit_torch import fit_and_predict_neural
from ivsurf.training.model_factory import TUNABLE_MODEL_NAMES, make_model_from_params
from ivsurf.training.tuning import TuningResult, load_required_tuning_results
from ivsurf.workflow import resolve_workflow_run_paths, tuning_manifest_path

app = typer.Typer(add_completion=False)


def _indices_for_dates(all_dates: np.ndarray, subset: tuple[str, ...]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(all_dates)}
    return np.asarray([lookup[item] for item in subset], dtype=np.int64)


def _merged_params(
    base_params: dict[str, object],
    tuning_result: TuningResult,
    *,
    expected_training_profile_name: str,
) -> dict[str, object]:
    if tuning_result.training_profile_name != expected_training_profile_name:
        message = (
            "Selected training profile does not match the training profile recorded in the "
            f"tuning manifest for {tuning_result.model_name}: "
            f"{expected_training_profile_name!r} != {tuning_result.training_profile_name!r}"
        )
        raise ValueError(message)
    merged = dict(base_params)
    merged.update(tuning_result.best_params)
    return merged


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    ridge_config_path: Path = Path("configs/models/ridge.yaml"),
    elasticnet_config_path: Path = Path("configs/models/elasticnet.yaml"),
    har_config_path: Path = Path("configs/models/har_factor.yaml"),
    lightgbm_config_path: Path = Path("configs/models/lightgbm.yaml"),
    random_forest_config_path: Path = Path("configs/models/random_forest.yaml"),
    neural_config_path: Path = Path("configs/models/neural_surface.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    grid = SurfaceGrid.from_config(surface_config)
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
    )
    split_manifest_path = raw_config.manifests_dir / "walkforward_splits.json"
    tuning_manifest_paths = [
        tuning_manifest_path(raw_config.manifests_dir, hpo_profile.profile_name, model_name)
        for model_name in TUNABLE_MODEL_NAMES
    ]
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "06_run_walkforward"),
        stage_name="06_run_walkforward",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                surface_config_path,
                ridge_config_path,
                elasticnet_config_path,
                har_config_path,
                lightgbm_config_path,
                random_forest_config_path,
                neural_config_path,
                hpo_profile_config_path,
                training_profile_config_path,
            ],
            input_artifact_paths=[
                raw_config.gold_dir / "daily_features.parquet",
                split_manifest_path,
                *tuning_manifest_paths,
            ],
            extra_tokens={"workflow_run_label": workflow_paths.run_label},
        ),
    )

    feature_frame = pl.read_parquet(
        raw_config.gold_dir / "daily_features.parquet"
    ).sort("quote_date")
    matrices = dataset_to_matrices(feature_frame)
    validate_no_change_feature_layout(
        feature_columns=matrices.feature_columns,
        target_columns=matrices.target_columns,
    )
    splits = load_splits(split_manifest_path)

    ridge_params = load_yaml_config(ridge_config_path)
    elasticnet_params = load_yaml_config(elasticnet_config_path)
    har_params = load_yaml_config(har_config_path)
    lightgbm_params = load_yaml_config(lightgbm_config_path)
    random_forest_params = load_yaml_config(random_forest_config_path)
    neural_config = NeuralModelConfig.model_validate(load_yaml_config(neural_config_path))
    neural_config = neural_config.model_copy(update={"epochs": training_profile.epochs})
    tuning_results = load_required_tuning_results(
        raw_config.manifests_dir,
        hpo_profile_name=hpo_profile.profile_name,
        model_names=TUNABLE_MODEL_NAMES,
    )

    base_param_map = {
        "ridge": {key: value for key, value in ridge_params.items() if key != "model_name"},
        "elasticnet": {
            key: value for key, value in elasticnet_params.items() if key != "model_name"
        },
        "har_factor": {key: value for key, value in har_params.items() if key != "model_name"},
        "lightgbm": {
            key: value for key, value in lightgbm_params.items() if key != "model_name"
        },
        "random_forest": {
            key: value for key, value in random_forest_params.items() if key != "model_name"
        },
        "neural_surface": neural_config.model_dump(mode="python", exclude={"model_name"}),
    }

    tuned_param_map = {
        model_name: _merged_params(
            base_param_map[model_name],
            tuning_result,
            expected_training_profile_name=training_profile.profile_name,
        )
        for model_name, tuning_result in tuning_results.items()
    }

    model_names = ("no_change", *TUNABLE_MODEL_NAMES)
    total_steps = len(model_names) * len(splits)
    with create_progress() as progress:
        task_id = progress.add_task("Stage 06 walk-forward forecasting", total=total_steps)
        for model_name in model_names:
            output_path = workflow_paths.forecast_dir / f"{model_name}.parquet"
            if resumer.item_complete(model_name, required_output_paths=[output_path]):
                progress.update(
                    task_id,
                    description=f"Stage 06 resume: skipping completed model {model_name}",
                )
                progress.advance(task_id, advance=len(splits))
                continue
            resumer.clear_item(model_name, output_paths=[output_path])
            prediction_blocks: list[np.ndarray] = []
            quote_date_blocks: list[np.ndarray] = []
            target_date_blocks: list[np.ndarray] = []
            for split in splits:
                progress.update(
                    task_id,
                    description=f"Stage 06 walk-forward {model_name}: {split.split_id}",
                )
                train_index = _indices_for_dates(matrices.quote_dates, split.train_dates)
                validation_index = _indices_for_dates(matrices.quote_dates, split.validation_dates)
                test_index = _indices_for_dates(matrices.quote_dates, split.test_dates)
                if model_name == "no_change":
                    fit_index = np.concatenate([train_index, validation_index])
                    model = make_model_from_params(
                        model_name=model_name,
                        params={},
                        target_dim=matrices.targets.shape[1],
                        grid_shape=grid.shape,
                        base_neural_config=neural_config,
                    )
                    predictions = fit_and_predict(model, fit_index, test_index, matrices)
                else:
                    model = make_model_from_params(
                        model_name=model_name,
                        params=tuned_param_map[model_name],
                        target_dim=matrices.targets.shape[1],
                        grid_shape=grid.shape,
                        base_neural_config=neural_config,
                    )
                    if model_name == "lightgbm":
                        predictions = fit_and_predict_lightgbm(
                            model,
                            train_index,
                            validation_index,
                            test_index,
                            matrices,
                            training_profile,
                        )
                    elif model_name == "neural_surface":
                        predictions = fit_and_predict_neural(
                            model,
                            train_index,
                            validation_index,
                            test_index,
                            matrices,
                            training_profile,
                        )
                    else:
                        fit_index = np.concatenate([train_index, validation_index])
                        predictions = fit_and_predict(model, fit_index, test_index, matrices)
                prediction_blocks.append(predictions)
                quote_date_blocks.append(matrices.quote_dates[test_index])
                target_date_blocks.append(matrices.target_dates[test_index])
                progress.advance(task_id)

            write_forecasts(
                output_path=workflow_paths.forecast_dir / f"{model_name}.parquet",
                model_name=model_name,
                quote_dates=np.concatenate(quote_date_blocks),
                target_dates=np.concatenate(target_date_blocks),
                predictions=np.vstack(prediction_blocks),
                grid=grid,
            )
            resumer.mark_complete(
                model_name,
                output_paths=[output_path],
                metadata={
                    "model_name": model_name,
                    "n_splits": len(splits),
                    "n_forecast_rows": int(np.concatenate(quote_date_blocks).shape[0]),
                    "workflow_run_label": workflow_paths.run_label,
                },
            )
    forecast_paths = sorted(workflow_paths.forecast_dir.glob("*.parquet"))
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="06_run_walkforward",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            ridge_config_path,
            elasticnet_config_path,
            har_config_path,
            lightgbm_config_path,
            random_forest_config_path,
            neural_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[
            raw_config.gold_dir / "daily_features.parquet",
            split_manifest_path,
            *tuning_manifest_paths,
        ],
        output_artifact_paths=forecast_paths,
        data_manifest_paths=[
            raw_config.gold_dir / "daily_features.parquet",
            *tuning_manifest_paths,
        ],
        split_manifest_path=split_manifest_path,
        random_seed=neural_config.seed,
        extra_metadata={
            "model_names": list(model_names),
            "n_splits": len(splits),
            "hpo_profile_name": hpo_profile.profile_name,
            "training_profile_name": training_profile.profile_name,
            "workflow_run_label": workflow_paths.run_label,
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved forecast artifacts to {workflow_paths.forecast_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
