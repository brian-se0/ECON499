from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.config import (
    EvaluationMetricsConfig,
    HpoProfileConfig,
    NeuralModelConfig,
    RawDataConfig,
    SurfaceGridConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.evaluation.forecast_store import write_forecasts
from ivsurf.exceptions import ModelConvergenceError
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.models.base import dataset_to_matrices
from ivsurf.models.har_factor import validate_har_feature_layout
from ivsurf.models.naive import validate_naive_feature_layout
from ivsurf.progress import create_progress
from ivsurf.reproducibility import sha256_bytes, sha256_file, write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.splits.manifests import (
    WalkforwardSplit,
    load_split_manifest,
    require_split_manifest_matches_artifacts,
)
from ivsurf.splits.walkforward import clean_evaluation_splits
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.training.fit_lightgbm import fit_and_predict_lightgbm
from ivsurf.training.fit_sklearn import fit_and_predict
from ivsurf.training.fit_torch import fit_and_predict_neural
from ivsurf.training.model_factory import TUNABLE_MODEL_NAMES, make_model_from_params
from ivsurf.training.tuning import (
    CleanEvaluationPolicy,
    TuningResult,
    load_required_tuning_results,
    require_consistent_clean_evaluation_policy,
    require_matching_primary_loss_metric,
)
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


def _stable_payload_hash(payload: dict[str, object]) -> str:
    return sha256_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))


def _model_config_hash(
    *,
    model_name: str,
    base_params: dict[str, object],
    tuned_params: dict[str, object],
    hpo_profile_name: str,
    training_profile_name: str,
) -> str:
    return _stable_payload_hash(
        {
            "model_name": model_name,
            "base_params": base_params,
            "tuned_params": tuned_params,
            "hpo_profile_name": hpo_profile_name,
            "training_profile_name": training_profile_name,
        }
    )


def _single_feature_metadata_value(feature_frame: pl.DataFrame, column_name: str) -> str:
    if column_name not in feature_frame.columns:
        message = f"daily_features.parquet is missing required metadata column {column_name}."
        raise ValueError(message)
    values = feature_frame[column_name].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        message = (
            "daily_features.parquet must contain exactly one non-empty "
            f"{column_name}; found {values!r}."
        )
        raise ValueError(message)
    return values[0]


def _require_split_boundary_match(
    policy: CleanEvaluationPolicy,
    *,
    split_manifest_splits: list[WalkforwardSplit],
) -> list[WalkforwardSplit]:
    boundary, clean_splits = clean_evaluation_splits(
        split_manifest_splits,
        tuning_splits_count=policy.tuning_splits_count,
    )
    if boundary.max_hpo_validation_date != policy.max_hpo_validation_date:
        message = (
            "Split manifest boundary does not match the tuning manifests: "
            f"{boundary.max_hpo_validation_date.isoformat()} != "
            f"{policy.max_hpo_validation_date.isoformat()}."
        )
        raise ValueError(message)
    if boundary.first_clean_test_split_id != policy.first_clean_test_split_id:
        message = (
            "Split manifest clean evaluation start does not match the tuning manifests: "
            f"{boundary.first_clean_test_split_id!r} != {policy.first_clean_test_split_id!r}."
        )
        raise ValueError(message)
    return clean_splits


def _require_metadata_string(
    metadata: dict[str, object],
    field_name: str,
    *,
    expected_value: str | None = None,
) -> str:
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value:
        message = f"Stage 06 resume metadata field {field_name!r} must be a non-empty string."
        raise ValueError(message)
    if expected_value is not None and value != expected_value:
        message = (
            f"Stage 06 resume metadata field {field_name!r} is stale: "
            f"{value!r} != {expected_value!r}."
        )
        raise ValueError(message)
    return value


def _require_metadata_int(
    metadata: dict[str, object],
    field_name: str,
    *,
    expected_value: int | None = None,
) -> int:
    value = metadata.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"Stage 06 resume metadata field {field_name!r} must be an integer."
        raise ValueError(message)
    if expected_value is not None and value != expected_value:
        message = (
            f"Stage 06 resume metadata field {field_name!r} is stale: "
            f"{value!r} != {expected_value!r}."
        )
        raise ValueError(message)
    return value


def _forecast_artifact_summary(output_path: Path) -> dict[str, object]:
    summary = (
        pl.scan_parquet(output_path)
        .select(
            pl.len().alias("row_count"),
            pl.col("quote_date").n_unique().alias("n_forecast_rows"),
            pl.col("model_name").n_unique().alias("model_name_unique_count"),
            pl.col("model_name").first().alias("model_name"),
            pl.col("model_config_hash").n_unique().alias("model_config_hash_unique_count"),
            pl.col("model_config_hash").first().alias("model_config_hash"),
            pl.col("training_run_id").n_unique().alias("training_run_id_unique_count"),
            pl.col("training_run_id").first().alias("training_run_id"),
            pl.col("surface_config_hash").n_unique().alias("surface_config_hash_unique_count"),
            pl.col("surface_config_hash").first().alias("surface_config_hash"),
        )
        .collect()
        .row(0, named=True)
    )
    return dict(summary)


def _require_single_forecast_artifact_value(
    summary: dict[str, object],
    *,
    count_field_name: str,
    value_field_name: str,
    expected_value: str,
    output_path: Path,
) -> None:
    unique_count = summary[count_field_name]
    value = summary[value_field_name]
    if unique_count != 1 or value != expected_value:
        message = (
            f"Stage 06 resumed forecast artifact {output_path} has stale "
            f"{value_field_name}: count={unique_count!r}, value={value!r}, "
            f"expected={expected_value!r}."
        )
        raise ValueError(message)


def _validated_stage06_resume_metadata(
    metadata: dict[str, object],
    *,
    model_name: str,
    output_path: Path,
    n_splits: int,
    workflow_run_label: str,
    max_hpo_validation_date: str,
    first_clean_test_split_id: str,
    model_config_hash: str,
    training_run_id: str,
    surface_config_hash: str,
) -> dict[str, object]:
    forecast_artifact_hash = sha256_file(output_path)
    _require_metadata_string(metadata, "model_name", expected_value=model_name)
    _require_metadata_int(metadata, "n_splits", expected_value=n_splits)
    n_forecast_rows = _require_metadata_int(metadata, "n_forecast_rows")
    if n_forecast_rows <= 0:
        message = "Stage 06 resume metadata field 'n_forecast_rows' must be positive."
        raise ValueError(message)
    _require_metadata_string(
        metadata,
        "workflow_run_label",
        expected_value=workflow_run_label,
    )
    _require_metadata_string(
        metadata,
        "max_hpo_validation_date",
        expected_value=max_hpo_validation_date,
    )
    _require_metadata_string(
        metadata,
        "first_clean_test_split_id",
        expected_value=first_clean_test_split_id,
    )
    _require_metadata_string(
        metadata,
        "model_config_hash",
        expected_value=model_config_hash,
    )
    _require_metadata_string(
        metadata,
        "training_run_id",
        expected_value=training_run_id,
    )
    _require_metadata_string(
        metadata,
        "surface_config_hash",
        expected_value=surface_config_hash,
    )
    _require_metadata_string(
        metadata,
        "forecast_artifact_hash",
        expected_value=forecast_artifact_hash,
    )

    summary = _forecast_artifact_summary(output_path)
    if summary["row_count"] == 0:
        message = f"Stage 06 resumed forecast artifact {output_path} is empty."
        raise ValueError(message)
    if summary["n_forecast_rows"] != n_forecast_rows:
        message = (
            "Stage 06 resume metadata field 'n_forecast_rows' does not match "
            f"{output_path}: {n_forecast_rows!r} != {summary['n_forecast_rows']!r}."
        )
        raise ValueError(message)
    _require_single_forecast_artifact_value(
        summary,
        count_field_name="model_name_unique_count",
        value_field_name="model_name",
        expected_value=model_name,
        output_path=output_path,
    )
    _require_single_forecast_artifact_value(
        summary,
        count_field_name="model_config_hash_unique_count",
        value_field_name="model_config_hash",
        expected_value=model_config_hash,
        output_path=output_path,
    )
    _require_single_forecast_artifact_value(
        summary,
        count_field_name="training_run_id_unique_count",
        value_field_name="training_run_id",
        expected_value=training_run_id,
        output_path=output_path,
    )
    _require_single_forecast_artifact_value(
        summary,
        count_field_name="surface_config_hash_unique_count",
        value_field_name="surface_config_hash",
        expected_value=surface_config_hash,
        output_path=output_path,
    )
    return dict(metadata)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    metrics_config_path: Path = Path("configs/eval/metrics.yaml"),
    ridge_config_path: Path = Path("configs/models/ridge.yaml"),
    elasticnet_config_path: Path = Path("configs/models/elasticnet.yaml"),
    har_config_path: Path = Path("configs/models/har_factor.yaml"),
    lightgbm_config_path: Path = Path("configs/models/lightgbm.yaml"),
    random_forest_config_path: Path = Path("configs/models/random_forest.yaml"),
    neural_config_path: Path = Path("configs/models/neural_surface.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    run_profile_name: str | None = None,
    only_model: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    metrics_config = EvaluationMetricsConfig.model_validate(load_yaml_config(metrics_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    grid = SurfaceGrid.from_config(surface_config)
    current_surface_config_hash = sha256_file(surface_config_path)
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
        run_profile_name=run_profile_name,
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
                metrics_config_path,
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
            extra_tokens={
                "artifact_schema_version": 4,
                "run_profile_name": run_profile_name,
                "workflow_run_label": workflow_paths.run_label,
            },
        ),
    )

    feature_frame = pl.read_parquet(raw_config.gold_dir / "daily_features.parquet").sort(
        "quote_date"
    )
    feature_dataset_hash = sha256_file(raw_config.gold_dir / "daily_features.parquet")
    split_manifest = load_split_manifest(split_manifest_path)
    require_split_manifest_matches_artifacts(
        split_manifest,
        date_universe=feature_frame["quote_date"].to_list(),
        feature_dataset_hash=feature_dataset_hash,
    )
    surface_config_hash = _single_feature_metadata_value(feature_frame, "surface_config_hash")
    if surface_config_hash != current_surface_config_hash:
        message = (
            "daily_features.parquet surface_config_hash does not match the current surface "
            f"config file hash: {surface_config_hash!r} != {current_surface_config_hash!r}."
        )
        raise ValueError(message)
    matrices = dataset_to_matrices(feature_frame)
    validate_naive_feature_layout(
        feature_columns=matrices.feature_columns,
        target_columns=matrices.target_columns,
    )
    splits = split_manifest.splits

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
    require_matching_primary_loss_metric(
        tuning_results.values(),
        expected_primary_loss_metric=metrics_config.primary_loss_metric,
    )
    clean_evaluation_policy = require_consistent_clean_evaluation_policy(tuning_results.values())
    clean_splits = _require_split_boundary_match(
        clean_evaluation_policy,
        split_manifest_splits=splits,
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

    model_names = ("naive", *TUNABLE_MODEL_NAMES)
    selected_model_names = model_names
    if only_model is not None:
        if only_model not in model_names:
            message = f"Unknown model requested via only_model: {only_model!r}."
            raise ValueError(message)
        selected_model_names = (only_model,)
    if "har_factor" in selected_model_names:
        validate_har_feature_layout(
            feature_columns=matrices.feature_columns,
            target_columns=matrices.target_columns,
        )
    model_run_metadata: dict[str, dict[str, object]] = {}
    total_steps = len(selected_model_names) * len(clean_splits)
    with create_progress() as progress:
        task_id = progress.add_task("Stage 06 walk-forward forecasting", total=total_steps)
        for model_name in selected_model_names:
            output_path = workflow_paths.forecast_dir / f"{model_name}.parquet"
            model_config_hash = _model_config_hash(
                model_name=model_name,
                base_params={} if model_name == "naive" else base_param_map[model_name],
                tuned_params={} if model_name == "naive" else tuned_param_map[model_name],
                hpo_profile_name=hpo_profile.profile_name,
                training_profile_name=training_profile.profile_name,
            )
            training_run_id = _stable_payload_hash(
                {
                    "stage": "06_run_walkforward",
                    "model_name": model_name,
                    "workflow_run_label": workflow_paths.run_label,
                    "resume_context_hash": resumer.context_hash,
                    "model_config_hash": model_config_hash,
                    "surface_config_hash": surface_config_hash,
                }
            )
            if resumer.item_complete(model_name, required_output_paths=[output_path]):
                progress.update(
                    task_id,
                    description=f"Stage 06 resume: skipping completed model {model_name}",
                )
                model_run_metadata[model_name] = _validated_stage06_resume_metadata(
                    resumer.metadata_for(model_name),
                    model_name=model_name,
                    output_path=output_path,
                    n_splits=len(clean_splits),
                    workflow_run_label=workflow_paths.run_label,
                    max_hpo_validation_date=(
                        clean_evaluation_policy.max_hpo_validation_date.isoformat()
                    ),
                    first_clean_test_split_id=(
                        clean_evaluation_policy.first_clean_test_split_id
                    ),
                    model_config_hash=model_config_hash,
                    training_run_id=training_run_id,
                    surface_config_hash=surface_config_hash,
                )
                progress.advance(task_id, advance=len(clean_splits))
                continue
            resumer.clear_item(model_name, output_paths=[output_path])
            prediction_blocks: list[np.ndarray] = []
            quote_date_blocks: list[np.ndarray] = []
            target_date_blocks: list[np.ndarray] = []
            split_id_blocks: list[np.ndarray] = []
            decision_timestamp_blocks: list[np.ndarray] = []
            target_decision_timestamp_blocks: list[np.ndarray] = []
            model_metadata: dict[str, object] = {
                "model_config_hash": model_config_hash,
                "training_run_id": training_run_id,
                "surface_config_hash": surface_config_hash,
            }
            for split in clean_splits:
                split_id = split.split_id
                progress.update(
                    task_id,
                    description=f"Stage 06 walk-forward {model_name}: {split_id}",
                )
                train_index = _indices_for_dates(matrices.quote_dates, split.train_dates)
                validation_index = _indices_for_dates(matrices.quote_dates, split.validation_dates)
                test_index = _indices_for_dates(matrices.quote_dates, split.test_dates)
                try:
                    if model_name == "naive":
                        fit_index = np.concatenate([train_index, validation_index])
                        model = make_model_from_params(
                            model_name=model_name,
                            params={},
                            target_dim=matrices.targets.shape[1],
                            grid_shape=grid.shape,
                            moneyness_points=grid.moneyness_points,
                            base_neural_config=neural_config,
                        )
                        predictions = fit_and_predict(model, fit_index, test_index, matrices)
                    else:
                        model = make_model_from_params(
                            model_name=model_name,
                            params=tuned_param_map[model_name],
                            target_dim=matrices.targets.shape[1],
                            grid_shape=grid.shape,
                            moneyness_points=grid.moneyness_points,
                            base_neural_config=neural_config,
                        )
                        if model_name == "lightgbm":
                            def on_factor_progress(
                                factor_index: int,
                                factor_count: int,
                                *,
                                split_id: str = split_id,
                            ) -> None:
                                progress.update(
                                    task_id,
                                    description=(
                                        "Stage 06 walk-forward lightgbm: "
                                        f"{split_id} factor {factor_index}/{factor_count}"
                                    ),
                                )

                            predictions = fit_and_predict_lightgbm(
                                model,
                                train_index,
                                validation_index,
                                test_index,
                                matrices,
                                training_profile,
                                on_factor_progress=on_factor_progress,
                            )
                            model_metadata["lightgbm_n_factors"] = model.n_factors
                            model_metadata["lightgbm_inner_fit_count_total"] = int(
                                model_metadata.get("lightgbm_inner_fit_count_total", 0)
                            ) + model.inner_fit_count
                        elif model_name == "neural_surface":
                            predictions = fit_and_predict_neural(
                                model,
                                train_index,
                                validation_index,
                                test_index,
                                matrices,
                                training_profile,
                                validation_metric_name=metrics_config.primary_loss_metric,
                                validation_positive_floor=metrics_config.positive_floor,
                            )
                            model_metadata["neural_validation_metric"] = (
                                metrics_config.primary_loss_metric
                            )
                        else:
                            fit_index = np.concatenate([train_index, validation_index])
                            predictions = fit_and_predict(model, fit_index, test_index, matrices)
                            if model_name == "elasticnet":
                                model_metadata["elasticnet_tol"] = model.tol
                                model_metadata["elasticnet_max_iter"] = model.max_iter
                except ModelConvergenceError as exc:
                    if model_name == "elasticnet":
                        message = (
                            "Stage 06 selected ElasticNet parameters still emit convergence "
                            f"warnings on {split.split_id}: {exc}"
                        )
                        raise RuntimeError(message) from exc
                    raise
                prediction_blocks.append(predictions)
                quote_date_blocks.append(matrices.quote_dates[test_index])
                target_date_blocks.append(matrices.target_dates[test_index])
                split_id_blocks.append(
                    np.full(test_index.shape[0], split_id, dtype=object)
                )
                decision_timestamp_blocks.append(matrices.decision_timestamps[test_index])
                target_decision_timestamp_blocks.append(
                    matrices.target_decision_timestamps[test_index]
                )
                progress.advance(task_id)

            write_forecasts(
                output_path=workflow_paths.forecast_dir / f"{model_name}.parquet",
                model_name=model_name,
                quote_dates=np.concatenate(quote_date_blocks),
                target_dates=np.concatenate(target_date_blocks),
                split_ids=np.concatenate(split_id_blocks),
                decision_timestamps=np.concatenate(decision_timestamp_blocks),
                target_decision_timestamps=np.concatenate(target_decision_timestamp_blocks),
                predictions=np.vstack(prediction_blocks),
                grid=grid,
                surface_config_hash=surface_config_hash,
                model_config_hash=model_config_hash,
                training_run_id=training_run_id,
            )
            forecast_artifact_hash = sha256_file(output_path)
            item_metadata = {
                "model_name": model_name,
                "n_splits": len(clean_splits),
                "n_forecast_rows": int(np.concatenate(quote_date_blocks).shape[0]),
                "workflow_run_label": workflow_paths.run_label,
                "max_hpo_validation_date": (
                    clean_evaluation_policy.max_hpo_validation_date.isoformat()
                ),
                "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
                "forecast_artifact_hash": forecast_artifact_hash,
                **model_metadata,
            }
            item_metadata = _validated_stage06_resume_metadata(
                item_metadata,
                model_name=model_name,
                output_path=output_path,
                n_splits=len(clean_splits),
                workflow_run_label=workflow_paths.run_label,
                max_hpo_validation_date=(
                    clean_evaluation_policy.max_hpo_validation_date.isoformat()
                ),
                first_clean_test_split_id=clean_evaluation_policy.first_clean_test_split_id,
                model_config_hash=model_config_hash,
                training_run_id=training_run_id,
                surface_config_hash=surface_config_hash,
            )
            resumer.mark_complete(
                model_name,
                output_paths=[output_path],
                metadata=item_metadata,
            )
            model_run_metadata[model_name] = item_metadata
    forecast_paths = sorted_artifact_files(workflow_paths.forecast_dir, "*.parquet")
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="06_run_walkforward",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            metrics_config_path,
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
            "model_names": list(selected_model_names),
            "only_model": only_model,
            "n_splits": len(clean_splits),
            "hpo_profile_name": hpo_profile.profile_name,
            "training_profile_name": training_profile.profile_name,
            "run_profile_name": run_profile_name,
            "primary_loss_metric": metrics_config.primary_loss_metric,
            "max_hpo_validation_date": (
                clean_evaluation_policy.max_hpo_validation_date.isoformat()
            ),
            "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
            "workflow_run_label": workflow_paths.run_label,
            "model_run_metadata": model_run_metadata,
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved forecast artifacts to {workflow_paths.forecast_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
