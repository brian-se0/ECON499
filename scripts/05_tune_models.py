from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
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
from ivsurf.evaluation.loss_panels import mean_daily_loss_metric
from ivsurf.evaluation.metrics import weighted_rmse
from ivsurf.exceptions import ModelConvergenceError
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.models.base import DatasetMatrices, dataset_to_matrices
from ivsurf.models.elasticnet import ElasticNetSurfaceModel
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.progress import create_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.splits.manifests import WalkforwardSplit, load_splits
from ivsurf.splits.walkforward import clean_evaluation_boundary
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.training.fit_lightgbm import fit_and_predict_lightgbm
from ivsurf.training.fit_sklearn import fit_and_predict
from ivsurf.training.fit_torch import fit_and_predict_neural
from ivsurf.training.model_factory import suggest_model_from_trial
from ivsurf.training.tuning import TuningResult, write_tuning_result
from ivsurf.workflow import tuning_diagnostics_path, tuning_manifest_path

app = typer.Typer(add_completion=False)


def _indices_for_dates(all_dates: np.ndarray, subset: tuple[str, ...]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(all_dates)}
    return np.asarray([lookup[item] for item in subset], dtype=np.int64)


def _legacy_pooled_validation_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_masks: np.ndarray,
    vega_weights: np.ndarray,
) -> float:
    weights = observed_masks * np.maximum(vega_weights, 0.0)
    return weighted_rmse(
        y_true=y_true.reshape(-1),
        y_pred=y_pred.reshape(-1),
        weights=weights.reshape(-1),
    )


def _make_pruner(hpo_profile: HpoProfileConfig) -> optuna.pruners.BasePruner:
    if hpo_profile.pruner.name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=hpo_profile.pruner.n_startup_trials,
            n_warmup_steps=hpo_profile.pruner.n_warmup_steps,
            interval_steps=hpo_profile.pruner.interval_steps,
        )
    message = f"Unsupported Optuna pruner: {hpo_profile.pruner.name}"
    raise ValueError(message)


def _make_sampler(hpo_profile: HpoProfileConfig) -> optuna.samplers.BaseSampler:
    if hpo_profile.sampler == "tpe":
        return optuna.samplers.TPESampler(seed=hpo_profile.seed)
    message = f"Unsupported Optuna sampler: {hpo_profile.sampler}"
    raise ValueError(message)


def _model_specific_diagnostics(model: object) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "elasticnet_tol": None,
        "elasticnet_max_iter": None,
        "elasticnet_convergence_warning_message": None,
        "lightgbm_n_factors": None,
        "lightgbm_inner_fit_count": None,
        "neural_best_epoch": None,
        "neural_prediction_mean": None,
        "neural_target_mean": None,
        "neural_prediction_target_ratio": None,
        "neural_prediction_below_1e_6_share": None,
        "neural_calendar_penalty": None,
        "neural_convexity_penalty": None,
        "neural_roughness_penalty": None,
    }
    if isinstance(model, ElasticNetSurfaceModel):
        diagnostics.update(
            {
                "elasticnet_tol": model.tol,
                "elasticnet_max_iter": model.max_iter,
                "elasticnet_convergence_warning_message": model.last_convergence_warning_message,
            }
        )
    if isinstance(model, LightGBMSurfaceModel):
        diagnostics.update(
            {
                "lightgbm_n_factors": model.n_factors,
                "lightgbm_inner_fit_count": model.inner_fit_count,
            }
        )
    if isinstance(model, NeuralSurfaceRegressor):
        diagnostics["neural_best_epoch"] = model.best_epoch
        if model.validation_diagnostics is not None:
            diagnostics.update(
                {
                    "neural_prediction_mean": model.validation_diagnostics.prediction_mean,
                    "neural_target_mean": model.validation_diagnostics.target_mean,
                    "neural_prediction_target_ratio": (
                        model.validation_diagnostics.prediction_target_ratio
                    ),
                    "neural_prediction_below_1e_6_share": (
                        model.validation_diagnostics.prediction_below_1e_6_share
                    ),
                    "neural_calendar_penalty": model.validation_diagnostics.calendar_penalty,
                    "neural_convexity_penalty": model.validation_diagnostics.convexity_penalty,
                    "neural_roughness_penalty": model.validation_diagnostics.roughness_penalty,
                }
            )
    return diagnostics


def _objective_factory(
    *,
    model_name: str,
    matrices: DatasetMatrices,
    tuning_splits: list[WalkforwardSplit],
    grid: SurfaceGrid,
    base_neural_config: NeuralModelConfig,
    base_lightgbm_params: dict[str, object],
    training_profile: TrainingProfileConfig,
    primary_loss_metric: str,
    positive_floor: float,
    diagnostic_rows: list[dict[str, object]],
    on_split_complete: Callable[[str], None] | None = None,
    on_inner_progress: Callable[[str], None] | None = None,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        scores: list[float] = []
        for split_index, split in enumerate(tuning_splits):
            split_id = split.split_id
            train_index = _indices_for_dates(matrices.quote_dates, split.train_dates)
            validation_index = _indices_for_dates(matrices.quote_dates, split.validation_dates)
            model = suggest_model_from_trial(
                model_name=model_name,
                trial=trial,
                target_dim=matrices.targets.shape[1],
                grid_shape=grid.shape,
                base_neural_config=base_neural_config,
                base_lightgbm_params=base_lightgbm_params,
            )
            split_started_at = perf_counter()
            row_written = False
            try:
                if model_name == "lightgbm":
                    predictions = fit_and_predict_lightgbm(
                        model,
                        train_index,
                        validation_index,
                        validation_index,
                        matrices,
                        training_profile,
                        on_factor_progress=(
                            None
                            if on_inner_progress is None
                            else lambda factor_index, factor_count, split_id=split_id: (
                                on_inner_progress(
                                    "Stage 05 tuning lightgbm: "
                                    f"trial {trial.number + 1} split {split_id} "
                                    f"factor {factor_index}/{factor_count}"
                                )
                            )
                        ),
                    )
                elif model_name == "neural_surface":
                    predictions = fit_and_predict_neural(
                        model,
                        train_index,
                        validation_index,
                        validation_index,
                        matrices,
                        training_profile,
                        trial=trial,
                        trial_step_offset=split_index * (training_profile.epochs + 1),
                        validation_metric_name=primary_loss_metric,
                        validation_positive_floor=positive_floor,
                    )
                else:
                    predictions = fit_and_predict(model, train_index, validation_index, matrices)
                selected_metric_value = mean_daily_loss_metric(
                    metric_name=primary_loss_metric,
                    y_true=matrices.targets[validation_index],
                    y_pred=predictions,
                    observed_masks=matrices.observed_masks[validation_index],
                    vega_weights=matrices.vega_weights[validation_index],
                    positive_floor=positive_floor,
                )
                legacy_pooled_wrmse = _legacy_pooled_validation_score(
                    y_true=matrices.targets[validation_index],
                    y_pred=predictions,
                    observed_masks=matrices.observed_masks[validation_index],
                    vega_weights=matrices.vega_weights[validation_index],
                )
                scores.append(selected_metric_value)
                diagnostic_rows.append(
                    {
                        "model_name": model_name,
                        "trial_number": trial.number,
                        "split_index": split_index,
                        "split_id": split.split_id,
                        "validation_rows": int(validation_index.size),
                        "validation_quote_start": split.validation_dates[0],
                        "validation_quote_end": split.validation_dates[-1],
                        "selected_metric_name": primary_loss_metric,
                        "selected_metric_value": selected_metric_value,
                        "running_mean_selected_metric": float(np.mean(scores)),
                        "legacy_pooled_observed_wrmse": legacy_pooled_wrmse,
                        "split_runtime_seconds": perf_counter() - split_started_at,
                        "trial_status": "completed_split",
                        "warning_reason": None,
                        **_model_specific_diagnostics(model),
                    }
                )
                row_written = True
                split_report_step = (
                    split_index * (training_profile.epochs + 1) + training_profile.epochs
                )
                trial.report(float(np.mean(scores)), split_report_step)
                if trial.should_prune():
                    diagnostic_rows[-1]["trial_status"] = "pruned"
                    diagnostic_rows[-1]["warning_reason"] = "optuna_pruner"
                    raise optuna.TrialPruned()
            except ModelConvergenceError as exc:
                diagnostic_rows.append(
                    {
                        "model_name": model_name,
                        "trial_number": trial.number,
                        "split_index": split_index,
                        "split_id": split.split_id,
                        "validation_rows": int(validation_index.size),
                        "validation_quote_start": split.validation_dates[0],
                        "validation_quote_end": split.validation_dates[-1],
                        "selected_metric_name": primary_loss_metric,
                        "selected_metric_value": None,
                        "running_mean_selected_metric": (
                            None if not scores else float(np.mean(scores))
                        ),
                        "legacy_pooled_observed_wrmse": None,
                        "split_runtime_seconds": perf_counter() - split_started_at,
                        "trial_status": "pruned",
                        "warning_reason": "elasticnet_convergence_warning",
                        **_model_specific_diagnostics(model),
                        "elasticnet_convergence_warning_message": str(exc),
                    }
                )
                trial.set_user_attr("prune_reason", "elasticnet_convergence_warning")
                raise optuna.TrialPruned(str(exc)) from exc
            except optuna.TrialPruned:
                if not row_written:
                    diagnostic_rows.append(
                        {
                            "model_name": model_name,
                            "trial_number": trial.number,
                            "split_index": split_index,
                            "split_id": split.split_id,
                            "validation_rows": int(validation_index.size),
                            "validation_quote_start": split.validation_dates[0],
                            "validation_quote_end": split.validation_dates[-1],
                            "selected_metric_name": primary_loss_metric,
                            "selected_metric_value": None,
                            "running_mean_selected_metric": (
                                None if not scores else float(np.mean(scores))
                            ),
                            "legacy_pooled_observed_wrmse": None,
                            "split_runtime_seconds": perf_counter() - split_started_at,
                            "trial_status": "pruned",
                            "warning_reason": "optuna_pruner",
                            **_model_specific_diagnostics(model),
                        }
                    )
                raise
            if on_split_complete is not None:
                on_split_complete(
                    f"Stage 05 tuning {model_name}: trial {trial.number + 1} "
                    f"split {split.split_id}"
                )
        return float(np.mean(scores))

    return objective


@app.command()
def main(
    model_name: str,
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    metrics_config_path: Path = Path("configs/eval/metrics.yaml"),
    lightgbm_config_path: Path = Path("configs/models/lightgbm.yaml"),
    neural_config_path: Path = Path("configs/models/neural_surface.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    metrics_config = EvaluationMetricsConfig.model_validate(load_yaml_config(metrics_config_path))
    lightgbm_params = load_yaml_config(lightgbm_config_path)
    neural_config = NeuralModelConfig.model_validate(load_yaml_config(neural_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    neural_config = neural_config.model_copy(update={"epochs": training_profile.epochs})
    grid = SurfaceGrid.from_config(surface_config)
    split_manifest_path = raw_config.manifests_dir / "walkforward_splits.json"
    output_path = tuning_manifest_path(
        raw_config.manifests_dir,
        hpo_profile.profile_name,
        model_name,
    )
    diagnostics_path = tuning_diagnostics_path(
        raw_config.manifests_dir,
        hpo_profile.profile_name,
        model_name,
    )
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "05_tune_models"),
        stage_name="05_tune_models",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                surface_config_path,
                metrics_config_path,
                lightgbm_config_path,
                neural_config_path,
                hpo_profile_config_path,
                training_profile_config_path,
            ],
            input_artifact_paths=[
                raw_config.gold_dir / "daily_features.parquet",
                split_manifest_path,
            ],
            extra_tokens={"model_name": model_name},
        ),
    )
    resume_item_id = model_name
    if resumer.item_complete(
        resume_item_id,
        required_output_paths=[output_path, diagnostics_path],
    ):
        typer.echo(
            f"Stage 05 resume: tuning manifest already complete for {model_name} "
            "under the current context; skipping."
        )
        return
    resumer.clear_item(resume_item_id, output_paths=[output_path, diagnostics_path])

    feature_frame = pl.read_parquet(raw_config.gold_dir / "daily_features.parquet").sort(
        "quote_date"
    )
    matrices = dataset_to_matrices(feature_frame)
    splits = load_splits(split_manifest_path)
    tuning_splits = splits[: hpo_profile.tuning_splits_count]
    if not tuning_splits:
        message = "No walk-forward splits available for tuning."
        raise ValueError(message)
    evaluation_boundary = clean_evaluation_boundary(
        splits,
        tuning_splits_count=hpo_profile.tuning_splits_count,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=_make_sampler(hpo_profile),
        pruner=_make_pruner(hpo_profile),
    )
    diagnostic_rows: list[dict[str, object]] = []
    total_progress_steps = hpo_profile.n_trials * len(tuning_splits)
    with create_progress() as progress:
        task_id = progress.add_task(
            f"Stage 05 tuning {model_name}",
            total=total_progress_steps,
        )

        def _on_split_complete(description: str) -> None:
            progress.update(task_id, description=description)
            progress.advance(task_id)

        def _on_inner_progress(description: str) -> None:
            progress.update(task_id, description=description)

        study.optimize(
            _objective_factory(
                model_name=model_name,
                matrices=matrices,
                tuning_splits=tuning_splits,
                grid=grid,
                base_neural_config=neural_config,
                base_lightgbm_params=lightgbm_params,
                training_profile=training_profile,
                primary_loss_metric=metrics_config.primary_loss_metric,
                positive_floor=metrics_config.positive_floor,
                diagnostic_rows=diagnostic_rows,
                on_split_complete=_on_split_complete,
                on_inner_progress=_on_inner_progress,
            ),
            n_trials=hpo_profile.n_trials,
        )

    diagnostic_frame = (
        pl.DataFrame(diagnostic_rows)
        if diagnostic_rows
        else pl.DataFrame(
            {
                "model_name": pl.Series([], dtype=pl.String),
                "trial_number": pl.Series([], dtype=pl.Int64),
                "split_index": pl.Series([], dtype=pl.Int64),
                "split_id": pl.Series([], dtype=pl.String),
            }
        )
    )
    if not diagnostic_frame.is_empty():
        diagnostic_frame = diagnostic_frame.sort(["trial_number", "split_index"])
    write_parquet_frame(diagnostic_frame, diagnostics_path)

    completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    pruned_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    if not completed_trials:
        message = (
            f"Optuna completed no trials for {model_name} under HPO profile "
            f"{hpo_profile.profile_name!r}."
        )
        raise RuntimeError(message)
    result = TuningResult(
        model_name=model_name,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials_requested=hpo_profile.n_trials,
        n_trials_completed=len(completed_trials),
        n_trials_pruned=len(pruned_trials),
        tuning_splits_count=hpo_profile.tuning_splits_count,
        max_hpo_validation_date=evaluation_boundary.max_hpo_validation_date,
        first_clean_test_split_id=evaluation_boundary.first_clean_test_split_id,
        seed=hpo_profile.seed,
        sampler=type(study.sampler).__name__,
        pruner=type(study.pruner).__name__,
    )
    write_tuning_result(result, output_path)

    model_metadata: dict[str, object] = {
        "model_name": model_name,
        "hpo_profile_name": hpo_profile.profile_name,
        "training_profile_name": training_profile.profile_name,
        "n_trials_requested": hpo_profile.n_trials,
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "max_hpo_validation_date": evaluation_boundary.max_hpo_validation_date.isoformat(),
        "first_clean_test_split_id": evaluation_boundary.first_clean_test_split_id,
        "primary_loss_metric": metrics_config.primary_loss_metric,
        "tuning_diagnostics_path": str(diagnostics_path),
    }
    if model_name == "lightgbm":
        lightgbm_n_factors = study.best_params.get("n_factors", lightgbm_params.get("n_factors"))
        model_metadata["lightgbm_n_factors"] = int(lightgbm_n_factors)
        model_metadata["lightgbm_inner_fit_count"] = int(lightgbm_n_factors)

    resumer.mark_complete(
        resume_item_id,
        output_paths=[output_path, diagnostics_path],
        metadata=model_metadata,
    )
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="05_tune_models",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            metrics_config_path,
            lightgbm_config_path,
            neural_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[raw_config.gold_dir / "daily_features.parquet", split_manifest_path],
        output_artifact_paths=[output_path, diagnostics_path],
        data_manifest_paths=[raw_config.gold_dir / "daily_features.parquet"],
        split_manifest_path=split_manifest_path,
        random_seed=hpo_profile.seed,
        extra_metadata={
            **model_metadata,
            "resume_context_hash": resumer.context_hash,
            "legacy_validation_metric": "pooled_observed_wrmse_total_variance",
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved tuning results to {output_path}")
    typer.echo(f"Saved tuning diagnostics to {diagnostics_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
