"""Post-hoc neural residual-parameterization diagnostic.

Reruns the flagship neural walk-forward under multiple seeds and compares it
against a persistence-anchored residual variant tuned with the saved official
HPO protocol. Everything is isolated from the official pipeline: the residual
tuning manifest, all forecast losses, and the summary live under
``data/manifests/postmortem/<run_label>/``. The official tuning directory,
forecast directory, and Stage-07 artifacts are read-only inputs, and the
official model universe is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import orjson
import polars as pl
import typer

from ivcast.config import (
    EvaluationMetricsConfig,
    HpoProfileConfig,
    NeuralModelConfig,
    RawDataConfig,
    SurfaceGridConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivcast.evaluation.loss_panels import daily_loss_metric_values
from ivcast.io.atomic import write_bytes_atomic
from ivcast.io.parquet import write_parquet_frame
from ivcast.models.base import DatasetMatrices, dataset_to_matrices
from ivcast.models.naive import validate_naive_feature_layout
from ivcast.models.neural_residual import (
    RESIDUAL_MODEL_NAME,
    NeuralResidualSurfaceRegressor,
)
from ivcast.models.neural_surface import NeuralSurfaceRegressor
from ivcast.progress import create_progress
from ivcast.reproducibility import (
    collect_execution_identity,
    sha256_file,
    write_run_manifest,
)
from ivcast.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivcast.splits.manifests import (
    WalkforwardSplit,
    load_split_manifest,
    require_split_manifest_matches_artifacts,
)
from ivcast.splits.walkforward import clean_evaluation_splits
from ivcast.surfaces.grid import SurfaceGrid
from ivcast.training.model_factory import TUNABLE_MODEL_NAMES
from ivcast.training.tuning import (
    load_required_tuning_results,
    require_consistent_clean_evaluation_policy,
    require_matching_primary_loss_metric,
)
from ivcast.workflow import resolve_workflow_run_paths, tuning_manifest_path

app = typer.Typer(add_completion=False)

POSTMORTEM_LOSS_METRICS: tuple[str, ...] = (
    "observed_mse_total_variance",
    "observed_qlike_total_variance",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _indices_for_dates(all_dates: np.ndarray, subset: tuple[str, ...]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(all_dates)}
    return np.asarray([lookup[item] for item in subset], dtype=np.int64)


def _residual_config_from_trial(
    trial: optuna.Trial,
    base_config: NeuralModelConfig,
) -> NeuralModelConfig:
    """Sample the official neural search space for the residual variant."""

    return base_config.model_copy(
        update={
            "hidden_width": trial.suggest_int("hidden_width", 64, 512, step=64),
            "depth": trial.suggest_int("depth", 2, 5),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1.0e-4, 1.0e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-6, 1.0e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "calendar_penalty_weight": trial.suggest_float(
                "calendar_penalty_weight", 1.0e-5, 5.0e-2, log=True
            ),
            "convexity_penalty_weight": trial.suggest_float(
                "convexity_penalty_weight", 1.0e-5, 5.0e-2, log=True
            ),
            "roughness_penalty_weight": trial.suggest_float(
                "roughness_penalty_weight", 1.0e-5, 0.05, log=True
            ),
        }
    )


def _fit_and_predict_torch_variant(
    model: NeuralSurfaceRegressor | NeuralResidualSurfaceRegressor,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    predict_index: np.ndarray,
    matrices: DatasetMatrices,
    training_profile: TrainingProfileConfig,
    *,
    trial: optuna.Trial | None = None,
    trial_step_offset: int = 0,
    validation_metric_name: str,
    validation_positive_floor: float,
) -> np.ndarray:
    model.fit(
        features=matrices.features[train_index],
        targets=matrices.targets[train_index],
        observed_masks=matrices.observed_masks[train_index],
        vega_weights=matrices.vega_weights[train_index],
        training_weights=matrices.training_weights[train_index],
        validation_features=matrices.features[validation_index],
        validation_targets=matrices.targets[validation_index],
        validation_observed_masks=matrices.observed_masks[validation_index],
        validation_vega_weights=matrices.vega_weights[validation_index],
        training_profile=training_profile,
        trial=trial,
        trial_step_offset=trial_step_offset,
        validation_metric_name=validation_metric_name,
        validation_positive_floor=validation_positive_floor,
    )
    return model.predict(matrices.features[predict_index])


def _daily_loss_rows(
    *,
    run_name: str,
    predictions: np.ndarray,
    row_index: np.ndarray,
    matrices: DatasetMatrices,
    positive_floor: float,
) -> pl.DataFrame:
    columns: dict[str, object] = {
        "run_name": np.full(row_index.shape[0], run_name, dtype=object),
        "quote_date": matrices.quote_dates[row_index].astype(str),
        "target_date": matrices.target_dates[row_index].astype(str),
    }
    for metric_name in POSTMORTEM_LOSS_METRICS:
        columns[metric_name] = daily_loss_metric_values(
            metric_name=metric_name,
            y_true=matrices.targets[row_index],
            y_pred=predictions,
            observed_masks=matrices.observed_masks[row_index],
            vega_weights=matrices.vega_weights[row_index],
            positive_floor=positive_floor,
        )
    return pl.DataFrame(columns)


def _collapse_diagnostics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    prediction_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    return {
        "prediction_mean": prediction_mean,
        "target_mean": target_mean,
        "prediction_target_ratio": (
            prediction_mean / target_mean if target_mean > 0.0 else float("nan")
        ),
        "prediction_below_1e_6_share": float(np.mean(predictions < 1.0e-6)),
    }


def _mean_loss(frame: pl.DataFrame, metric_name: str) -> float:
    values = frame[metric_name].to_numpy()
    if not values.size or not np.isfinite(values).all():
        message = f"Post-mortem loss {metric_name!r} must contain finite, nonempty values."
        raise ValueError(message)
    mean = frame[metric_name].mean()
    if isinstance(mean, bool) or not isinstance(mean, int | float):
        message = f"Post-mortem mean loss {metric_name!r} must be numeric."
        raise TypeError(message)
    return float(mean)


def _walkforward_run(
    *,
    run_name: str,
    variant: str,
    seed: int,
    config: NeuralModelConfig,
    matrices: DatasetMatrices,
    clean_splits: list[WalkforwardSplit],
    grid: SurfaceGrid,
    training_profile: TrainingProfileConfig,
    metrics_config: EvaluationMetricsConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    prediction_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    best_epochs: list[int] = []
    for split in clean_splits:
        train_index = _indices_for_dates(matrices.quote_dates, split.train_dates)
        validation_index = _indices_for_dates(matrices.quote_dates, split.validation_dates)
        test_index = _indices_for_dates(matrices.quote_dates, split.test_dates)
        seeded_config = config.model_copy(update={"seed": seed})
        model: NeuralSurfaceRegressor | NeuralResidualSurfaceRegressor
        if variant == "neural_surface":
            model = NeuralSurfaceRegressor(
                config=seeded_config,
                grid_shape=grid.shape,
                moneyness_points=grid.moneyness_points,
            )
        elif variant == RESIDUAL_MODEL_NAME:
            model = NeuralResidualSurfaceRegressor(
                config=seeded_config,
                grid_shape=grid.shape,
                moneyness_points=grid.moneyness_points,
            )
        else:
            message = f"Unsupported post-mortem variant: {variant!r}."
            raise ValueError(message)
        predictions = _fit_and_predict_torch_variant(
            model,
            train_index,
            validation_index,
            test_index,
            matrices,
            training_profile,
            validation_metric_name=metrics_config.primary_loss_metric,
            validation_positive_floor=metrics_config.positive_floor,
        )
        prediction_blocks.append(predictions)
        index_blocks.append(test_index)
        if model.best_epoch is not None:
            best_epochs.append(model.best_epoch)
        if progress_callback is not None:
            progress_callback(split.split_id)
    all_predictions = np.vstack(prediction_blocks)
    all_indices = np.concatenate(index_blocks)
    daily_losses = _daily_loss_rows(
        run_name=run_name,
        predictions=all_predictions,
        row_index=all_indices,
        matrices=matrices,
        positive_floor=metrics_config.positive_floor,
    )
    diagnostics: dict[str, object] = {
        "run_name": run_name,
        "variant": variant,
        "seed": seed,
        "n_splits": len(clean_splits),
        "n_target_dates": int(all_indices.shape[0]),
        "median_best_epoch": (float(np.median(best_epochs)) if best_epochs else None),
        **_collapse_diagnostics(all_predictions, matrices.targets[all_indices]),
        "mean_observed_mse_total_variance": _mean_loss(
            daily_losses, "observed_mse_total_variance"
        ),
        "mean_observed_qlike_total_variance": _mean_loss(
            daily_losses, "observed_qlike_total_variance"
        ),
    }
    return daily_losses, diagnostics


def _summary_markdown(diagnostics_rows: list[dict[str, object]], run_label: str) -> str:
    lines = [
        f"# Neural post-mortem summary — `{run_label}`",
        "",
        "Post-hoc diagnostic. The residual variant predicts the log-innovation",
        "around the lag-1 completed surface with a zero-initialized output head,",
        "so an untrained model reproduces persistence. It is trained and scored",
        "under the saved official protocol but is NOT part of the official",
        "benchmark universe; official artifacts are unchanged.",
        "",
        "| run | variant | seed | mean observed MSE | mean observed QLIKE | "
        "pred/target ratio | share < 1e-6 | median best epoch |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in diagnostics_rows:
        median_best_epoch = row.get("median_best_epoch")
        median_text = "n/a" if median_best_epoch is None else f"{median_best_epoch:.1f}"
        lines.append(
            f"| {row['run_name']} "
            f"| {row['variant']} "
            f"| {row['seed']} "
            f"| {row['mean_observed_mse_total_variance']:.9f} "
            f"| {row['mean_observed_qlike_total_variance']:.6f} "
            f"| {row['prediction_target_ratio']:.6f} "
            f"| {row['prediction_below_1e_6_share']:.6f} "
            f"| {median_text} |"
        )
    lines.append("")
    return "\n".join(lines)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    metrics_config_path: Path = Path("configs/eval/metrics.yaml"),
    neural_config_path: Path = Path("configs/models/neural_surface.yaml"),
    residual_config_path: Path = Path("configs/models/neural_surface_residual.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    run_profile_name: str | None = None,
    seeds: str = "7,17,27",
) -> None:
    started_at = datetime.now(UTC)
    execution_identity = collect_execution_identity(_repo_root())
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    metrics_config = EvaluationMetricsConfig.model_validate(load_yaml_config(metrics_config_path))
    neural_config = NeuralModelConfig.model_validate(load_yaml_config(neural_config_path))
    residual_config = NeuralModelConfig.model_validate(load_yaml_config(residual_config_path))
    if residual_config.model_name != RESIDUAL_MODEL_NAME:
        message = (
            "Residual config model_name must be "
            f"{RESIDUAL_MODEL_NAME!r}, found {residual_config.model_name!r}."
        )
        raise ValueError(message)
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    neural_config = neural_config.model_copy(update={"epochs": training_profile.epochs})
    residual_config = residual_config.model_copy(update={"epochs": training_profile.epochs})
    seed_values = tuple(int(item) for item in seeds.split(",") if item.strip())
    if not seed_values or len(set(seed_values)) != len(seed_values):
        message = f"seeds must be a comma-separated list of unique integers, found {seeds!r}."
        raise ValueError(message)
    grid = SurfaceGrid.from_config(surface_config)
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
        run_profile_name=run_profile_name,
    )
    output_dir = raw_config.manifests_dir / "postmortem" / workflow_paths.run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = raw_config.manifests_dir / "walkforward_splits.json"
    features_path = raw_config.gold_dir / "daily_features.parquet"
    feature_frame = pl.read_parquet(features_path).sort("quote_date")
    feature_dataset_hash = sha256_file(features_path)
    split_manifest = load_split_manifest(split_manifest_path)
    require_split_manifest_matches_artifacts(
        split_manifest,
        date_universe=feature_frame["quote_date"].to_list(),
        feature_dataset_hash=feature_dataset_hash,
    )
    matrices = dataset_to_matrices(feature_frame)
    validate_naive_feature_layout(
        feature_columns=matrices.feature_columns,
        target_columns=matrices.target_columns,
    )

    official_tuning_results = load_required_tuning_results(
        raw_config.manifests_dir,
        hpo_profile_name=hpo_profile.profile_name,
        model_names=TUNABLE_MODEL_NAMES,
    )
    official_tuning_paths = [
        tuning_manifest_path(raw_config.manifests_dir, hpo_profile.profile_name, model_name)
        for model_name in TUNABLE_MODEL_NAMES
    ]
    for model_name, tuning_result in official_tuning_results.items():
        if (
            tuning_result.model_name != model_name
            or tuning_result.hpo_profile_name != hpo_profile.profile_name
            or tuning_result.training_profile_name != training_profile.profile_name
            or tuning_result.tuning_splits_count != hpo_profile.tuning_splits_count
        ):
            message = (
                f"Official tuning manifest for {model_name!r} does not match the selected "
                "model, HPO profile, training profile, or tuning split count."
            )
            raise ValueError(message)
    require_matching_primary_loss_metric(
        official_tuning_results.values(),
        expected_primary_loss_metric=metrics_config.primary_loss_metric,
    )
    clean_evaluation_policy = require_consistent_clean_evaluation_policy(
        official_tuning_results.values()
    )
    boundary, clean_splits = clean_evaluation_splits(
        split_manifest.splits,
        tuning_splits_count=clean_evaluation_policy.tuning_splits_count,
    )
    if (
        boundary.max_hpo_validation_date != clean_evaluation_policy.max_hpo_validation_date
        or boundary.first_clean_test_split_id != clean_evaluation_policy.first_clean_test_split_id
    ):
        message = "Split manifest boundary does not match the official tuning manifests."
        raise ValueError(message)
    tuning_splits = split_manifest.splits[: clean_evaluation_policy.tuning_splits_count]

    baseline_tuned_params = dict(official_tuning_results["neural_surface"].best_params)
    baseline_config = NeuralModelConfig.model_validate(
        {
            **neural_config.model_dump(mode="python"),
            **baseline_tuned_params,
        },
        strict=True,
    )

    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "11_neural_postmortem"),
        stage_name="11_neural_postmortem",
        context_hash=build_resume_context_hash(
            execution_identity=execution_identity,
            config_paths=[
                raw_config_path,
                surface_config_path,
                metrics_config_path,
                neural_config_path,
                residual_config_path,
                hpo_profile_config_path,
                training_profile_config_path,
            ],
            input_artifact_paths=[features_path, split_manifest_path, *official_tuning_paths],
            extra_tokens={
                "artifact_schema_version": 2,
                "run_profile_name": run_profile_name,
                "workflow_run_label": workflow_paths.run_label,
                "seeds": ",".join(str(value) for value in seed_values),
            },
        ),
    )

    residual_tuning_path = output_dir / "residual_tuning_result.json"
    residual_diagnostics_path = output_dir / "residual_tuning_diagnostics.parquet"

    with create_progress() as progress:
        hpo_task = progress.add_task(
            "Post-mortem residual HPO",
            total=hpo_profile.n_trials * len(tuning_splits),
        )
        if resumer.item_complete(
            "residual_hpo",
            required_output_paths=[residual_tuning_path, residual_diagnostics_path],
        ):
            progress.update(hpo_task, description="Post-mortem resume: residual HPO complete")
            progress.advance(hpo_task, advance=hpo_profile.n_trials * len(tuning_splits))
            tuned_payload = orjson.loads(residual_tuning_path.read_bytes())
            residual_best_params = dict(tuned_payload["best_params"])
        else:
            resumer.clear_item(
                "residual_hpo",
                output_paths=[residual_tuning_path, residual_diagnostics_path],
            )
            diagnostic_rows: list[dict[str, object]] = []

            def _objective(trial: optuna.Trial) -> float:
                scores: list[float] = []
                for split_index, split in enumerate(tuning_splits):
                    train_index = _indices_for_dates(matrices.quote_dates, split.train_dates)
                    validation_index = _indices_for_dates(
                        matrices.quote_dates, split.validation_dates
                    )
                    trial_config = _residual_config_from_trial(trial, residual_config)
                    model = NeuralResidualSurfaceRegressor(
                        config=trial_config,
                        grid_shape=grid.shape,
                        moneyness_points=grid.moneyness_points,
                    )
                    predictions = _fit_and_predict_torch_variant(
                        model,
                        train_index,
                        validation_index,
                        validation_index,
                        matrices,
                        training_profile,
                        trial=trial,
                        trial_step_offset=split_index * (training_profile.epochs + 1),
                        validation_metric_name=metrics_config.primary_loss_metric,
                        validation_positive_floor=metrics_config.positive_floor,
                    )
                    score_values = daily_loss_metric_values(
                        metric_name=metrics_config.primary_loss_metric,
                        y_true=matrices.targets[validation_index],
                        y_pred=predictions,
                        observed_masks=matrices.observed_masks[validation_index],
                        vega_weights=matrices.vega_weights[validation_index],
                        positive_floor=metrics_config.positive_floor,
                    )
                    selected_metric_value = float(np.mean(score_values))
                    scores.append(selected_metric_value)
                    diagnostics = model.validation_diagnostics
                    diagnostic_rows.append(
                        {
                            "model_name": RESIDUAL_MODEL_NAME,
                            "trial_number": trial.number,
                            "split_index": split_index,
                            "split_id": split.split_id,
                            "selected_metric_name": metrics_config.primary_loss_metric,
                            "selected_metric_value": selected_metric_value,
                            "neural_best_epoch": model.best_epoch,
                            "neural_prediction_target_ratio": (
                                None
                                if diagnostics is None
                                else diagnostics.prediction_target_ratio
                            ),
                            "neural_prediction_below_1e_6_share": (
                                None
                                if diagnostics is None
                                else diagnostics.prediction_below_1e_6_share
                            ),
                        }
                    )
                    split_report_step = (
                        split_index * (training_profile.epochs + 1) + training_profile.epochs
                    )
                    trial.report(float(np.mean(scores)), split_report_step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    progress.update(
                        hpo_task,
                        description=(
                            f"Post-mortem residual HPO: trial {trial.number + 1} "
                            f"split {split.split_id}"
                        ),
                    )
                    progress.advance(hpo_task)
                return float(np.mean(scores))

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=hpo_profile.seed),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=hpo_profile.pruner.n_startup_trials,
                    n_warmup_steps=hpo_profile.pruner.n_warmup_steps,
                    interval_steps=hpo_profile.pruner.interval_steps,
                ),
            )
            study.optimize(_objective, n_trials=hpo_profile.n_trials)
            completed_trials = study.get_trials(
                deepcopy=False,
                states=(optuna.trial.TrialState.COMPLETE,),
            )
            if not completed_trials:
                message = "Residual HPO completed no trials."
                raise RuntimeError(message)
            residual_best_params = dict(study.best_params)
            tuning_payload = {
                "model_name": RESIDUAL_MODEL_NAME,
                "hpo_profile_name": hpo_profile.profile_name,
                "training_profile_name": training_profile.profile_name,
                "primary_loss_metric": metrics_config.primary_loss_metric,
                "best_value": float(study.best_value),
                "best_params": residual_best_params,
                "n_trials_requested": hpo_profile.n_trials,
                "n_trials_completed": len(completed_trials),
                "tuning_splits_count": clean_evaluation_policy.tuning_splits_count,
                "max_hpo_validation_date": (
                    clean_evaluation_policy.max_hpo_validation_date.isoformat()
                ),
                "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
                "seed": hpo_profile.seed,
                "sampler": type(study.sampler).__name__,
                "pruner": type(study.pruner).__name__,
                "isolation_note": (
                    "Post-hoc diagnostic manifest; intentionally stored outside "
                    "data/manifests/tuning so official stages never consume it."
                ),
            }
            write_bytes_atomic(
                residual_tuning_path,
                orjson.dumps(tuning_payload, option=orjson.OPT_INDENT_2),
            )
            write_parquet_frame(
                pl.DataFrame(diagnostic_rows).sort(["trial_number", "split_index"]),
                residual_diagnostics_path,
            )
            resumer.mark_complete(
                "residual_hpo",
                output_paths=[residual_tuning_path, residual_diagnostics_path],
                metadata={"n_trials_completed": len(completed_trials)},
            )

        tuned_residual_config = NeuralModelConfig.model_validate(
            {
                **residual_config.model_dump(mode="python"),
                **residual_best_params,
            },
            strict=True,
        )

        run_plan: list[tuple[str, str, int, NeuralModelConfig]] = []
        for seed in seed_values:
            run_plan.append(
                (
                    f"neural_surface_seed{seed}",
                    "neural_surface",
                    seed,
                    baseline_config,
                )
            )
        for seed in seed_values:
            run_plan.append(
                (
                    f"{RESIDUAL_MODEL_NAME}_seed{seed}",
                    RESIDUAL_MODEL_NAME,
                    seed,
                    tuned_residual_config,
                )
            )

        walk_task = progress.add_task(
            "Post-mortem walk-forward runs",
            total=len(run_plan) * len(clean_splits) + 1,
        )
        diagnostics_rows: list[dict[str, object]] = []
        daily_loss_frames: list[pl.DataFrame] = []

        naive_daily_path = output_dir / "daily_losses_naive_reference.parquet"
        naive_diag_path = output_dir / "diagnostics_naive_reference.json"
        if resumer.item_complete(
            "naive_reference",
            required_output_paths=[naive_daily_path, naive_diag_path],
        ):
            daily_loss_frames.append(pl.read_parquet(naive_daily_path))
            diagnostics_rows.append(dict(orjson.loads(naive_diag_path.read_bytes())))
        else:
            resumer.clear_item(
                "naive_reference",
                output_paths=[naive_daily_path, naive_diag_path],
            )
            progress.update(walk_task, description="Post-mortem naive reference losses")
            test_indices = np.concatenate(
                [
                    _indices_for_dates(matrices.quote_dates, split.test_dates)
                    for split in clean_splits
                ]
            )
            naive_predictions = matrices.features[
                test_indices, : matrices.targets.shape[1]
            ].astype(np.float64)
            naive_daily = _daily_loss_rows(
                run_name="naive_reference",
                predictions=naive_predictions,
                row_index=test_indices,
                matrices=matrices,
                positive_floor=metrics_config.positive_floor,
            )
            naive_diagnostics: dict[str, object] = {
                "run_name": "naive_reference",
                "variant": "naive",
                "seed": None,
                "n_splits": len(clean_splits),
                "n_target_dates": int(test_indices.shape[0]),
                "median_best_epoch": None,
                **_collapse_diagnostics(
                    naive_predictions,
                    matrices.targets[test_indices],
                ),
                "mean_observed_mse_total_variance": _mean_loss(
                    naive_daily, "observed_mse_total_variance"
                ),
                "mean_observed_qlike_total_variance": _mean_loss(
                    naive_daily, "observed_qlike_total_variance"
                ),
            }
            write_parquet_frame(naive_daily, naive_daily_path)
            write_bytes_atomic(
                naive_diag_path,
                orjson.dumps(naive_diagnostics, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                "naive_reference",
                output_paths=[naive_daily_path, naive_diag_path],
                metadata={"n_target_dates": int(test_indices.shape[0])},
            )
            daily_loss_frames.append(naive_daily)
            diagnostics_rows.append(naive_diagnostics)
        progress.advance(walk_task)

        for run_name, variant, seed, run_config in run_plan:
            run_daily_path = output_dir / f"daily_losses_{run_name}.parquet"
            run_diag_path = output_dir / f"diagnostics_{run_name}.json"
            if resumer.item_complete(
                run_name,
                required_output_paths=[run_daily_path, run_diag_path],
            ):
                progress.update(
                    walk_task,
                    description=f"Post-mortem resume: {run_name} complete",
                )
                progress.advance(walk_task, advance=len(clean_splits))
                daily_loss_frames.append(pl.read_parquet(run_daily_path))
                diagnostics_rows.append(dict(orjson.loads(run_diag_path.read_bytes())))
                continue
            resumer.clear_item(run_name, output_paths=[run_daily_path, run_diag_path])

            def _on_split(split_id: str, run_name: str = run_name) -> None:
                progress.update(
                    walk_task,
                    description=f"Post-mortem {run_name}: {split_id}",
                )
                progress.advance(walk_task)

            run_daily, run_diagnostics = _walkforward_run(
                run_name=run_name,
                variant=variant,
                seed=seed,
                config=run_config,
                matrices=matrices,
                clean_splits=clean_splits,
                grid=grid,
                training_profile=training_profile,
                metrics_config=metrics_config,
                progress_callback=_on_split,
            )
            write_parquet_frame(run_daily, run_daily_path)
            write_bytes_atomic(
                run_diag_path,
                orjson.dumps(run_diagnostics, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                run_name,
                output_paths=[run_daily_path, run_diag_path],
                metadata={"variant": variant, "seed": seed},
            )
            daily_loss_frames.append(run_daily)
            diagnostics_rows.append(run_diagnostics)

    combined_daily = pl.concat(daily_loss_frames).sort(["run_name", "target_date"])
    combined_daily_path = output_dir / "postmortem_daily_losses.parquet"
    write_parquet_frame(combined_daily, combined_daily_path)
    summary_markdown_path = output_dir / "postmortem_summary.md"
    write_bytes_atomic(
        summary_markdown_path,
        _summary_markdown(diagnostics_rows, workflow_paths.run_label).encode("utf-8"),
    )
    summary_json_path = output_dir / "postmortem_summary.json"
    write_bytes_atomic(
        summary_json_path,
        orjson.dumps(diagnostics_rows, option=orjson.OPT_INDENT_2),
    )

    output_paths = [
        residual_tuning_path,
        residual_diagnostics_path,
        combined_daily_path,
        summary_markdown_path,
        summary_json_path,
    ]
    run_manifest_path = write_run_manifest(
        execution_identity=execution_identity,
        manifests_dir=raw_config.manifests_dir,
        repo_root=_repo_root(),
        script_name="11_neural_postmortem",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            metrics_config_path,
            neural_config_path,
            residual_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[features_path, split_manifest_path, *official_tuning_paths],
        output_artifact_paths=output_paths,
        data_manifest_paths=[features_path, *official_tuning_paths],
        split_manifest_path=split_manifest_path,
        random_seed=hpo_profile.seed,
        extra_metadata={
            "seeds": list(seed_values),
            "variants": ["neural_surface", RESIDUAL_MODEL_NAME],
            "n_clean_splits": len(clean_splits),
            "max_hpo_validation_date": (
                clean_evaluation_policy.max_hpo_validation_date.isoformat()
            ),
            "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
            "run_profile_name": run_profile_name,
            "workflow_run_label": workflow_paths.run_label,
            "official_universe_unchanged": True,
            "resume_context_hash": resumer.context_hash,
        },
    )
    typer.echo(f"Saved post-mortem outputs to {output_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
