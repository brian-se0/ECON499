from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.config import (
    EvaluationMetricsConfig,
    HpoProfileConfig,
    RawDataConfig,
    StatsTestConfig,
    SurfaceGridConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.evaluation.alignment import (
    assert_forecast_origins_after_hpo_boundary,
    build_forecast_realization_panel,
    load_actual_surface_frame,
    load_forecast_frame,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.stats.diebold_mariano import diebold_mariano_test
from ivsurf.stats.mcs import model_confidence_set
from ivsurf.stats.spa import superior_predictive_ability_test
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.training.model_factory import TUNABLE_MODEL_NAMES
from ivsurf.training.tuning import (
    load_required_tuning_results,
    require_consistent_clean_evaluation_policy,
    require_matching_primary_loss_metric,
)
from ivsurf.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


def _daily_loss_matrix(
    loss_frame: pl.DataFrame,
    metric_column: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[object, ...]]:
    if metric_column not in loss_frame.columns:
        message = f"Daily loss frame is missing configured metric column {metric_column!r}."
        raise ValueError(message)
    duplicate_keys = (
        loss_frame.group_by(["model_name", "target_date"])
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
    )
    if not duplicate_keys.is_empty():
        message = (
            "Daily loss frame contains duplicate model_name/target_date rows; "
            f"duplicate key count={duplicate_keys.height}."
        )
        raise ValueError(message)
    pivoted = (
        loss_frame.select("model_name", "target_date", metric_column)
        .pivot(on="model_name", index="target_date", values=metric_column)
        .sort("target_date")
    )
    model_columns = tuple(column for column in pivoted.columns if column != "target_date")
    if not model_columns:
        message = "Daily loss matrix must contain at least one model column."
        raise ValueError(message)
    if any(pivoted[column].null_count() > 0 for column in model_columns):
        message = (
            "Daily loss matrix contains missing model/date coverage after pivoting "
            f"metric {metric_column!r}."
        )
        raise ValueError(message)
    matrix = pivoted.select(model_columns).to_numpy().astype(np.float64)
    if not np.isfinite(matrix).all():
        message = f"Daily loss matrix for metric {metric_column!r} contains non-finite values."
        raise ValueError(message)
    return matrix, model_columns, tuple(pivoted["target_date"].to_list())


def _require_finite_loss_metrics(
    daily_loss_frame: pl.DataFrame,
    *,
    loss_metrics: tuple[str, ...],
) -> None:
    for metric_column in loss_metrics:
        _daily_loss_matrix(daily_loss_frame, metric_column)


def _require_matching_loss_matrix_contract(
    *,
    metric_column: str,
    metric_model_columns: tuple[str, ...],
    metric_target_dates: tuple[object, ...],
    reference_model_columns: tuple[str, ...],
    reference_target_dates: tuple[object, ...],
) -> None:
    if metric_model_columns != reference_model_columns:
        message = (
            "All loss metrics must produce the same model ordering in the daily loss frame, "
            f"found {metric_model_columns!r} != {reference_model_columns!r} "
            f"for {metric_column}."
        )
        raise ValueError(message)
    if metric_target_dates != reference_target_dates:
        message = (
            "All loss metrics must produce the same target-date coverage in the daily loss "
            f"frame for {metric_column}."
        )
        raise ValueError(message)


def _loss_summary_frame(
    daily_loss_frame: pl.DataFrame,
    *,
    loss_metrics: tuple[str, ...],
) -> pl.DataFrame:
    aggregation_expressions: list[pl.Expr] = []
    for metric_column in loss_metrics:
        aggregation_expressions.extend(
            (
                pl.col(metric_column).mean().alias(f"mean_{metric_column}"),
                pl.col(metric_column).std(ddof=1).alias(f"std_{metric_column}"),
            )
        )
    return (
        daily_loss_frame.group_by("model_name")
        .agg(*aggregation_expressions, pl.len().alias("n_target_dates"))
        .sort(f"mean_{loss_metrics[0]}")
    )


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    metrics_config_path: Path = Path("configs/eval/metrics.yaml"),
    stats_config_path: Path = Path("configs/eval/stats_tests.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    run_profile_name: str | None = None,
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
    metrics_config = EvaluationMetricsConfig.model_validate(load_yaml_config(metrics_config_path))
    stats_config = StatsTestConfig.model_validate(load_yaml_config(stats_config_path))
    grid = SurfaceGrid.from_config(surface_config)
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
        run_profile_name=run_profile_name,
    )

    output_dir = workflow_paths.stats_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_paths = sorted_artifact_files(workflow_paths.forecast_dir, "*.parquet")
    forecast_reuse_manifest_paths = []
    if run_profile_name is not None:
        forecast_reuse_manifest_path = (
            raw_config.manifests_dir / "forecast_profile_reuse" / f"{run_profile_name}.json"
        )
        if forecast_reuse_manifest_path.exists():
            forecast_reuse_manifest_paths.append(forecast_reuse_manifest_path)
    tuning_manifest_paths = [
        raw_config.manifests_dir / "tuning" / hpo_profile.profile_name / f"{model_name}.json"
        for model_name in TUNABLE_MODEL_NAMES
    ]
    panel_path = output_dir / "forecast_realization_panel.parquet"
    daily_loss_path = output_dir / "daily_loss_frame.parquet"
    dm_results_path = output_dir / "dm_results.json"
    spa_result_path = output_dir / "spa_result.json"
    mcs_result_path = output_dir / "mcs_result.json"
    loss_summary_path = output_dir / "loss_summary.parquet"
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "07_run_stats"),
        stage_name="07_run_stats",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                surface_config_path,
                metrics_config_path,
                stats_config_path,
                hpo_profile_config_path,
                training_profile_config_path,
            ],
            input_artifact_paths=[
                raw_config.manifests_dir / "gold_surface_summary.json",
                *forecast_reuse_manifest_paths,
                *tuning_manifest_paths,
                *forecast_paths,
            ],
            extra_tokens={
                "run_profile_name": run_profile_name,
                "workflow_run_label": workflow_paths.run_label,
                "artifact_schema_version": 5,
            },
        ),
    )
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

    with create_progress() as progress:
        task_id = progress.add_task("Stage 07 statistical evaluation", total=None)

        if resumer.item_complete("forecast_realization_panel", required_output_paths=[panel_path]):
            progress.update(task_id, description="Stage 07 resume: loading saved panel")
            panel = pl.read_parquet(panel_path)
        else:
            resumer.clear_item("forecast_realization_panel", output_paths=[panel_path])
            progress.update(task_id, description="Stage 07 loading realized surfaces and forecasts")
            actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir, grid)
            forecast_frame = load_forecast_frame(workflow_paths.forecast_dir, grid)
            assert_forecast_origins_after_hpo_boundary(
                forecast_frame,
                max_hpo_validation_date=clean_evaluation_policy.max_hpo_validation_date,
            )
            panel = build_forecast_realization_panel(
                actual_surface_frame=actual_surface_frame,
                forecast_frame=forecast_frame,
                total_variance_floor=metrics_config.positive_floor,
            )
            write_parquet_frame(panel, panel_path)
            resumer.mark_complete(
                "forecast_realization_panel",
                output_paths=[panel_path],
                metadata={"rows": panel.height},
            )

        if resumer.item_complete("daily_loss_frame", required_output_paths=[daily_loss_path]):
            progress.update(task_id, description="Stage 07 resume: loading saved daily loss frame")
            daily_loss_frame = pl.read_parquet(daily_loss_path)
        else:
            resumer.clear_item("daily_loss_frame", output_paths=[daily_loss_path])
            progress.update(task_id, description="Stage 07 computing daily loss panel")
            daily_loss_frame = build_daily_loss_frame(
                panel=panel,
                positive_floor=metrics_config.positive_floor,
                full_grid_weighting=stats_config.full_grid_weighting,
            )
            write_parquet_frame(daily_loss_frame, daily_loss_path)
            resumer.mark_complete(
                "daily_loss_frame",
                output_paths=[daily_loss_path],
                metadata={"rows": daily_loss_frame.height},
            )

        _require_finite_loss_metrics(
            daily_loss_frame,
            loss_metrics=stats_config.loss_metrics,
        )
        primary_metric_column = stats_config.loss_metrics[0]
        _, model_columns, target_dates = _daily_loss_matrix(
            loss_frame=daily_loss_frame,
            metric_column=primary_metric_column,
        )
        benchmark_model = stats_config.benchmark_model
        if benchmark_model not in model_columns:
            message = f"Benchmark model {benchmark_model} not found in loss frame."
            raise ValueError(message)

        dm_model_count = max(len(model_columns) - 1, 0) * len(stats_config.loss_metrics)
        progress.update(
            task_id,
            total=4 + dm_model_count + (2 * len(stats_config.loss_metrics)),
            completed=2,
            description="Stage 07 running Diebold-Mariano tests",
        )
        if not resumer.item_complete("dm_results", required_output_paths=[dm_results_path]):
            resumer.clear_item("dm_results", output_paths=[dm_results_path])
            dm_results: list[dict[str, object]] = []
            for metric_column in stats_config.loss_metrics:
                loss_matrix, metric_model_columns, metric_target_dates = _daily_loss_matrix(
                    loss_frame=daily_loss_frame,
                    metric_column=metric_column,
                )
                _require_matching_loss_matrix_contract(
                    metric_column=metric_column,
                    metric_model_columns=metric_model_columns,
                    metric_target_dates=metric_target_dates,
                    reference_model_columns=model_columns,
                    reference_target_dates=target_dates,
                )
                benchmark_index = metric_model_columns.index(benchmark_model)
                benchmark_losses = loss_matrix[:, benchmark_index]
                for model_name in metric_model_columns:
                    if model_name == benchmark_model:
                        continue
                    model_index = metric_model_columns.index(model_name)
                    result_row = asdict(
                        diebold_mariano_test(
                            loss_a=benchmark_losses,
                            loss_b=loss_matrix[:, model_index],
                            model_a=benchmark_model,
                            model_b=model_name,
                            alternative=stats_config.dm_alternative,
                            max_lag=stats_config.dm_max_lag,
                        )
                    )
                    result_row["loss_metric"] = metric_column
                    dm_results.append(result_row)
                    progress.advance(task_id)
            write_bytes_atomic(
                dm_results_path,
                orjson.dumps(dm_results, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                "dm_results",
                output_paths=[dm_results_path],
                metadata={
                    "model_count": len(model_columns),
                    "loss_metrics": list(stats_config.loss_metrics),
                },
            )
        else:
            progress.advance(task_id, advance=dm_model_count)

        progress.update(task_id, description="Stage 07 running SPA bootstrap")
        if not resumer.item_complete("spa_result", required_output_paths=[spa_result_path]):
            resumer.clear_item("spa_result", output_paths=[spa_result_path])
            spa_results: list[dict[str, object]] = []
            for metric_column in stats_config.loss_metrics:
                loss_matrix, metric_model_columns, metric_target_dates = _daily_loss_matrix(
                    loss_frame=daily_loss_frame,
                    metric_column=metric_column,
                )
                _require_matching_loss_matrix_contract(
                    metric_column=metric_column,
                    metric_model_columns=metric_model_columns,
                    metric_target_dates=metric_target_dates,
                    reference_model_columns=model_columns,
                    reference_target_dates=target_dates,
                )
                candidate_models = tuple(
                    model for model in metric_model_columns if model != benchmark_model
                )
                candidate_losses = np.column_stack(
                    [
                        loss_matrix[:, metric_model_columns.index(model)]
                        for model in candidate_models
                    ]
                )
                benchmark_index = metric_model_columns.index(benchmark_model)
                benchmark_losses = loss_matrix[:, benchmark_index]
                spa_result = asdict(
                    superior_predictive_ability_test(
                        benchmark_losses=benchmark_losses,
                        candidate_losses=candidate_losses,
                        benchmark_model=benchmark_model,
                        candidate_models=candidate_models,
                        alpha=stats_config.spa_alpha,
                        block_size=stats_config.spa_block_size,
                        bootstrap_reps=stats_config.spa_bootstrap_reps,
                        seed=stats_config.bootstrap_seed,
                    )
                )
                spa_result["loss_metric"] = metric_column
                spa_results.append(spa_result)
            write_bytes_atomic(
                spa_result_path,
                orjson.dumps(spa_results, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                "spa_result",
                output_paths=[spa_result_path],
                metadata={"loss_metrics": list(stats_config.loss_metrics)},
            )
        progress.advance(task_id, advance=len(stats_config.loss_metrics))

        progress.update(task_id, description="Stage 07 running simplified Tmax bootstrap")
        if not resumer.item_complete("mcs_result", required_output_paths=[mcs_result_path]):
            resumer.clear_item("mcs_result", output_paths=[mcs_result_path])
            mcs_results: list[dict[str, object]] = []
            for metric_column in stats_config.loss_metrics:
                loss_matrix, metric_model_columns, metric_target_dates = _daily_loss_matrix(
                    loss_frame=daily_loss_frame,
                    metric_column=metric_column,
                )
                _require_matching_loss_matrix_contract(
                    metric_column=metric_column,
                    metric_model_columns=metric_model_columns,
                    metric_target_dates=metric_target_dates,
                    reference_model_columns=model_columns,
                    reference_target_dates=target_dates,
                )
                mcs_result = asdict(
                    model_confidence_set(
                        losses=loss_matrix,
                        model_names=metric_model_columns,
                        alpha=stats_config.mcs_alpha,
                        block_size=stats_config.mcs_block_size,
                        bootstrap_reps=stats_config.mcs_bootstrap_reps,
                        seed=stats_config.bootstrap_seed,
                    )
                )
                mcs_result["loss_metric"] = metric_column
                mcs_results.append(mcs_result)
            write_bytes_atomic(
                mcs_result_path,
                orjson.dumps(mcs_results, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                "mcs_result",
                output_paths=[mcs_result_path],
                metadata={
                    "model_names": list(model_columns),
                    "loss_metrics": list(stats_config.loss_metrics),
                    "procedure_name": "simplified_tmax_elimination",
                },
            )
        progress.advance(task_id, advance=len(stats_config.loss_metrics))

        progress.update(task_id, description="Stage 07 writing summary tables")
        if not resumer.item_complete("loss_summary", required_output_paths=[loss_summary_path]):
            resumer.clear_item("loss_summary", output_paths=[loss_summary_path])
            summary_frame = _loss_summary_frame(
                daily_loss_frame,
                loss_metrics=stats_config.loss_metrics,
            )
            write_parquet_frame(summary_frame, loss_summary_path)
            resumer.mark_complete(
                "loss_summary",
                output_paths=[loss_summary_path],
                metadata={
                    "model_count": len(model_columns),
                    "loss_metrics": list(stats_config.loss_metrics),
                },
            )
        progress.advance(task_id)

    output_paths = [
        panel_path,
        daily_loss_path,
        dm_results_path,
        spa_result_path,
        mcs_result_path,
        loss_summary_path,
    ]
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="07_run_stats",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            metrics_config_path,
            stats_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            *forecast_reuse_manifest_paths,
            *tuning_manifest_paths,
            *forecast_paths,
        ],
        output_artifact_paths=output_paths,
        data_manifest_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            *forecast_reuse_manifest_paths,
            *tuning_manifest_paths,
            *forecast_paths,
        ],
        random_seed=stats_config.bootstrap_seed,
        extra_metadata={
            "loss_metrics": list(stats_config.loss_metrics),
            "benchmark_model": benchmark_model,
            "mcs_procedure_name": "simplified_tmax_elimination",
            "max_hpo_validation_date": clean_evaluation_policy.max_hpo_validation_date.isoformat(),
            "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
            "run_profile_name": run_profile_name,
            "workflow_run_label": workflow_paths.run_label,
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved stats outputs to {output_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
