from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.config import HpoProfileConfig, RawDataConfig, TrainingProfileConfig, load_yaml_config
from ivsurf.evaluation.alignment import (
    build_forecast_realization_panel,
    load_actual_surface_frame,
    load_forecast_frame,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.progress import create_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.stats.diebold_mariano import diebold_mariano_test
from ivsurf.stats.mcs import model_confidence_set
from ivsurf.stats.spa import superior_predictive_ability_test
from ivsurf.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


def _daily_loss_matrix(
    loss_frame: pl.DataFrame,
    metric_column: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[object, ...]]:
    pivoted = (
        loss_frame.select("model_name", "target_date", metric_column)
        .pivot(on="model_name", index="target_date", values=metric_column)
        .sort("target_date")
    )
    model_columns = tuple(column for column in pivoted.columns if column != "target_date")
    matrix = pivoted.select(model_columns).to_numpy().astype(np.float64)
    return matrix, model_columns, tuple(pivoted["target_date"].to_list())


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    metrics_config_path: Path = Path("configs/eval/metrics.yaml"),
    stats_config_path: Path = Path("configs/eval/stats_tests.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    metrics_config = load_yaml_config(metrics_config_path)
    stats_config = load_yaml_config(stats_config_path)
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
    )

    output_dir = workflow_paths.stats_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dm_results: list[dict[str, object]] = []
    with create_progress() as progress:
        task_id = progress.add_task("Stage 07 statistical evaluation", total=None)

        progress.update(task_id, description="Stage 07 loading realized surfaces and forecasts")
        actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir)
        forecast_frame = load_forecast_frame(workflow_paths.forecast_dir)
        panel = build_forecast_realization_panel(
            actual_surface_frame=actual_surface_frame,
            forecast_frame=forecast_frame,
        )
        panel.write_parquet(output_dir / "forecast_realization_panel.parquet", compression="zstd")

        progress.update(task_id, description="Stage 07 computing daily loss panel")
        daily_loss_frame = build_daily_loss_frame(
            panel=panel,
            positive_floor=float(metrics_config["positive_floor"]),
            full_grid_weighting=str(stats_config["full_grid_weighting"]),
        )
        daily_loss_frame.write_parquet(output_dir / "daily_loss_frame.parquet", compression="zstd")

        metric_column = str(stats_config["loss_metric"])
        loss_matrix, model_columns, _target_dates = _daily_loss_matrix(
            loss_frame=daily_loss_frame,
            metric_column=metric_column,
        )
        benchmark_model = str(stats_config["benchmark_model"])
        if benchmark_model not in model_columns:
            message = f"Benchmark model {benchmark_model} not found in loss frame."
            raise ValueError(message)
        benchmark_index = model_columns.index(benchmark_model)
        benchmark_losses = loss_matrix[:, benchmark_index]

        dm_model_count = max(len(model_columns) - 1, 0)
        progress.update(
            task_id,
            total=4 + dm_model_count,
            completed=2,
            description="Stage 07 running Diebold-Mariano tests",
        )
        for model_name in model_columns:
            if model_name == benchmark_model:
                continue
            model_index = model_columns.index(model_name)
            result = diebold_mariano_test(
                loss_a=benchmark_losses,
                loss_b=loss_matrix[:, model_index],
                model_a=benchmark_model,
                model_b=model_name,
                alternative=str(stats_config["dm_alternative"]),
                max_lag=int(stats_config["dm_max_lag"]),
            )
            dm_results.append(asdict(result))
            progress.advance(task_id)
        (output_dir / "dm_results.json").write_bytes(
            orjson.dumps(dm_results, option=orjson.OPT_INDENT_2)
        )

        candidate_models = tuple(model for model in model_columns if model != benchmark_model)
        candidate_losses = np.column_stack(
            [loss_matrix[:, model_columns.index(model)] for model in candidate_models]
        )
        progress.update(task_id, description="Stage 07 running SPA bootstrap")
        spa_result = superior_predictive_ability_test(
            benchmark_losses=benchmark_losses,
            candidate_losses=candidate_losses,
            benchmark_model=benchmark_model,
            candidate_models=candidate_models,
            block_size=int(stats_config["spa_block_size"]),
            bootstrap_reps=int(stats_config["spa_bootstrap_reps"]),
            seed=int(stats_config["bootstrap_seed"]),
        )
        (output_dir / "spa_result.json").write_bytes(
            orjson.dumps(asdict(spa_result), option=orjson.OPT_INDENT_2)
        )
        progress.advance(task_id)

        progress.update(task_id, description="Stage 07 running MCS bootstrap")
        mcs_result = model_confidence_set(
            losses=loss_matrix,
            model_names=model_columns,
            alpha=float(stats_config["mcs_alpha"]),
            block_size=int(stats_config["mcs_block_size"]),
            bootstrap_reps=int(stats_config["mcs_bootstrap_reps"]),
            seed=int(stats_config["bootstrap_seed"]),
        )
        (output_dir / "mcs_result.json").write_bytes(
            orjson.dumps(asdict(mcs_result), option=orjson.OPT_INDENT_2)
        )
        progress.advance(task_id)

        progress.update(task_id, description="Stage 07 writing summary tables")
        summary_frame = (
            daily_loss_frame.group_by("model_name")
            .agg(
                pl.col(metric_column).mean().alias(f"mean_{metric_column}"),
                pl.col(metric_column).std(ddof=1).alias(f"std_{metric_column}"),
                pl.len().alias("n_target_dates"),
            )
            .sort(f"mean_{metric_column}")
        )
        summary_frame.write_parquet(output_dir / "loss_summary.parquet", compression="zstd")
        progress.advance(task_id)

    forecast_paths = sorted(workflow_paths.forecast_dir.glob("*.parquet"))
    output_paths = [
        output_dir / "forecast_realization_panel.parquet",
        output_dir / "daily_loss_frame.parquet",
        output_dir / "dm_results.json",
        output_dir / "spa_result.json",
        output_dir / "mcs_result.json",
        output_dir / "loss_summary.parquet",
    ]
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="07_run_stats",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            metrics_config_path,
            stats_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            *forecast_paths,
        ],
        output_artifact_paths=output_paths,
        data_manifest_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            *forecast_paths,
        ],
        random_seed=int(stats_config["bootstrap_seed"]),
        extra_metadata={
            "loss_metric": metric_column,
            "benchmark_model": benchmark_model,
            "workflow_run_label": workflow_paths.run_label,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved stats outputs to {output_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
