"""Post-hoc statistical-inference sensitivity over saved Stage-07 artifacts.

Reads the frozen official stats outputs (daily loss frame and aligned
forecast-realization panel) and reruns the comparison tests at higher
bootstrap depth, alternative block lengths, and alternative Diebold-Mariano
lags, plus weighting-robustness model rankings. All outputs land under
``data/manifests/stats_sensitivity/<run_label>/``; the official Stage-07
directory is read-only for this stage.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivcast.config import (
    HpoProfileConfig,
    RawDataConfig,
    StatsSensitivityConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivcast.evaluation.loss_matrix import daily_loss_matrix
from ivcast.evaluation.robust_aggregation import (
    build_weighting_robustness_daily_losses,
    summarize_weighting_robustness,
)
from ivcast.io.atomic import write_bytes_atomic
from ivcast.io.parquet import write_parquet_frame
from ivcast.progress import create_progress
from ivcast.reproducibility import (
    collect_execution_identity,
    write_run_manifest,
)
from ivcast.stats.diebold_mariano import diebold_mariano_test
from ivcast.stats.mcs import model_confidence_set
from ivcast.stats.spa import superior_predictive_ability_test
from ivcast.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_official_input(path: Path) -> Path:
    if not path.exists():
        message = (
            f"Required official Stage-07 artifact is missing: {path}. "
            "Run the official pipeline before the sensitivity stage."
        )
        raise FileNotFoundError(message)
    return path


def _robust_summary_markdown(summary: pl.DataFrame, run_label: str) -> str:
    lines = [
        f"# Weighting-robustness model ranking — `{run_label}`",
        "",
        "Observed-cell MSE in total variance under three weighting schemes.",
        "Official = target-day vega weights; maturity-balanced = vega weights",
        "within each observed maturity slice, equal weight across slices;",
        "equal-cell = unweighted across observed cells.",
        "",
        "| model | official mean | official rank | maturity-balanced mean | "
        "maturity-balanced rank | equal-cell mean | equal-cell rank |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['model_name']} "
            f"| {row['mean_official_vega_weighted_mse']:.9f} "
            f"| {row['rank_official']} "
            f"| {row['mean_maturity_balanced_observed_mse_total_variance']:.9f} "
            f"| {row['rank_maturity_balanced']} "
            f"| {row['mean_equal_cell_observed_mse_total_variance']:.9f} "
            f"| {row['rank_equal_cell']} |"
        )
    lines.append("")
    return "\n".join(lines)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    sensitivity_config_path: Path = Path("configs/eval/stats_sensitivity.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    run_profile_name: str | None = None,
) -> None:
    started_at = datetime.now(UTC)
    execution_identity = collect_execution_identity(_repo_root())
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    sensitivity_config = StatsSensitivityConfig.model_validate(
        load_yaml_config(sensitivity_config_path)
    )
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
        run_profile_name=run_profile_name,
    )

    daily_loss_path = _require_official_input(
        workflow_paths.stats_dir / "daily_loss_frame.parquet"
    )
    panel_path = _require_official_input(
        workflow_paths.stats_dir / "forecast_realization_panel.parquet"
    )
    output_dir = raw_config.manifests_dir / "stats_sensitivity" / workflow_paths.run_label
    if output_dir.resolve() == workflow_paths.stats_dir.resolve():
        message = "Sensitivity output directory must not be the official stats directory."
        raise ValueError(message)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_loss_frame = pl.read_parquet(daily_loss_path)
    benchmark_model = sensitivity_config.benchmark_model

    dm_results_path = output_dir / "dm_lag_sensitivity.json"
    spa_results_path = output_dir / "spa_sensitivity.json"
    mcs_results_path = output_dir / "mcs_sensitivity.json"
    robust_daily_path = output_dir / "weighting_robustness_daily_losses.parquet"
    robust_summary_path = output_dir / "weighting_robustness_summary.parquet"
    robust_summary_markdown_path = output_dir / "weighting_robustness_summary.md"

    metric_count = len(sensitivity_config.loss_metrics)
    total_steps = (
        metric_count * len(sensitivity_config.dm_max_lags)
        + (2 * metric_count * len(sensitivity_config.block_sizes))
        + 2
    )
    with create_progress() as progress:
        task_id = progress.add_task("Stage 10 stats sensitivity", total=total_steps)

        loss_matrices: dict[str, tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]] = {}
        reference_models: tuple[str, ...] | None = None
        for metric_column in sensitivity_config.loss_metrics:
            matrix, model_columns, target_dates = daily_loss_matrix(
                daily_loss_frame,
                metric_column,
            )
            if benchmark_model not in model_columns:
                message = f"Benchmark model {benchmark_model!r} not found in loss frame."
                raise ValueError(message)
            if reference_models is None:
                reference_models = model_columns
            elif model_columns != reference_models:
                message = (
                    "All loss metrics must produce the same model ordering; found "
                    f"{model_columns!r} != {reference_models!r} for {metric_column!r}."
                )
                raise ValueError(message)
            loss_matrices[metric_column] = (matrix, model_columns, target_dates)

        progress.update(task_id, description="Stage 10 Diebold-Mariano lag sensitivity")
        dm_rows: list[dict[str, object]] = []
        for metric_column in sensitivity_config.loss_metrics:
            matrix, model_columns, _ = loss_matrices[metric_column]
            benchmark_losses = matrix[:, model_columns.index(benchmark_model)]
            for max_lag in sensitivity_config.dm_max_lags:
                for model_name in model_columns:
                    if model_name == benchmark_model:
                        continue
                    result_row = asdict(
                        diebold_mariano_test(
                            loss_a=benchmark_losses,
                            loss_b=matrix[:, model_columns.index(model_name)],
                            model_a=benchmark_model,
                            model_b=model_name,
                            alternative=sensitivity_config.dm_alternative,
                            max_lag=max_lag,
                        )
                    )
                    result_row["loss_metric"] = metric_column
                    dm_rows.append(result_row)
                progress.advance(task_id)
        write_bytes_atomic(dm_results_path, orjson.dumps(dm_rows, option=orjson.OPT_INDENT_2))

        progress.update(task_id, description="Stage 10 SPA block-length sensitivity")
        spa_rows: list[dict[str, object]] = []
        for metric_column in sensitivity_config.loss_metrics:
            matrix, model_columns, _ = loss_matrices[metric_column]
            candidate_models = tuple(
                model for model in model_columns if model != benchmark_model
            )
            candidate_losses = np.column_stack(
                [matrix[:, model_columns.index(model)] for model in candidate_models]
            )
            benchmark_losses = matrix[:, model_columns.index(benchmark_model)]
            for block_size in sensitivity_config.block_sizes:
                spa_row = asdict(
                    superior_predictive_ability_test(
                        benchmark_losses=benchmark_losses,
                        candidate_losses=candidate_losses,
                        benchmark_model=benchmark_model,
                        candidate_models=candidate_models,
                        alpha=sensitivity_config.alpha,
                        block_size=block_size,
                        bootstrap_reps=sensitivity_config.bootstrap_reps,
                        seed=sensitivity_config.bootstrap_seed,
                    )
                )
                spa_row["loss_metric"] = metric_column
                spa_rows.append(spa_row)
                progress.advance(task_id)
        write_bytes_atomic(spa_results_path, orjson.dumps(spa_rows, option=orjson.OPT_INDENT_2))

        progress.update(task_id, description="Stage 10 model-confidence-set block sensitivity")
        mcs_rows: list[dict[str, object]] = []
        for metric_column in sensitivity_config.loss_metrics:
            matrix, model_columns, _ = loss_matrices[metric_column]
            for block_size in sensitivity_config.block_sizes:
                mcs_row = asdict(
                    model_confidence_set(
                        losses=matrix,
                        model_names=model_columns,
                        alpha=sensitivity_config.alpha,
                        block_size=block_size,
                        bootstrap_reps=sensitivity_config.bootstrap_reps,
                        seed=sensitivity_config.bootstrap_seed,
                    )
                )
                mcs_row["loss_metric"] = metric_column
                mcs_rows.append(mcs_row)
                progress.advance(task_id)
        write_bytes_atomic(mcs_results_path, orjson.dumps(mcs_rows, option=orjson.OPT_INDENT_2))

        progress.update(task_id, description="Stage 10 weighting-robustness aggregation")
        panel = pl.read_parquet(panel_path)
        robust_daily = build_weighting_robustness_daily_losses(panel)
        write_parquet_frame(robust_daily, robust_daily_path)
        progress.advance(task_id)

        robust_summary = summarize_weighting_robustness(robust_daily, daily_loss_frame)
        write_parquet_frame(robust_summary, robust_summary_path)
        write_bytes_atomic(
            robust_summary_markdown_path,
            _robust_summary_markdown(robust_summary, workflow_paths.run_label).encode("utf-8"),
        )
        progress.advance(task_id)

    output_paths = [
        dm_results_path,
        spa_results_path,
        mcs_results_path,
        robust_daily_path,
        robust_summary_path,
        robust_summary_markdown_path,
    ]
    run_manifest_path = write_run_manifest(
        execution_identity=execution_identity,
        manifests_dir=raw_config.manifests_dir,
        repo_root=_repo_root(),
        script_name="10_run_stats_sensitivity",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            sensitivity_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[daily_loss_path, panel_path],
        output_artifact_paths=output_paths,
        data_manifest_paths=[daily_loss_path, panel_path],
        random_seed=sensitivity_config.bootstrap_seed,
        extra_metadata={
            "loss_metrics": list(sensitivity_config.loss_metrics),
            "benchmark_model": benchmark_model,
            "dm_max_lags": list(sensitivity_config.dm_max_lags),
            "block_sizes": list(sensitivity_config.block_sizes),
            "bootstrap_reps": sensitivity_config.bootstrap_reps,
            "run_profile_name": run_profile_name,
            "workflow_run_label": workflow_paths.run_label,
            "official_stats_dir_read_only": str(workflow_paths.stats_dir),
        },
    )
    typer.echo(f"Saved stats-sensitivity outputs to {output_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
