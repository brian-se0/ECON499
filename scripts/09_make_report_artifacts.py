from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import polars as pl
import typer

from ivsurf.config import (
    HpoProfileConfig,
    RawDataConfig,
    ReportArtifactsConfig,
    SurfaceGridConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.evaluation.alignment import load_actual_surface_frame, load_forecast_frame
from ivsurf.evaluation.diagnostics import (
    build_actual_diagnostic_frame,
    build_forecast_diagnostic_frame,
    summarize_diagnostic_frame,
)
from ivsurf.evaluation.interpolation_sensitivity import (
    build_interpolation_sensitivity_frame,
    summarize_interpolation_sensitivity,
)
from ivsurf.evaluation.slice_reports import build_slice_metric_frame
from ivsurf.io.atomic import write_text_atomic
from ivsurf.io.parquet import write_csv_frame, write_parquet_frame
from ivsurf.progress import create_progress
from ivsurf.reports.figures import write_multi_line_chart, write_ranked_bar_chart
from ivsurf.reports.tables import (
    build_dm_results_table,
    build_mcs_table,
    build_ranked_hedging_table,
    build_ranked_loss_table,
    build_report_overview_markdown,
    build_slice_leader_table,
    build_spa_table,
    write_table_artifacts,
)
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


def _require_file(path: Path) -> Path:
    if not path.exists():
        message = f"Required artifact is missing: {path}"
        raise FileNotFoundError(message)
    return path


def _require_columns(frame: pl.DataFrame, *, columns: tuple[str, ...], artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if not missing:
        return
    message = f"{artifact_name} is missing required columns: {missing}."
    raise ValueError(message)


def _write_frame_bundle(
    output_dir: Path,
    name: str,
    frame: pl.DataFrame,
    *,
    include_markdown: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"
    write_parquet_frame(frame, parquet_path)
    write_csv_frame(frame, csv_path)
    written = [parquet_path, csv_path]
    if include_markdown:
        from ivsurf.reports.tables import frame_to_markdown

        markdown_path = output_dir / f"{name}.md"
        write_text_atomic(markdown_path, frame_to_markdown(frame), encoding="utf-8")
        written.append(markdown_path)
    return written


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    metrics_config_path: Path = Path("configs/eval/metrics.yaml"),
    stats_config_path: Path = Path("configs/eval/stats_tests.yaml"),
    report_config_path: Path = Path("configs/eval/report_artifacts.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    repo_root = Path.cwd()
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    metrics_config = load_yaml_config(metrics_config_path)
    stats_config = load_yaml_config(stats_config_path)
    report_config = ReportArtifactsConfig.model_validate(load_yaml_config(report_config_path))
    grid = SurfaceGrid.from_config(surface_config)
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
    )

    if str(stats_config["benchmark_model"]) != report_config.benchmark_model:
        message = (
            "Report benchmark_model must match configs/eval/stats_tests.yaml "
            f"({stats_config['benchmark_model']} != {report_config.benchmark_model})."
        )
        raise ValueError(message)

    stats_dir = workflow_paths.stats_dir
    hedging_dir = workflow_paths.hedging_dir
    report_dir = workflow_paths.report_dir
    tables_dir = report_dir / "tables"
    details_dir = report_dir / "details"
    figures_dir = report_dir / "figures"

    panel_path = _require_file(stats_dir / "forecast_realization_panel.parquet")
    daily_loss_path = _require_file(stats_dir / "daily_loss_frame.parquet")
    loss_summary_path = _require_file(stats_dir / "loss_summary.parquet")
    dm_results_path = _require_file(stats_dir / "dm_results.json")
    spa_result_path = _require_file(stats_dir / "spa_result.json")
    mcs_result_path = _require_file(stats_dir / "mcs_result.json")
    hedging_results_path = _require_file(hedging_dir / "hedging_results.parquet")
    hedging_summary_path = _require_file(hedging_dir / "hedging_summary.parquet")
    forecast_paths = sorted(workflow_paths.forecast_dir.glob("*.parquet"))
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "09_make_report_artifacts"),
        stage_name="09_make_report_artifacts",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                surface_config_path,
                metrics_config_path,
                stats_config_path,
                report_config_path,
                hpo_profile_config_path,
                training_profile_config_path,
            ],
            input_artifact_paths=[
                panel_path,
                daily_loss_path,
                loss_summary_path,
                dm_results_path,
                spa_result_path,
                mcs_result_path,
                hedging_results_path,
                hedging_summary_path,
                raw_config.manifests_dir / "gold_surface_summary.json",
                *forecast_paths,
            ],
            extra_tokens={"workflow_run_label": workflow_paths.run_label},
        ),
    )

    with create_progress() as progress:
        task_id = progress.add_task("Stage 09 report artifact generation", total=6)

        progress.update(task_id, description="Stage 09 loading saved artifacts")
        actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir)
        forecast_frame = load_forecast_frame(workflow_paths.forecast_dir)
        panel = pl.read_parquet(panel_path)
        daily_loss_frame = pl.read_parquet(daily_loss_path)
        loss_summary = pl.read_parquet(loss_summary_path)
        hedging_results = pl.read_parquet(hedging_results_path)
        hedging_summary = pl.read_parquet(hedging_summary_path)
        dm_results = orjson.loads(dm_results_path.read_bytes())
        spa_result = orjson.loads(spa_result_path.read_bytes())
        mcs_result = orjson.loads(mcs_result_path.read_bytes())
        progress.advance(task_id)

        progress.update(task_id, description="Stage 09 computing slice and diagnostic reports")
        slice_metric_frame = build_slice_metric_frame(
            panel=panel,
            positive_floor=float(metrics_config["positive_floor"]),
            stress_windows=report_config.stress_windows,
        )
        forecast_diagnostics = build_forecast_diagnostic_frame(forecast_frame, grid)
        actual_diagnostics = build_actual_diagnostic_frame(actual_surface_frame, grid)
        combined_diagnostics = pl.concat([forecast_diagnostics, actual_diagnostics]).sort(
            ["model_name", "quote_date", "target_date"]
        )
        diagnostic_summary = summarize_diagnostic_frame(combined_diagnostics)

        interpolation_sensitivity = build_interpolation_sensitivity_frame(
            actual_surface_frame,
            grid=grid,
            surface_config=surface_config,
            alternate_order=report_config.interpolation_comparison_order,
            interpolation_cycles=report_config.interpolation_cycles,
        )
        interpolation_summary = summarize_interpolation_sensitivity(interpolation_sensitivity)
        progress.advance(task_id)

        progress.update(task_id, description="Stage 09 building report tables")
        summary_mean_column = f"mean_{report_config.primary_loss_metric}"
        summary_std_column = f"std_{report_config.primary_loss_metric}"
        _require_columns(
            loss_summary,
            columns=(summary_mean_column, summary_std_column, "model_name"),
            artifact_name="loss_summary.parquet",
        )
        ranked_loss_table = build_ranked_loss_table(
            loss_summary=loss_summary,
            benchmark_model=report_config.benchmark_model,
            metric_column=summary_mean_column,
        )
        ranked_hedging_table = build_ranked_hedging_table(
            hedging_summary=hedging_summary,
            benchmark_model=report_config.benchmark_model,
        )
        dm_table = build_dm_results_table(dm_results)
        spa_table = build_spa_table(spa_result)
        mcs_table = build_mcs_table(
            mcs_result,
            all_models=loss_summary["model_name"].to_list(),
        )
        slice_leader_table = build_slice_leader_table(
            slice_metric_frame=slice_metric_frame,
            benchmark_model=report_config.benchmark_model,
            metric_column="wrmse_total_variance",
        )
        table_names = (
            "ranked_loss_summary",
            "ranked_hedging_summary",
            "dm_results",
            "spa_result",
            "mcs_result",
            "slice_leaders",
            "arbitrage_diagnostic_summary",
            "interpolation_sensitivity_summary",
        )
        table_output_paths = [
            path
            for name in table_names
            for path in (tables_dir / f"{name}.csv", tables_dir / f"{name}.md")
        ]

        if resumer.item_complete("report_tables", required_output_paths=table_output_paths):
            table_paths = table_output_paths
        else:
            resumer.clear_item("report_tables", output_paths=table_output_paths)
            table_paths = write_table_artifacts(
                tables_dir,
                tables={
                    "ranked_loss_summary": ranked_loss_table,
                    "ranked_hedging_summary": ranked_hedging_table,
                    "dm_results": dm_table,
                    "spa_result": spa_table,
                    "mcs_result": mcs_table,
                    "slice_leaders": slice_leader_table,
                    "arbitrage_diagnostic_summary": diagnostic_summary,
                    "interpolation_sensitivity_summary": interpolation_summary,
                },
            )
            resumer.mark_complete(
                "report_tables",
                output_paths=table_paths,
                metadata={"table_count": len(table_names)},
            )
        progress.advance(task_id)

        progress.update(task_id, description="Stage 09 writing detailed report frames")
        detail_names = (
            "slice_metric_frame",
            "forecast_diagnostics",
            "actual_diagnostics",
            "combined_diagnostics",
            "interpolation_sensitivity",
            "daily_loss_frame",
            "hedging_results",
        )
        detail_output_paths = [
            path
            for name in detail_names
            for path in (details_dir / f"{name}.parquet", details_dir / f"{name}.csv")
        ]
        if resumer.item_complete("report_details", required_output_paths=detail_output_paths):
            detail_paths = detail_output_paths
        else:
            resumer.clear_item("report_details", output_paths=detail_output_paths)
            detail_paths = []
            detail_paths.extend(
                _write_frame_bundle(details_dir, "slice_metric_frame", slice_metric_frame)
            )
            detail_paths.extend(
                _write_frame_bundle(details_dir, "forecast_diagnostics", forecast_diagnostics)
            )
            detail_paths.extend(
                _write_frame_bundle(details_dir, "actual_diagnostics", actual_diagnostics)
            )
            detail_paths.extend(
                _write_frame_bundle(details_dir, "combined_diagnostics", combined_diagnostics)
            )
            detail_paths.extend(
                _write_frame_bundle(
                    details_dir,
                    "interpolation_sensitivity",
                    interpolation_sensitivity,
                )
            )
            detail_paths.extend(
                _write_frame_bundle(details_dir, "daily_loss_frame", daily_loss_frame)
            )
            detail_paths.extend(
                _write_frame_bundle(details_dir, "hedging_results", hedging_results)
            )
            resumer.mark_complete(
                "report_details",
                output_paths=detail_paths,
                metadata={"detail_bundle_count": len(detail_names)},
            )
        progress.advance(task_id)

        progress.update(task_id, description="Stage 09 rendering report figures")
        top_models = tuple(
            ranked_loss_table.head(report_config.top_models_per_figure)["model_name"].to_list()
        )
        expected_figure_paths = [
            figures_dir / "loss_ranking.svg",
            figures_dir / "hedging_ranking.svg",
            figures_dir / "calendar_violation_ranking.svg",
            figures_dir / "interpolation_sensitivity_worst_days.svg",
            figures_dir / "maturity_slice_wrmse.svg",
            figures_dir / "moneyness_slice_wrmse.svg",
        ]
        if resumer.item_complete("report_figures", required_output_paths=expected_figure_paths):
            figure_paths = expected_figure_paths
        else:
            resumer.clear_item("report_figures", output_paths=expected_figure_paths)
            figure_paths = [
            write_ranked_bar_chart(
                ranked_loss_table,
                label_column="model_name",
                value_column=summary_mean_column,
                output_path=figures_dir / "loss_ranking.svg",
                title=f"Loss Ranking by mean {report_config.primary_loss_metric}",
                top_n=report_config.top_models_per_figure,
            ),
            write_ranked_bar_chart(
                ranked_hedging_table,
                label_column="model_name",
                value_column="mean_abs_revaluation_error",
                output_path=figures_dir / "hedging_ranking.svg",
                title="Hedging Revaluation Ranking",
                top_n=report_config.top_models_per_figure,
            ),
            write_ranked_bar_chart(
                diagnostic_summary,
                label_column="model_name",
                value_column="mean_calendar_violation_magnitude",
                output_path=figures_dir / "calendar_violation_ranking.svg",
                title="Average Calendar Violation Magnitude",
                top_n=report_config.top_models_per_figure + 1,
            ),
            write_ranked_bar_chart(
                interpolation_sensitivity.sort("max_abs_diff", descending=True).head(15),
                label_column="quote_date",
                value_column="max_abs_diff",
                output_path=figures_dir / "interpolation_sensitivity_worst_days.svg",
                title="Worst Interpolation-Order Sensitivity Days",
                top_n=15,
            ),
            write_multi_line_chart(
                slice_metric_frame.filter(
                    (pl.col("slice_family") == "maturity")
                    & (pl.col("evaluation_scope") == "observed")
                ),
                x_column="slice_value_float",
                y_column="wrmse_total_variance",
                series_column="model_name",
                output_path=figures_dir / "maturity_slice_wrmse.svg",
                title="Observed-Cell WRMSE by Maturity Slice",
                x_label="Maturity (days)",
                y_label="WRMSE total variance",
                include_series=top_models,
            ),
            write_multi_line_chart(
                slice_metric_frame.filter(
                    (pl.col("slice_family") == "moneyness")
                    & (pl.col("evaluation_scope") == "observed")
                ),
                x_column="slice_value_float",
                y_column="wrmse_total_variance",
                series_column="model_name",
                output_path=figures_dir / "moneyness_slice_wrmse.svg",
                title="Observed-Cell WRMSE by Moneyness Slice",
                x_label="Log-moneyness",
                y_label="WRMSE total variance",
                include_series=top_models,
            ),
            ]
            resumer.mark_complete(
                "report_figures",
                output_paths=figure_paths,
                metadata={"figure_count": len(figure_paths)},
            )
        progress.advance(task_id)

        progress.update(task_id, description="Stage 09 writing report index")
        overview_path = report_dir / "index.md"
        if not resumer.item_complete("report_index", required_output_paths=[overview_path]):
            resumer.clear_item("report_index", output_paths=[overview_path])
            write_text_atomic(
                overview_path,
                build_report_overview_markdown(
                    benchmark_model=report_config.benchmark_model,
                    primary_loss_metric=report_config.primary_loss_metric,
                    summary_metric_column=summary_mean_column,
                    ranked_loss_table=ranked_loss_table,
                    ranked_hedging_table=ranked_hedging_table,
                    mcs_table=mcs_table,
                    slice_leader_table=slice_leader_table,
                    interpolation_summary=interpolation_summary,
                ),
                encoding="utf-8",
            )
            resumer.mark_complete(
                "report_index",
                output_paths=[overview_path],
                metadata={"report_dir": str(report_dir)},
            )
        progress.advance(task_id)

    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=repo_root,
        script_name="09_make_report_artifacts",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            metrics_config_path,
            stats_config_path,
            report_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[
            panel_path,
            daily_loss_path,
            loss_summary_path,
            dm_results_path,
            spa_result_path,
            mcs_result_path,
            hedging_results_path,
            hedging_summary_path,
            raw_config.manifests_dir / "gold_surface_summary.json",
            *sorted(workflow_paths.forecast_dir.glob("*.parquet")),
        ],
        output_artifact_paths=[overview_path, *table_paths, *detail_paths, *figure_paths],
        data_manifest_paths=[
            panel_path,
            daily_loss_path,
            loss_summary_path,
            hedging_results_path,
            hedging_summary_path,
            raw_config.manifests_dir / "gold_surface_summary.json",
            *sorted(workflow_paths.forecast_dir.glob("*.parquet")),
        ],
        random_seed=int(stats_config["bootstrap_seed"]),
        extra_metadata={
            "report_dir": str(report_dir),
            "top_models_for_figures": list(top_models),
            "workflow_run_label": workflow_paths.run_label,
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved report artifacts to {report_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
