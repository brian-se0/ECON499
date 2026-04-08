from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.config import RawDataConfig, load_yaml_config
from ivsurf.evaluation.alignment import (
    build_forecast_realization_panel,
    load_actual_surface_frame,
    load_forecast_frame,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.stats.diebold_mariano import diebold_mariano_test
from ivsurf.stats.mcs import model_confidence_set
from ivsurf.stats.spa import superior_predictive_ability_test

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
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    metrics_config = load_yaml_config(metrics_config_path)
    stats_config = load_yaml_config(stats_config_path)

    actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir)
    forecast_frame = load_forecast_frame(raw_config.gold_dir / "forecasts")
    panel = build_forecast_realization_panel(
        actual_surface_frame=actual_surface_frame,
        forecast_frame=forecast_frame,
    )
    output_dir = raw_config.manifests_dir / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(output_dir / "forecast_realization_panel.parquet", compression="zstd")

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

    dm_results = []
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
    (output_dir / "dm_results.json").write_bytes(
        orjson.dumps(dm_results, option=orjson.OPT_INDENT_2)
    )

    candidate_models = tuple(model for model in model_columns if model != benchmark_model)
    candidate_losses = np.column_stack(
        [loss_matrix[:, model_columns.index(model)] for model in candidate_models]
    )
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
    typer.echo(f"Saved stats outputs to {output_dir}")


if __name__ == "__main__":
    app()

