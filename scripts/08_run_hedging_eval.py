from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from ivsurf.config import HpoProfileConfig, RawDataConfig, TrainingProfileConfig, load_yaml_config
from ivsurf.evaluation.alignment import (
    load_actual_surface_frame,
    load_daily_spot_frame,
    load_forecast_frame,
)
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import surface_interpolator_from_frame
from ivsurf.progress import create_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


def _actual_surface_lookup(actual_surface_frame: pl.DataFrame) -> dict[object, pl.DataFrame]:
    groups = actual_surface_frame.partition_by("quote_date", as_dict=True)
    return {key[0]: value for key, value in groups.items()}


def _forecast_surface_groups(forecast_frame: pl.DataFrame) -> list[pl.DataFrame]:
    return forecast_frame.partition_by(
        ["model_name", "quote_date", "target_date"],
        maintain_order=True,
    )


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    hedging_config_path: Path = Path("configs/eval/hedging.yaml"),
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
    hedging_config = load_yaml_config(hedging_config_path)
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
    )

    actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir)
    forecast_frame = load_forecast_frame(workflow_paths.forecast_dir)
    spot_frame = load_daily_spot_frame(raw_config.silver_dir)
    spot_lookup = {
        row["quote_date"]: float(row["spot_1545"])
        for row in spot_frame.iter_rows(named=True)
    }
    actual_lookup = _actual_surface_lookup(actual_surface_frame)
    forecast_groups = _forecast_surface_groups(forecast_frame)

    results = []
    with create_progress() as progress:
        task_id = progress.add_task("Stage 08 hedging evaluation", total=len(forecast_groups))
        for group in forecast_groups:
            model_name = str(group["model_name"][0])
            quote_date = group["quote_date"][0]
            target_date = group["target_date"][0]
            progress.update(
                task_id,
                description=f"Stage 08 hedging {model_name}: {quote_date} -> {target_date}",
            )
            if quote_date not in actual_lookup or target_date not in actual_lookup:
                message = (
                    f"Missing actual surface for quote_date={quote_date} "
                    f"or target_date={target_date}."
                )
                raise ValueError(message)
            if quote_date not in spot_lookup or target_date not in spot_lookup:
                message = (
                    f"Missing spot state for quote_date={quote_date} "
                    f"or target_date={target_date}."
                )
                raise ValueError(message)

            result = evaluate_model_hedging(
                model_name=model_name,
                quote_date=quote_date,
                target_date=target_date,
                trade_spot=spot_lookup[quote_date],
                target_spot=spot_lookup[target_date],
                actual_surface_t=surface_interpolator_from_frame(
                    actual_lookup[quote_date],
                    total_variance_column="completed_total_variance",
                ),
                actual_surface_t1=surface_interpolator_from_frame(
                    actual_lookup[target_date],
                    total_variance_column="completed_total_variance",
                ),
                predicted_surface_t1=surface_interpolator_from_frame(
                    group,
                    total_variance_column="predicted_total_variance",
                ),
                rate=float(hedging_config["risk_free_rate"]),
                level_notional=float(hedging_config["level_notional"]),
                skew_notional=float(hedging_config["skew_notional"]),
                calendar_notional=float(hedging_config["calendar_notional"]),
                skew_moneyness_abs=float(hedging_config["skew_moneyness_abs"]),
                short_maturity_days=int(hedging_config["short_maturity_days"]),
                long_maturity_days=int(hedging_config["long_maturity_days"]),
                hedge_maturity_days=int(hedging_config["hedge_maturity_days"]),
                hedge_straddle_moneyness=float(hedging_config["hedge_straddle_moneyness"]),
            )
            results.append(asdict(result))
            progress.advance(task_id)

    output_dir = workflow_paths.hedging_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pl.DataFrame(results).sort(["model_name", "quote_date", "target_date"])
    results_frame.write_parquet(output_dir / "hedging_results.parquet", compression="zstd")

    summary_frame = summarize_hedging_results(results_frame)
    summary_frame.write_parquet(output_dir / "hedging_summary.parquet", compression="zstd")
    forecast_paths = sorted(workflow_paths.forecast_dir.glob("*.parquet"))
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="08_run_hedging_eval",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            hedging_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            raw_config.manifests_dir / "silver_build_summary.json",
            *forecast_paths,
        ],
        output_artifact_paths=[
            output_dir / "hedging_results.parquet",
            output_dir / "hedging_summary.parquet",
        ],
        data_manifest_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            raw_config.manifests_dir / "silver_build_summary.json",
            *forecast_paths,
        ],
        extra_metadata={
            "benchmark_model": "no_change",
            "n_results": results_frame.height,
            "workflow_run_label": workflow_paths.run_label,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved hedging outputs to {output_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
