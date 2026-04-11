from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from ivsurf.config import (
    HedgingConfig,
    HpoProfileConfig,
    RawDataConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.evaluation.alignment import (
    assert_forecast_origins_after_hpo_boundary,
    load_actual_surface_frame,
    load_daily_spot_frame,
)
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import surface_interpolator_from_frame
from ivsurf.io.parquet import read_parquet_files, write_parquet_frame
from ivsurf.progress import create_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.training.model_factory import TUNABLE_MODEL_NAMES
from ivsurf.training.tuning import (
    load_required_tuning_results,
    require_consistent_clean_evaluation_policy,
)
from ivsurf.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


def _actual_surface_lookup(actual_surface_frame: pl.DataFrame) -> dict[object, pl.DataFrame]:
    groups = actual_surface_frame.partition_by("quote_date", as_dict=True)
    return {key[0]: value for key, value in groups.items()}


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    hedging_config_path: Path = Path("configs/eval/hedging.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    """Run stage 08 hedging evaluation with active_underlying_price_1545 spot states."""

    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    hedging_config = HedgingConfig.model_validate(load_yaml_config(hedging_config_path))
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
    )

    forecast_paths = sorted(workflow_paths.forecast_dir.glob("*.parquet"))
    tuning_manifest_paths = [
        raw_config.manifests_dir / "tuning" / hpo_profile.profile_name / f"{model_name}.json"
        for model_name in TUNABLE_MODEL_NAMES
    ]
    output_dir = workflow_paths.hedging_dir
    by_model_dir = output_dir / "by_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_model_dir.mkdir(parents=True, exist_ok=True)
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "08_run_hedging_eval"),
        stage_name="08_run_hedging_eval",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                hedging_config_path,
                hpo_profile_config_path,
                training_profile_config_path,
            ],
            input_artifact_paths=[
                raw_config.manifests_dir / "gold_surface_summary.json",
                raw_config.manifests_dir / "silver_build_summary.json",
                *tuning_manifest_paths,
                *forecast_paths,
            ],
            extra_tokens={"workflow_run_label": workflow_paths.run_label},
        ),
    )
    tuning_results = load_required_tuning_results(
        raw_config.manifests_dir,
        hpo_profile_name=hpo_profile.profile_name,
        model_names=TUNABLE_MODEL_NAMES,
    )
    clean_evaluation_policy = require_consistent_clean_evaluation_policy(tuning_results.values())

    actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir)
    # For SPX/index data the vendor underlying bid/ask fields may be zero, so stage 08 uses
    # the calc-model 15:45 active underlying price as its single daily spot contract.
    spot_frame = load_daily_spot_frame(raw_config.silver_dir)
    spot_lookup = {
        row["quote_date"]: float(row["spot_1545"]) for row in spot_frame.iter_rows(named=True)
    }
    actual_lookup = _actual_surface_lookup(actual_surface_frame)

    model_output_paths: list[Path] = []
    with create_progress() as progress:
        task_id = progress.add_task("Stage 08 hedging evaluation", total=len(forecast_paths))
        for forecast_path in forecast_paths:
            forecast_frame = pl.read_parquet(forecast_path).sort(
                ["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"]
            )
            assert_forecast_origins_after_hpo_boundary(
                forecast_frame,
                max_hpo_validation_date=clean_evaluation_policy.max_hpo_validation_date,
            )
            if forecast_frame.is_empty():
                message = f"Forecast artifact {forecast_path} is empty."
                raise ValueError(message)
            model_name = str(forecast_frame["model_name"][0])
            model_output_path = by_model_dir / f"{model_name}.parquet"
            model_output_paths.append(model_output_path)
            if resumer.item_complete(model_name, required_output_paths=[model_output_path]):
                progress.update(
                    task_id,
                    description=f"Stage 08 resume: skipping completed model {model_name}",
                )
                progress.advance(task_id)
                continue
            resumer.clear_item(model_name, output_paths=[model_output_path])
            progress.update(task_id, description=f"Stage 08 hedging model {model_name}")

            results: list[dict[str, object]] = []
            for group in forecast_frame.partition_by(
                ["model_name", "quote_date", "target_date"],
                maintain_order=True,
            ):
                quote_date = group["quote_date"][0]
                target_date = group["target_date"][0]
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
                    rate=hedging_config.risk_free_rate,
                    level_notional=hedging_config.level_notional,
                    skew_notional=hedging_config.skew_notional,
                    calendar_notional=hedging_config.calendar_notional,
                    skew_moneyness_abs=hedging_config.skew_moneyness_abs,
                    short_maturity_days=hedging_config.short_maturity_days,
                    long_maturity_days=hedging_config.long_maturity_days,
                    hedge_maturity_days=hedging_config.hedge_maturity_days,
                    hedge_straddle_moneyness=hedging_config.hedge_straddle_moneyness,
                )
                results.append(asdict(result))

            results_frame = pl.DataFrame(results).sort(["model_name", "quote_date", "target_date"])
            write_parquet_frame(results_frame, model_output_path)
            resumer.mark_complete(
                model_name,
                output_paths=[model_output_path],
                metadata={"model_name": model_name, "n_results": results_frame.height},
            )
            progress.advance(task_id)

    results_frame = read_parquet_files(model_output_paths).sort(
        ["model_name", "quote_date", "target_date"]
    )
    hedging_results_path = output_dir / "hedging_results.parquet"
    hedging_summary_path = output_dir / "hedging_summary.parquet"
    write_parquet_frame(results_frame, hedging_results_path)
    summary_frame = summarize_hedging_results(results_frame)
    write_parquet_frame(summary_frame, hedging_summary_path)
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
            *tuning_manifest_paths,
            *forecast_paths,
        ],
        output_artifact_paths=[hedging_results_path, hedging_summary_path, *model_output_paths],
        data_manifest_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            raw_config.manifests_dir / "silver_build_summary.json",
            *tuning_manifest_paths,
            *forecast_paths,
        ],
        extra_metadata={
            "benchmark_model": "no_change",
            "n_results": results_frame.height,
            "hedge_spot_assumption": "no_change",
            "spot_source": "active_underlying_price_1545",
            "max_hpo_validation_date": clean_evaluation_policy.max_hpo_validation_date.isoformat(),
            "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
            "workflow_run_label": workflow_paths.run_label,
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved hedging outputs to {output_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
