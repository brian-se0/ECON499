from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import orjson
import polars as pl
import typer

from ivsurf.cleaning.derived_fields import add_derived_fields, build_tau_lookup
from ivsurf.cleaning.option_filters import apply_option_quality_flags
from ivsurf.config import (
    CleaningConfig,
    RawDataConfig,
    calendar_config_from_raw,
    load_yaml_config,
)
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import require_quote_date_in_sample_window, sample_window_label
from ivsurf.qc.schema_checks import assert_required_columns
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path

app = typer.Typer(add_completion=False)


def _silver_path(bronze_path: Path, raw_config: RawDataConfig) -> Path:
    year_partition = bronze_path.parent.name
    output_dir = raw_config.silver_dir / year_partition
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / bronze_path.name


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    cleaning_config_path: Path = Path("configs/data/cleaning.yaml"),
    limit: int | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    cleaning_config = CleaningConfig.model_validate(load_yaml_config(cleaning_config_path))
    calendar_config = calendar_config_from_raw(raw_config)

    bronze_files = sorted_artifact_files(raw_config.bronze_dir, "year=*/*.parquet")
    if limit is not None:
        bronze_files = bronze_files[:limit]
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "02_build_option_panel"),
        stage_name="02_build_option_panel",
        context_hash=build_resume_context_hash(
            config_paths=[raw_config_path, cleaning_config_path],
            input_artifact_paths=bronze_files,
        ),
    )

    summary_rows: list[dict[str, object]] = []
    with create_progress() as progress:
        for bronze_path in iter_with_progress(
            progress,
            bronze_files,
            description="Stage 02 building silver option panel",
        ):
            item_id = str(bronze_path.resolve())
            if resumer.item_complete(item_id):
                summary_rows.append(resumer.metadata_for(item_id))
                continue
            stale_output_paths = resumer.output_paths_for(item_id)
            if stale_output_paths:
                resumer.clear_item(item_id, output_paths=stale_output_paths)
            frame = pl.read_parquet(bronze_path)
            assert_required_columns(
                frame.columns,
                required_columns=(
                    "quote_date",
                    "expiration",
                    "root",
                    "strike",
                    "option_type",
                    "bid_1545",
                    "ask_1545",
                    "vega_1545",
                    "active_underlying_price_1545",
                    "implied_volatility_1545",
                ),
                dataset_name=str(bronze_path),
            )
            quote_date = frame["quote_date"][0]
            if not isinstance(quote_date, date):
                message = f"Unexpected quote_date type in {bronze_path}"
                raise TypeError(message)
            require_quote_date_in_sample_window(
                quote_date,
                raw_config,
                context=f"Stage 02 bronze artifact {bronze_path}",
            )

            tau_lookup = build_tau_lookup(frame=frame, calendar_config=calendar_config)
            enriched = add_derived_fields(frame=frame, tau_lookup=tau_lookup)
            enriched = apply_option_quality_flags(frame=enriched, config=cleaning_config)
            summary_status = "built"

            output_path = _silver_path(bronze_path=bronze_path, raw_config=raw_config)
            write_parquet_frame(enriched, output_path)
            summary_row = {
                "silver_path": str(output_path),
                "quote_date": quote_date.isoformat(),
                "status": summary_status,
                "rows": enriched.height,
                "valid_rows": enriched.filter(pl.col("is_valid_observation")).height,
            }
            resumer.mark_complete(
                item_id,
                output_paths=[output_path],
                metadata=summary_row,
            )
            summary_rows.append(summary_row)

    summary_path = raw_config.manifests_dir / "silver_build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(summary_path, orjson.dumps(summary_rows, option=orjson.OPT_INDENT_2))
    silver_output_paths = [Path(str(row["silver_path"])) for row in summary_rows]
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="02_build_option_panel",
        started_at=started_at,
        config_paths=[raw_config_path, cleaning_config_path],
        input_artifact_paths=[
            raw_config.manifests_dir / "bronze_ingestion_summary.json",
            *bronze_files,
        ],
        output_artifact_paths=[summary_path, *silver_output_paths],
        data_manifest_paths=bronze_files,
        extra_metadata={
            "limit": limit,
            "bronze_files_processed": len(bronze_files),
            "sample_window": sample_window_label(raw_config),
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved silver build summary to {summary_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
