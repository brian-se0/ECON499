from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import typer

from ivsurf.config import RawDataConfig, load_yaml_config
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.ingest_cboe import ingest_one_zip, list_raw_zip_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import sample_window_label
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    limit: int | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    zip_paths = list_raw_zip_files(config)
    if limit is not None:
        zip_paths = zip_paths[:limit]
    resumer = StageResumer(
        state_path=resume_state_path(config.manifests_dir, "01_ingest_cboe"),
        stage_name="01_ingest_cboe",
        context_hash=build_resume_context_hash(
            config_paths=[raw_config_path],
            input_artifact_paths=zip_paths,
            extra_tokens={"artifact_schema_version": 2},
        ),
    )

    result_rows: list[dict[str, object]] = []
    with create_progress() as progress:
        for zip_path in iter_with_progress(
            progress,
            zip_paths,
            description="Stage 01 ingesting raw Cboe daily zips",
        ):
            item_id = str(zip_path.resolve())
            if resumer.item_complete(item_id):
                result_rows.append(resumer.metadata_for(item_id))
                continue
            stale_output_paths = resumer.output_paths_for(item_id)
            if stale_output_paths:
                resumer.clear_item(item_id, output_paths=stale_output_paths)
            result = ingest_one_zip(zip_path=zip_path, config=config)
            result_row = {
                "source_zip": str(result.source_zip),
                "quote_date": result.quote_date.isoformat(),
                "status": result.status,
                "bronze_path": str(result.bronze_path) if result.bronze_path is not None else None,
                "raw_row_count": result.raw_row_count,
                "target_symbol_row_count": result.target_symbol_row_count,
                "non_target_symbol_row_count": result.non_target_symbol_row_count,
                "row_count": result.row_count,
            }
            resumer.mark_complete(
                item_id,
                output_paths=[] if result.bronze_path is None else [result.bronze_path],
                metadata=result_row,
            )
            result_rows.append(result_row)
    written_results = [row for row in result_rows if row["status"] == "written"]
    written_output_paths = [
        Path(str(row["bronze_path"]))
        for row in written_results
        if row["bronze_path"] is not None
    ]
    skipped_results = [
        row for row in result_rows if row["status"] == "skipped_out_of_sample_window"
    ]
    payload = {
        "files_seen": len(result_rows),
        "files_written": len(written_results),
        "files_skipped_out_of_sample_window": len(skipped_results),
        "rows_parsed": sum(int(row["raw_row_count"]) for row in result_rows),
        "rows_target_symbol": sum(int(row["target_symbol_row_count"]) for row in result_rows),
        "rows_filtered_non_target_symbol": sum(
            int(row["non_target_symbol_row_count"]) for row in result_rows
        ),
        "rows_written": sum(int(row["row_count"]) for row in written_results),
        "sample_window": {
            "start_date": config.sample_start_date.isoformat(),
            "end_date": config.sample_end_date.isoformat(),
        },
        "resume_context_hash": resumer.context_hash,
        "results": result_rows,
    }
    manifest_path = config.manifests_dir / "bronze_ingestion_summary.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(manifest_path, orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    run_manifest_path = write_run_manifest(
        manifests_dir=config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="01_ingest_cboe",
        started_at=started_at,
        config_paths=[raw_config_path],
        input_artifact_paths=zip_paths,
        output_artifact_paths=[manifest_path, *written_output_paths],
        data_manifest_paths=zip_paths,
        extra_metadata={
            "limit": limit,
            "files_processed": len(result_rows),
            "files_written": len(written_results),
            "files_skipped_out_of_sample_window": len(skipped_results),
            "sample_window": sample_window_label(config),
            "resume_context_hash": resumer.context_hash,
            "source_zips": [str(zip_path) for zip_path in zip_paths],
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved bronze ingestion summary to {manifest_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
