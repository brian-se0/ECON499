from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import typer

from ivsurf.config import RawDataConfig, load_yaml_config
from ivsurf.io.ingest_cboe import ingest_one_zip, list_raw_zip_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.reproducibility import write_run_manifest

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

    results = []
    with create_progress() as progress:
        for zip_path in iter_with_progress(
            progress,
            zip_paths,
            description="Stage 01 ingesting raw Cboe daily zips",
        ):
            results.append(ingest_one_zip(zip_path=zip_path, config=config))
    payload = {
        "files_processed": len(results),
        "rows_processed": sum(result.row_count for result in results),
        "results": [
            {
                "source_zip": str(result.source_zip),
                "bronze_path": str(result.bronze_path),
                "row_count": result.row_count,
            }
            for result in results
        ],
    }
    manifest_path = config.manifests_dir / "bronze_ingestion_summary.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    run_manifest_path = write_run_manifest(
        manifests_dir=config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="01_ingest_cboe",
        started_at=started_at,
        config_paths=[raw_config_path],
        input_artifact_paths=[],
        output_artifact_paths=[manifest_path],
        data_manifest_paths=[manifest_path],
        extra_metadata={
            "limit": limit,
            "files_processed": len(results),
            "source_zips": [str(result.source_zip) for result in results],
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved bronze ingestion summary to {manifest_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
