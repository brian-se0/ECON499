from __future__ import annotations

from pathlib import Path

import orjson
import typer

from ivsurf.config import RawDataConfig, load_yaml_config
from ivsurf.io.ingest_cboe import ingest_all

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    limit: int | None = None,
) -> None:
    config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    results = ingest_all(config=config, limit=limit)
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


if __name__ == "__main__":
    app()

