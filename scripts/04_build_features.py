from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import typer

from ivsurf.config import (
    FeatureConfig,
    RawDataConfig,
    SurfaceGridConfig,
    WalkforwardConfig,
    calendar_config_from_raw,
    load_yaml_config,
)
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.splits.manifests import serialize_splits
from ivsurf.splits.walkforward import build_walkforward_splits
from ivsurf.surfaces.grid import SurfaceGrid

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    feature_config_path: Path = Path("configs/data/features.yaml"),
    walkforward_config_path: Path = Path("configs/eval/walkforward.yaml"),
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    feature_config = FeatureConfig.model_validate(load_yaml_config(feature_config_path))
    calendar_config = calendar_config_from_raw(raw_config)
    walkforward_config = WalkforwardConfig.model_validate(load_yaml_config(walkforward_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    gold_files = sorted(raw_config.gold_dir.glob("year=*/*.parquet"))
    if not gold_files:
        message = "No gold surface files found. Run scripts/03_build_surfaces.py first."
        raise FileNotFoundError(message)
    gold_frames: list[pl.DataFrame] = []
    with create_progress() as progress:
        for gold_path in iter_with_progress(
            progress,
            gold_files,
            description="Stage 04 loading gold surfaces for feature build",
        ):
            gold_frames.append(pl.read_parquet(gold_path))
    surface_frame = pl.concat(gold_frames).sort(["quote_date", "maturity_index", "moneyness_index"])

    daily_dataset = build_daily_feature_dataset(
        surface_frame=surface_frame,
        grid=grid,
        feature_config=feature_config,
        calendar_config=calendar_config,
    ).feature_frame
    output_path = raw_config.gold_dir / "daily_features.parquet"
    write_parquet_frame(daily_dataset, output_path)

    dates = daily_dataset["quote_date"].to_list()
    if any(not isinstance(value, date) for value in dates):
        message = "daily_features.parquet must contain Polars Date quote_date values."
        raise TypeError(message)
    split_manifest_path = raw_config.manifests_dir / "walkforward_splits.json"
    split_hash = serialize_splits(
        splits=build_walkforward_splits(dates=dates, config=walkforward_config),
        output_path=split_manifest_path,
    )
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="04_build_features",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            feature_config_path,
            walkforward_config_path,
        ],
        input_artifact_paths=[raw_config.manifests_dir / "gold_surface_summary.json"],
        output_artifact_paths=[output_path, split_manifest_path],
        data_manifest_paths=[raw_config.manifests_dir / "gold_surface_summary.json"],
        split_manifest_path=split_manifest_path,
        extra_metadata={
            "feature_rows": daily_dataset.height,
            "split_hash": split_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved feature dataset to {output_path} and split hash {split_hash}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
