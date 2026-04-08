from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import typer

from ivsurf.config import (
    FeatureConfig,
    MarketCalendarConfig,
    RawDataConfig,
    SurfaceGridConfig,
    WalkforwardConfig,
    load_yaml_config,
)
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
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
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    feature_config = FeatureConfig.model_validate(load_yaml_config(feature_config_path))
    calendar_config = MarketCalendarConfig.model_validate(load_yaml_config(raw_config_path))
    walkforward_config = WalkforwardConfig.model_validate(load_yaml_config(walkforward_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    gold_files = sorted(raw_config.gold_dir.glob("year=*/*.parquet"))
    if not gold_files:
        message = "No gold surface files found. Run scripts/03_build_surfaces.py first."
        raise FileNotFoundError(message)
    surface_frame = pl.concat([pl.read_parquet(path) for path in gold_files]).sort(
        ["quote_date", "maturity_index", "moneyness_index"]
    )

    daily_dataset = build_daily_feature_dataset(
        surface_frame=surface_frame,
        grid=grid,
        feature_config=feature_config,
        calendar_config=calendar_config,
    ).feature_frame
    output_path = raw_config.gold_dir / "daily_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_dataset.write_parquet(output_path, compression="zstd", statistics=True)

    dates = daily_dataset["quote_date"].to_list()
    if any(not isinstance(value, date) for value in dates):
        message = "daily_features.parquet must contain Polars Date quote_date values."
        raise TypeError(message)
    split_hash = serialize_splits(
        splits=build_walkforward_splits(dates=dates, config=walkforward_config),
        output_path=raw_config.manifests_dir / "walkforward_splits.json",
    )
    typer.echo(f"Saved feature dataset to {output_path} and split hash {split_hash}")


if __name__ == "__main__":
    app()

