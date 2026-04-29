from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import orjson
import typer

from ivsurf.config import (
    FeatureConfig,
    RawDataConfig,
    SurfaceGridConfig,
    WalkforwardConfig,
    calendar_config_from_raw,
    load_yaml_config,
)
from ivsurf.exceptions import DataValidationError
from ivsurf.features.availability import build_feature_availability_manifest
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import scan_parquet_files, write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import (
    assert_frame_dates_in_sample_window,
    sample_window_expr,
    sample_window_label,
)
from ivsurf.reproducibility import sha256_file, write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.splits.manifests import serialize_splits
from ivsurf.splits.walkforward import build_walkforward_splits
from ivsurf.surfaces.grid import SurfaceGrid, require_surface_grid_metadata

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

    gold_files = sorted_artifact_files(raw_config.gold_dir, "year=*/*.parquet")
    if not gold_files:
        message = "No gold surface files found. Run scripts/03_build_surfaces.py first."
        raise FileNotFoundError(message)
    output_path = raw_config.gold_dir / "daily_features.parquet"
    availability_manifest_path = raw_config.manifests_dir / "feature_availability_manifest.json"
    split_manifest_path = raw_config.manifests_dir / "walkforward_splits.json"
    skipped_dates_manifest_path = raw_config.manifests_dir / "gold_surface_skipped_dates.json"
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "04_build_features"),
        stage_name="04_build_features",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                surface_config_path,
                feature_config_path,
                walkforward_config_path,
            ],
            input_artifact_paths=gold_files,
            extra_tokens={"artifact_schema_version": 5},
        ),
    )
    resume_item_id = "daily_feature_dataset"
    if resumer.item_complete(
        resume_item_id,
        required_output_paths=[output_path, availability_manifest_path, split_manifest_path],
    ):
        typer.echo(
            "Stage 04 resume: existing daily_features.parquet and walkforward_splits.json "
            "match the current context; skipping rebuild."
        )
        return
    resumer.clear_item(
        resume_item_id,
        output_paths=[output_path, availability_manifest_path, split_manifest_path],
    )
    with create_progress() as progress:
        for _gold_path in iter_with_progress(
            progress,
            gold_files,
            description="Stage 04 validating gold surface scope",
        ):
            continue
    gold_scan = scan_parquet_files(gold_files)
    out_of_window_dates = (
        gold_scan.select("quote_date")
        .filter(~sample_window_expr(raw_config, column="quote_date"))
        .unique()
        .sort("quote_date")
        .collect(engine="streaming")
    )
    if not out_of_window_dates.is_empty():
        offending_dates = ", ".join(
            value.isoformat()
            for value in out_of_window_dates["quote_date"].to_list()
            if isinstance(value, date)
        )
        message = (
            "Stage 04 found gold surfaces outside the configured sample window "
            f"{sample_window_label(raw_config)}: {offending_dates}."
        )
        raise DataValidationError(message)
    surface_frame = (
        gold_scan.filter(sample_window_expr(raw_config, column="quote_date"))
        .collect(engine="streaming")
        .sort(["quote_date", "maturity_index", "moneyness_index"])
    )
    if surface_frame.is_empty():
        message = (
            "No in-window gold surfaces remain after applying the configured sample window "
            f"{sample_window_label(raw_config)}."
        )
        raise DataValidationError(message)
    require_surface_grid_metadata(surface_frame, grid, dataset_name="Stage 04 gold surfaces")

    daily_dataset = build_daily_feature_dataset(
        surface_frame=surface_frame,
        grid=grid,
        feature_config=feature_config,
        calendar_config=calendar_config,
    ).feature_frame
    assert_frame_dates_in_sample_window(
        daily_dataset,
        raw_config,
        context="Stage 04 daily feature dataset",
    )
    assert_frame_dates_in_sample_window(
        daily_dataset,
        raw_config,
        column="target_date",
        context="Stage 04 daily feature dataset",
    )
    write_parquet_frame(daily_dataset, output_path)
    availability_manifest = build_feature_availability_manifest(daily_dataset)
    write_bytes_atomic(
        availability_manifest_path,
        orjson.dumps(availability_manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
    )
    feature_dataset_hash = sha256_file(output_path)

    dates = daily_dataset["quote_date"].to_list()
    if any(not isinstance(value, date) for value in dates):
        message = "daily_features.parquet must contain Polars Date quote_date values."
        raise TypeError(message)
    split_hash = serialize_splits(
        splits=build_walkforward_splits(dates=dates, config=walkforward_config),
        output_path=split_manifest_path,
        walkforward_config=walkforward_config.model_dump(mode="json"),
        sample_window={
            "sample_start_date": raw_config.sample_start_date.isoformat(),
            "sample_end_date": raw_config.sample_end_date.isoformat(),
        },
        date_universe=dates,
        feature_dataset_hash=feature_dataset_hash,
    )
    resumer.mark_complete(
        resume_item_id,
        output_paths=[output_path, availability_manifest_path, split_manifest_path],
        metadata={
            "feature_rows": daily_dataset.height,
            "feature_availability_manifest_path": str(availability_manifest_path),
            "feature_dataset_hash": feature_dataset_hash,
            "split_hash": split_hash,
        },
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
        input_artifact_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            skipped_dates_manifest_path,
            *gold_files,
        ],
        output_artifact_paths=[output_path, availability_manifest_path, split_manifest_path],
        data_manifest_paths=gold_files,
        split_manifest_path=split_manifest_path,
        extra_metadata={
            "feature_rows": daily_dataset.height,
            "feature_dataset_hash": feature_dataset_hash,
            "feature_availability_columns": len(availability_manifest),
            "split_hash": split_hash,
            "gold_input_file_count": len(gold_files),
            "sample_window": sample_window_label(raw_config),
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved feature dataset to {output_path} and split hash {split_hash}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
