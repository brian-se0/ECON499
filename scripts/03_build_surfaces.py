from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.cleaning.option_filters import valid_option_rows
from ivsurf.config import RawDataConfig, SurfaceGridConfig, load_yaml_config
from ivsurf.evaluation.metrics import total_variance_to_iv
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import require_quote_date_in_sample_window, sample_window_label
from ivsurf.qc.schema_checks import assert_required_columns
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.surfaces.aggregation import aggregate_daily_surface
from ivsurf.surfaces.arbitrage_diagnostics import summarize_diagnostics
from ivsurf.surfaces.grid import SurfaceGrid, assign_grid_indices
from ivsurf.surfaces.interpolation import complete_surface

app = typer.Typer(add_completion=False)


def _gold_path(silver_path: Path, raw_config: RawDataConfig) -> Path:
    year_partition = silver_path.parent.name
    output_dir = raw_config.gold_dir / year_partition
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / silver_path.name


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    limit: int | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    silver_files = sorted_artifact_files(raw_config.silver_dir, "year=*/*.parquet")
    if limit is not None:
        silver_files = silver_files[:limit]
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "03_build_surfaces"),
        stage_name="03_build_surfaces",
        context_hash=build_resume_context_hash(
            config_paths=[raw_config_path, surface_config_path],
            input_artifact_paths=silver_files,
            extra_tokens={"artifact_schema_version": 2},
        ),
    )

    summary_rows: list[dict[str, object]] = []
    maturity_years = grid.maturity_years
    with create_progress() as progress:
        for silver_path in iter_with_progress(
            progress,
            silver_files,
            description="Stage 03 constructing gold surfaces",
        ):
            item_id = str(silver_path.resolve())
            if resumer.item_complete(item_id):
                summary_rows.append(resumer.metadata_for(item_id))
                continue
            stale_output_paths = resumer.output_paths_for(item_id)
            if stale_output_paths:
                resumer.clear_item(item_id, output_paths=stale_output_paths)
            silver_frame = pl.read_parquet(silver_path)
            assert_required_columns(
                silver_frame.columns,
                required_columns=(
                    "quote_date",
                    "tau_years",
                    "log_moneyness",
                    "total_variance",
                    "implied_volatility_1545",
                    "vega_1545",
                    "is_valid_observation",
                ),
                dataset_name=str(silver_path),
            )
            quote_date = silver_frame["quote_date"][0]
            if not isinstance(quote_date, date):
                message = f"Unexpected quote_date type in {silver_path}"
                raise TypeError(message)
            require_quote_date_in_sample_window(
                quote_date,
                raw_config,
                context=f"Stage 03 silver artifact {silver_path}",
            )
            valid_frame = valid_option_rows(silver_frame)
            if valid_frame.is_empty():
                summary_row = {
                    "silver_path": str(silver_path),
                    "quote_date": quote_date.isoformat(),
                    "status": "skipped_no_valid_rows",
                    "reason": "NO_VALID_ROWS_AFTER_CLEANING",
                }
                resumer.mark_complete(
                    item_id,
                    output_paths=[],
                    metadata=summary_row,
                )
                summary_rows.append(summary_row)
                continue

            assigned = assign_grid_indices(frame=valid_frame, grid=grid)
            observed = aggregate_daily_surface(
                frame=assigned,
                grid=grid,
                config=surface_config,
            ).sort(["quote_date", "maturity_index", "moneyness_index"])
            observed_matrix = (
                observed["observed_total_variance"]
                .fill_null(np.nan)
                .to_numpy()
                .reshape(grid.shape)
            )
            observed_mask = observed["observed_mask"].to_numpy().reshape(grid.shape)
            completed = complete_surface(
                observed_total_variance=observed_matrix,
                observed_mask=observed_mask,
                maturity_coordinates=maturity_years,
                moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
                interpolation_order=surface_config.interpolation_order,
                interpolation_cycles=surface_config.interpolation_cycles,
                total_variance_floor=surface_config.total_variance_floor,
            )
            diagnostics = summarize_diagnostics(
                completed.completed_total_variance,
                moneyness_points=np.asarray(grid.moneyness_points, dtype=np.float64),
            )
            completed_flat = completed.completed_total_variance.reshape(-1)
            completed_iv = total_variance_to_iv(
                total_variance=completed.completed_total_variance,
                maturity_years=maturity_years[:, None],
            ).reshape(-1)

            output_frame = observed.with_columns(
                pl.Series("completed_total_variance", completed_flat),
                pl.Series("completed_iv", completed_iv),
                pl.lit(diagnostics.calendar_violation_count).alias("calendar_violation_count"),
                pl.lit(diagnostics.calendar_violation_magnitude).alias(
                    "calendar_violation_magnitude"
                ),
                pl.lit(diagnostics.convexity_violation_count).alias("convexity_violation_count"),
                pl.lit(diagnostics.convexity_violation_magnitude).alias(
                    "convexity_violation_magnitude"
                ),
            )
            output_path = _gold_path(silver_path=silver_path, raw_config=raw_config)
            write_parquet_frame(output_frame, output_path)
            summary_row = {
                "gold_path": str(output_path),
                "quote_date": str(output_frame["quote_date"][0]),
                "status": "built",
                "observed_cells": int(output_frame["observed_mask"].sum()),
            }
            resumer.mark_complete(
                item_id,
                output_paths=[output_path],
                metadata=summary_row,
            )
            summary_rows.append(summary_row)

    summary_path = raw_config.manifests_dir / "gold_surface_summary.json"
    skipped_dates_path = raw_config.manifests_dir / "gold_surface_skipped_dates.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(summary_path, orjson.dumps(summary_rows, option=orjson.OPT_INDENT_2))
    skipped_date_rows = [
        {
            "quote_date": row["quote_date"],
            "reason": row["reason"],
            "silver_path": row["silver_path"],
        }
        for row in summary_rows
        if row.get("status") == "skipped_no_valid_rows"
    ]
    write_bytes_atomic(
        skipped_dates_path,
        orjson.dumps(skipped_date_rows, option=orjson.OPT_INDENT_2),
    )
    gold_output_paths = [
        Path(str(row["gold_path"]))
        for row in summary_rows
        if row.get("status") == "built" and "gold_path" in row
    ]
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="03_build_surfaces",
        started_at=started_at,
        config_paths=[raw_config_path, surface_config_path],
        input_artifact_paths=[
            raw_config.manifests_dir / "silver_build_summary.json",
            *silver_files,
        ],
        output_artifact_paths=[summary_path, skipped_dates_path, *gold_output_paths],
        data_manifest_paths=silver_files,
        extra_metadata={
            "limit": limit,
            "silver_files_processed": len(silver_files),
            "gold_files_written": len(gold_output_paths),
            "skipped_dates_count": len(skipped_date_rows),
            "sample_window": sample_window_label(raw_config),
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved gold surface summary to {summary_path}")
    typer.echo(f"Saved skipped-date manifest to {skipped_dates_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()
