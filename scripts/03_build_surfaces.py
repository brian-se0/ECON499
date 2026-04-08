from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.cleaning.option_filters import valid_option_rows
from ivsurf.config import RawDataConfig, SurfaceGridConfig, load_yaml_config
from ivsurf.evaluation.metrics import total_variance_to_iv
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
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    silver_files = sorted(raw_config.silver_dir.glob("year=*/*.parquet"))
    if limit is not None:
        silver_files = silver_files[:limit]

    summary_rows: list[dict[str, object]] = []
    maturity_years = grid.maturity_years
    for silver_path in silver_files:
        silver_frame = pl.read_parquet(silver_path)
        valid_frame = valid_option_rows(silver_frame)
        if valid_frame.is_empty():
            summary_rows.append(
                {
                    "silver_path": str(silver_path),
                    "status": "skipped_no_valid_rows",
                }
            )
            continue

        assigned = assign_grid_indices(frame=valid_frame, grid=grid)
        observed = aggregate_daily_surface(frame=assigned, grid=grid, config=surface_config).sort(
            ["quote_date", "maturity_index", "moneyness_index"]
        )
        observed_matrix = (
            observed["observed_total_variance"]
            .fill_null(np.nan)
            .to_numpy()
            .reshape(grid.shape)
        )
        completed = complete_surface(
            observed_total_variance=observed_matrix,
            maturity_coordinates=maturity_years,
            moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
            interpolation_order=surface_config.interpolation_order,
            interpolation_cycles=surface_config.interpolation_cycles,
            total_variance_floor=surface_config.total_variance_floor,
        )
        diagnostics = summarize_diagnostics(completed.completed_total_variance)
        completed_flat = completed.completed_total_variance.reshape(-1)
        completed_iv = total_variance_to_iv(
            total_variance=completed.completed_total_variance,
            maturity_years=maturity_years[:, None],
        ).reshape(-1)

        output_frame = observed.with_columns(
            pl.Series("completed_total_variance", completed_flat),
            pl.Series("completed_iv", completed_iv),
            pl.lit(diagnostics.calendar_violation_count).alias("calendar_violation_count"),
            pl.lit(diagnostics.calendar_violation_magnitude).alias("calendar_violation_magnitude"),
            pl.lit(diagnostics.convexity_violation_count).alias("convexity_violation_count"),
            pl.lit(diagnostics.convexity_violation_magnitude).alias("convexity_violation_magnitude"),
        )
        output_path = _gold_path(silver_path=silver_path, raw_config=raw_config)
        output_frame.write_parquet(output_path, compression="zstd", statistics=True)
        summary_rows.append(
            {
                "gold_path": str(output_path),
                "quote_date": str(output_frame["quote_date"][0]),
                "observed_cells": int(output_frame["observed_mask"].sum()),
            }
        )

    summary_path = raw_config.manifests_dir / "gold_surface_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(orjson.dumps(summary_rows, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    app()
