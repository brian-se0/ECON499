# Round 003B Prompt: Verify First Batch 1 Fix Slice

You are GPT 5.5 Pro acting as the audit brain. Codex implemented the first fix slice for these findings only:

- B1-CODE-001: mask-false cells could anchor interpolation.
- B1-CODE-002: out-of-grid rows could be silently boundary-bucketed.
- B1-CODE-006: completed-grid cells lacked interpolation/extrapolation provenance.

Do not re-audit the entire project in this verification round. Do not reopen literature archives. Use only the changed code/tests pasted below and the reported local verification output.

Local verification already run by Codex:

```bash
uv run python -m ruff check src/ivsurf/surfaces/grid.py src/ivsurf/surfaces/interpolation.py scripts/03_build_surfaces.py tests/unit/test_grid.py tests/unit/test_interpolation.py tests/integration/test_stage03_stage04_target_gap_alignment.py
# All checks passed.

uv run python -m pytest tests/unit/test_grid.py tests/unit/test_interpolation.py tests/regression/test_observed_mask_preservation.py tests/integration/test_stage03_stage04_target_gap_alignment.py
# 6 passed in 1.60s.
```

Return exactly these sections:

## VERIFICATION_DECISION
For each of B1-CODE-001, B1-CODE-002, and B1-CODE-006, say one of: fixed, partially fixed, not fixed. Include concise evidence.

## REQUIRED_ADJUSTMENTS
List any concrete code/test changes Codex must make before this slice is considered fixed. If none, write `none`.

## NEW_REGRESSION_RISKS
List any new risk introduced by the fixes. If none, write `none`.

## NEXT_FIX_SLICE
Tell Codex which remaining finding(s) should be implemented next and why.

## CHANGED_FILES


### src/ivsurf/surfaces/grid.py

```python
"""Fixed-grid helpers for daily surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivsurf.config import SurfaceGridConfig


@dataclass(frozen=True, slots=True)
class SurfaceGrid:
    """Fixed maturity x moneyness grid."""

    maturity_days: tuple[int, ...]
    moneyness_points: tuple[float, ...]

    @property
    def maturity_years(self) -> np.ndarray:
        return np.asarray(self.maturity_days, dtype=np.float64) / 365.0

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.maturity_days), len(self.moneyness_points))

    @classmethod
    def from_config(cls, config: SurfaceGridConfig) -> SurfaceGrid:
        return cls(maturity_days=config.maturity_days, moneyness_points=config.moneyness_points)


def assign_grid_indices(frame: pl.DataFrame, grid: SurfaceGrid) -> pl.DataFrame:
    """Assign each option row to the nearest fixed grid point."""

    maturity_years = grid.maturity_years
    maturity_edges = np.empty(max(len(grid.maturity_days) - 1, 0), dtype=np.float64)
    if maturity_edges.size > 0:
        maturity_edges[:] = (maturity_years[:-1] + maturity_years[1:]) / 2.0

    money = np.asarray(grid.moneyness_points, dtype=np.float64)
    moneyness_edges = np.empty(max(len(grid.moneyness_points) - 1, 0), dtype=np.float64)
    if moneyness_edges.size > 0:
        moneyness_edges[:] = (money[:-1] + money[1:]) / 2.0

    tau_values = frame["tau_years"].to_numpy()
    log_moneyness_values = frame["log_moneyness"].to_numpy()
    maturity_finite = np.isfinite(tau_values)
    moneyness_finite = np.isfinite(log_moneyness_values)
    inside_maturity_domain = (
        maturity_finite
        & (tau_values >= maturity_years[0])
        & (tau_values <= maturity_years[-1])
    )
    inside_moneyness_domain = (
        moneyness_finite
        & (log_moneyness_values >= money[0])
        & (log_moneyness_values <= money[-1])
    )
    inside_grid_domain = inside_maturity_domain & inside_moneyness_domain

    grid_domain_reason = np.full(frame.height, "VALID_GRID_DOMAIN", dtype=object)
    grid_domain_reason[~maturity_finite & ~moneyness_finite] = (
        "NONFINITE_MATURITY_AND_MONEYNESS_GRID_COORDINATE"
    )
    grid_domain_reason[~maturity_finite & moneyness_finite] = (
        "NONFINITE_MATURITY_GRID_COORDINATE"
    )
    grid_domain_reason[maturity_finite & ~moneyness_finite] = (
        "NONFINITE_MONEYNESS_GRID_COORDINATE"
    )
    grid_domain_reason[
        maturity_finite
        & moneyness_finite
        & ~inside_maturity_domain
        & ~inside_moneyness_domain
    ] = "OUTSIDE_MATURITY_AND_MONEYNESS_GRID_DOMAIN"
    grid_domain_reason[
        maturity_finite
        & moneyness_finite
        & ~inside_maturity_domain
        & inside_moneyness_domain
    ] = "OUTSIDE_MATURITY_GRID_DOMAIN"
    grid_domain_reason[
        maturity_finite
        & moneyness_finite
        & inside_maturity_domain
        & ~inside_moneyness_domain
    ] = "OUTSIDE_MONEYNESS_GRID_DOMAIN"

    maturity_index = np.searchsorted(
        maturity_edges,
        tau_values,
        side="right",
    ).astype(np.int64)
    moneyness_index = np.searchsorted(
        moneyness_edges,
        log_moneyness_values,
        side="right",
    ).astype(np.int64)
    maturity_index_values = [
        int(index) if inside else None
        for index, inside in zip(maturity_index, inside_grid_domain, strict=True)
    ]
    moneyness_index_values = [
        int(index) if inside else None
        for index, inside in zip(moneyness_index, inside_grid_domain, strict=True)
    ]

    return frame.with_columns(
        pl.Series("inside_maturity_grid_domain", inside_maturity_domain),
        pl.Series("inside_moneyness_grid_domain", inside_moneyness_domain),
        pl.Series("inside_grid_domain", inside_grid_domain),
        pl.Series("grid_domain_reason", grid_domain_reason),
        pl.Series("maturity_index", maturity_index_values, dtype=pl.Int64),
        pl.Series("moneyness_index", moneyness_index_values, dtype=pl.Int64),
    )

```

### src/ivsurf/surfaces/interpolation.py

```python
"""Deterministic sequential axis-wise surface completion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from ivsurf.exceptions import InterpolationError

COMPLETION_STATUS_OBSERVED = "observed"
COMPLETION_STATUS_INTERPOLATED = "interpolated"
COMPLETION_STATUS_EXTRAPOLATED = "extrapolated_boundary_fill"
COMPLETION_STATUS_MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CompletedSurface:
    """Completed daily surface with mask information."""

    completed_total_variance: np.ndarray
    observed_mask: np.ndarray
    completion_status: np.ndarray


def _fill_axis(
    values: np.ndarray,
    coordinates: np.ndarray,
    completion_status: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    status = completion_status.copy()
    finite_mask = np.isfinite(values)
    count = int(finite_mask.sum())
    if count == 0:
        return result, status
    missing_positions = np.flatnonzero(~finite_mask)
    if missing_positions.size == 0:
        return result, status
    if count == 1:
        result[missing_positions] = values[finite_mask][0]
        status[missing_positions] = COMPLETION_STATUS_EXTRAPOLATED
        return result, status

    observed_x = coordinates[finite_mask]
    observed_y = values[finite_mask]
    interpolator = PchipInterpolator(observed_x, observed_y, extrapolate=False)

    target_x = coordinates[missing_positions]
    predicted = interpolator(target_x)
    extrapolated = (target_x < observed_x.min()) | (target_x > observed_x.max())
    predicted = np.where(
        target_x < observed_x.min(),
        observed_y[0],
        np.where(target_x > observed_x.max(), observed_y[-1], predicted),
    )
    result[missing_positions] = predicted
    status[missing_positions] = np.where(
        extrapolated,
        COMPLETION_STATUS_EXTRAPOLATED,
        COMPLETION_STATUS_INTERPOLATED,
    )
    return result, status


def complete_surface(
    observed_total_variance: np.ndarray,
    observed_mask: np.ndarray,
    maturity_coordinates: np.ndarray,
    moneyness_coordinates: np.ndarray,
    interpolation_order: tuple[str, ...],
    interpolation_cycles: int,
    total_variance_floor: float,
) -> CompletedSurface:
    """Complete a surface by fixed-order sequential one-dimensional interpolation."""

    completed = observed_total_variance.astype(np.float64, copy=True)
    normalized_observed_mask = np.asarray(observed_mask, dtype=bool)
    if normalized_observed_mask.shape != completed.shape:
        message = (
            "observed_mask must have the same shape as observed_total_variance, "
            f"found {normalized_observed_mask.shape} != {completed.shape}."
        )
        raise ValueError(message)
    completed[~normalized_observed_mask] = np.nan
    completion_status = np.full(
        completed.shape,
        COMPLETION_STATUS_MISSING,
        dtype=object,
    )
    completion_status[normalized_observed_mask] = COMPLETION_STATUS_OBSERVED

    for _ in range(interpolation_cycles):
        for axis_name in interpolation_order:
            if axis_name == "maturity":
                for money_idx in range(completed.shape[1]):
                    completed[:, money_idx], completion_status[:, money_idx] = _fill_axis(
                        completed[:, money_idx],
                        maturity_coordinates,
                        completion_status[:, money_idx],
                    )
            elif axis_name == "moneyness":
                for maturity_idx in range(completed.shape[0]):
                    completed[maturity_idx, :], completion_status[maturity_idx, :] = _fill_axis(
                        completed[maturity_idx, :],
                        moneyness_coordinates,
                        completion_status[maturity_idx, :],
                    )
            else:
                message = f"Unsupported interpolation axis: {axis_name}"
                raise ValueError(message)

    if not np.isfinite(completed).all():
        message = (
            "Surface completion left NaN or infinite values "
            "after deterministic interpolation."
        )
        raise InterpolationError(message)

    completed = np.maximum(completed, total_variance_floor)
    return CompletedSurface(
        completed_total_variance=completed,
        observed_mask=normalized_observed_mask,
        completion_status=completion_status,
    )

```

### scripts/03_build_surfaces.py

```python
from __future__ import annotations

from collections import Counter
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


def _reason_counts(frame: pl.DataFrame, column_name: str) -> dict[str, int]:
    return {
        str(reason): int(count)
        for reason, count in Counter(frame[column_name].to_list()).items()
    }


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
            grid_domain_reason_counts = _reason_counts(assigned, "grid_domain_reason")
            assigned = assigned.filter(pl.col("inside_grid_domain"))
            if assigned.is_empty():
                summary_row = {
                    "silver_path": str(silver_path),
                    "quote_date": quote_date.isoformat(),
                    "status": "skipped_no_rows_inside_grid_domain",
                    "reason": "NO_ROWS_INSIDE_GRID_DOMAIN",
                    "valid_rows": valid_frame.height,
                    "rows_inside_grid_domain": 0,
                    "rows_outside_grid_domain": valid_frame.height,
                    "grid_domain_reason_counts": grid_domain_reason_counts,
                }
                resumer.mark_complete(
                    item_id,
                    output_paths=[],
                    metadata=summary_row,
                )
                summary_rows.append(summary_row)
                continue
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
            completion_status_flat = completed.completion_status.reshape(-1)
            completed_iv = total_variance_to_iv(
                total_variance=completed.completed_total_variance,
                maturity_years=maturity_years[:, None],
            ).reshape(-1)

            output_frame = observed.with_columns(
                pl.Series("completed_total_variance", completed_flat),
                pl.Series("completed_iv", completed_iv),
                pl.Series("completion_status", completion_status_flat),
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
                "valid_rows": valid_frame.height,
                "rows_inside_grid_domain": assigned.height,
                "rows_outside_grid_domain": valid_frame.height - assigned.height,
                "grid_domain_reason_counts": grid_domain_reason_counts,
                "completion_status_counts": _reason_counts(output_frame, "completion_status"),
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
        if str(row.get("status", "")).startswith("skipped")
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

```

### tests/unit/test_grid.py

```python
from __future__ import annotations

import polars as pl

from ivsurf.surfaces.grid import SurfaceGrid, assign_grid_indices


def test_assign_grid_indices_flags_outside_grid_domain_without_boundary_bucket() -> None:
    grid = SurfaceGrid(maturity_days=(30, 60), moneyness_points=(-0.1, 0.0, 0.1))
    frame = pl.DataFrame(
        {
            "tau_years": [30.0 / 365.0, 90.0 / 365.0, 30.0 / 365.0],
            "log_moneyness": [0.0, 0.0, 0.2],
        }
    )

    assigned = assign_grid_indices(frame=frame, grid=grid)

    assert assigned["inside_grid_domain"].to_list() == [True, False, False]
    assert assigned["grid_domain_reason"].to_list() == [
        "VALID_GRID_DOMAIN",
        "OUTSIDE_MATURITY_GRID_DOMAIN",
        "OUTSIDE_MONEYNESS_GRID_DOMAIN",
    ]
    assert assigned["maturity_index"].to_list() == [0, None, None]
    assert assigned["moneyness_index"].to_list() == [1, None, None]

```

### tests/unit/test_interpolation.py

```python
from __future__ import annotations

import numpy as np

from ivsurf.surfaces.interpolation import (
    COMPLETION_STATUS_EXTRAPOLATED,
    COMPLETION_STATUS_INTERPOLATED,
    COMPLETION_STATUS_OBSERVED,
    complete_surface,
)


def test_surface_completion_fills_all_cells() -> None:
    observed = np.asarray([[0.04, np.nan, 0.09], [np.nan, 0.07, np.nan]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=np.isfinite(observed),
        maturity_coordinates=np.asarray([7.0 / 365.0, 30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.1, 0.0, 0.1]),
        interpolation_order=("maturity", "moneyness"),
        interpolation_cycles=2,
        total_variance_floor=1.0e-8,
    )
    assert np.isfinite(completed.completed_total_variance).all()
    assert completed.completed_total_variance[0, 0] == observed[0, 0]


def test_surface_completion_ignores_finite_values_without_observed_mask() -> None:
    observed = np.asarray([[0.01, 999.0, 0.03]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=np.asarray([[True, False, True]]),
        maturity_coordinates=np.asarray([30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.1, 0.0, 0.1]),
        interpolation_order=("moneyness",),
        interpolation_cycles=1,
        total_variance_floor=1.0e-8,
    )

    np.testing.assert_allclose(completed.completed_total_variance[0], [0.01, 0.02, 0.03])
    assert bool(completed.observed_mask[0, 1]) is False


def test_surface_completion_marks_interpolation_and_extrapolation_status() -> None:
    observed = np.asarray([[np.nan, 0.02, np.nan, 0.04, np.nan]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=np.asarray([[False, True, False, True, False]]),
        maturity_coordinates=np.asarray([30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2]),
        interpolation_order=("moneyness",),
        interpolation_cycles=1,
        total_variance_floor=1.0e-8,
    )

    np.testing.assert_allclose(
        completed.completed_total_variance[0],
        [0.02, 0.02, 0.03, 0.04, 0.04],
    )
    assert completed.completion_status[0].tolist() == [
        COMPLETION_STATUS_EXTRAPOLATED,
        COMPLETION_STATUS_OBSERVED,
        COMPLETION_STATUS_INTERPOLATED,
        COMPLETION_STATUS_OBSERVED,
        COMPLETION_STATUS_EXTRAPOLATED,
    ]

```

### tests/integration/test_stage03_stage04_target_gap_alignment.py

```python
from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import polars as pl


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _silver_rows(
    quote_date: date,
    *,
    total_variance: float,
    is_valid: bool,
    include_outside_grid_row: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "quote_date": quote_date,
            "tau_years": 30.0 / 365.0,
            "log_moneyness": 0.0,
            "total_variance": total_variance,
            "implied_volatility_1545": float((total_variance / (30.0 / 365.0)) ** 0.5),
            "vega_1545": 1.0,
            "spread_1545": 0.01,
            "is_valid_observation": is_valid,
        }
    ]
    if include_outside_grid_row:
        rows.append(
            {
                "quote_date": quote_date,
                "tau_years": 30.0 / 365.0,
                "log_moneyness": 0.2,
                "total_variance": total_variance * 100.0,
                "implied_volatility_1545": float(
                    ((total_variance * 100.0) / (30.0 / 365.0)) ** 0.5
                ),
                "vega_1545": 1.0,
                "spread_1545": 0.01,
                "is_valid_observation": is_valid,
            }
        )
    return rows


def test_stage03_and_stage04_preserve_skipped_day_gaps_and_gold_input_provenance(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    silver_year = tmp_path / "data" / "silver" / "year=2021"
    silver_year.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)

    dates_and_validity = [
        (date(2021, 1, 4), True),
        (date(2021, 1, 5), True),
        (date(2021, 1, 6), False),
        (date(2021, 1, 7), True),
        (date(2021, 1, 8), True),
    ]
    silver_summary: list[dict[str, object]] = []
    for offset, (quote_date_value, is_valid) in enumerate(dates_and_validity, start=1):
        silver_path = silver_year / f"{quote_date_value.isoformat()}.parquet"
        rows = _silver_rows(
            quote_date_value,
            total_variance=0.001 * offset,
            is_valid=is_valid,
            include_outside_grid_row=quote_date_value == date(2021, 1, 4),
        )
        pl.DataFrame(rows).write_parquet(silver_path)
        silver_summary.append(
            {
                "silver_path": str(silver_path),
                "quote_date": quote_date_value.isoformat(),
                "status": "built",
                "rows": len(rows),
                "valid_rows": sum(1 for row in rows if row["is_valid_observation"]),
            }
        )
    (manifests_dir / "silver_build_summary.json").write_bytes(
        orjson.dumps(silver_summary, option=orjson.OPT_INDENT_2)
    )

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-08"\n'
        ),
    )
    surface_config_path = _write_yaml(
        tmp_path / "surface.yaml",
        (
            "moneyness_points: [0.0]\n"
            "maturity_days: [30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    feature_config_path = _write_yaml(
        tmp_path / "features.yaml",
        (
            "lag_windows: [1]\n"
            "include_daily_change: false\n"
            "include_mask: true\n"
            "include_liquidity: false\n"
        ),
    )
    walkforward_config_path = _write_yaml(
        tmp_path / "walkforward.yaml",
        (
            "train_size: 1\n"
            "validation_size: 1\n"
            "test_size: 1\n"
            "step_size: 1\n"
            "expanding_train: true\n"
        ),
    )

    stage03 = _load_script_module(
        repo_root / "scripts" / "03_build_surfaces.py",
        "stage03_target_gap_alignment",
    )
    stage04 = _load_script_module(
        repo_root / "scripts" / "04_build_features.py",
        "stage04_target_gap_alignment",
    )

    stage03.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
    )

    skipped_payload = orjson.loads((manifests_dir / "gold_surface_skipped_dates.json").read_bytes())
    assert skipped_payload == [
        {
            "quote_date": "2021-01-06",
            "reason": "NO_VALID_ROWS_AFTER_CLEANING",
            "silver_path": str(silver_year / "2021-01-06.parquet"),
        }
    ]

    stage03_run_manifest = sorted((manifests_dir / "runs" / "03_build_surfaces").glob("*.json"))[-1]
    stage03_manifest_payload = orjson.loads(stage03_run_manifest.read_bytes())
    stage03_output_paths = {
        artifact["path"] for artifact in stage03_manifest_payload["output_artifacts"]
    }
    assert (
        str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage03_output_paths
    )
    gold_surface_summary = orjson.loads((manifests_dir / "gold_surface_summary.json").read_bytes())
    first_built_row = next(row for row in gold_surface_summary if row["quote_date"] == "2021-01-04")
    assert first_built_row["rows_inside_grid_domain"] == 1
    assert first_built_row["rows_outside_grid_domain"] == 1
    assert first_built_row["grid_domain_reason_counts"] == {
        "VALID_GRID_DOMAIN": 1,
        "OUTSIDE_MONEYNESS_GRID_DOMAIN": 1,
    }
    assert first_built_row["completion_status_counts"] == {"observed": 1}
    first_gold_surface = pl.read_parquet(
        tmp_path / "data" / "gold" / "year=2021" / "2021-01-04.parquet"
    )
    assert first_gold_surface["completion_status"].to_list() == ["observed"]

    stage04.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        feature_config_path=feature_config_path,
        walkforward_config_path=walkforward_config_path,
    )

    feature_frame = pl.read_parquet(tmp_path / "data" / "gold" / "daily_features.parquet")
    target_gap_row = feature_frame.filter(pl.col("quote_date") == date(2021, 1, 5))
    assert target_gap_row.height == 1
    assert target_gap_row["target_date"].to_list() == [date(2021, 1, 7)]
    assert target_gap_row["target_gap_sessions"].to_list() == [1]

    gold_files = sorted((tmp_path / "data" / "gold").glob("year=*/*.parquet"))
    stage04_run_manifest = sorted((manifests_dir / "runs" / "04_build_features").glob("*.json"))[-1]
    stage04_manifest_payload = orjson.loads(stage04_run_manifest.read_bytes())
    stage04_input_paths = {
        artifact["path"] for artifact in stage04_manifest_payload["input_artifacts"]
    }
    assert str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage04_input_paths
    for gold_file in gold_files:
        assert str(gold_file.resolve()) in stage04_input_paths

```