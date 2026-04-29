# Round 003C Prompt: Verify B1-CODE-006 Propagation Adjustment

You are GPT 5.5 Pro acting as the audit brain. This is a narrow verification follow-up for B1-CODE-006 only. In Round 003B you said B1-CODE-006 was partially fixed and required downstream propagation of `completion_status` into evaluation/loss artifacts plus stronger tests.

Codex has now implemented that adjustment. Do not re-audit unrelated findings. Use only the pasted code/tests and local verification output.

Local verification run by Codex:

```bash
uv run python -m ruff check src/ivsurf/evaluation/alignment.py src/ivsurf/evaluation/loss_panels.py src/ivsurf/surfaces/grid.py src/ivsurf/surfaces/interpolation.py scripts/03_build_surfaces.py tests/unit/test_alignment.py tests/unit/test_stats.py tests/unit/test_tuning_workflow.py tests/unit/test_grid.py tests/unit/test_interpolation.py tests/integration/test_stage03_stage04_target_gap_alignment.py
# All checks passed.

uv run python -m pytest tests/unit/test_alignment.py tests/unit/test_stats.py tests/unit/test_tuning_workflow.py tests/unit/test_grid.py tests/unit/test_interpolation.py tests/regression/test_observed_mask_preservation.py tests/integration/test_stage03_stage04_target_gap_alignment.py
# 20 passed in 2.20s.
```

Return exactly these sections:

## VERIFICATION_DECISION
Say whether B1-CODE-006 is now fixed, partially fixed, or not fixed. Include concise evidence.

## REQUIRED_ADJUSTMENTS
If any remain for B1-CODE-006, list concrete changes. If none, write `none`.

## NEW_REGRESSION_RISKS
List any new risk introduced by this adjustment. If none, write `none`.

## NEXT_FIX_SLICE
If B1-CODE-006 is fixed, confirm the next slice should be B1-CODE-007/B1-CODE-010 or revise that recommendation.

## CHANGED_FILES


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

### src/ivsurf/evaluation/alignment.py

```python
"""Forecast-artifact alignment against realized surfaces and spot states."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import total_variance_to_iv, validate_total_variance_array
from ivsurf.io.parquet import scan_parquet_files
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.surfaces.interpolation import (
    COMPLETION_STATUS_EXTRAPOLATED,
    COMPLETION_STATUS_INTERPOLATED,
)


def _require_files(paths: list[Path], description: str) -> None:
    if not paths:
        message = f"No {description} files found."
        raise FileNotFoundError(message)


def _require_non_null_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        if frame[column].null_count() > 0:
            message = f"Aligned evaluation panel contains nulls in required column {column}."
            raise ValueError(message)


def _require_finite_float_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            message = f"Aligned evaluation panel contains non-finite values in column {column}."
            raise ValueError(message)


def _require_non_negative_float_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy().astype(np.float64, copy=False)
        validate_total_variance_array(
            values,
            context=f"Aligned evaluation panel column {column}",
            allow_zero=True,
        )


def _format_spot_contract_violations(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
) -> str:
    violations: list[str] = []
    for row in frame.select("quote_date", *columns).iter_rows(named=True):
        quote_date = cast(date, row["quote_date"])
        details = ", ".join(f"{column}={row[column]!r}" for column in columns)
        violations.append(f"{quote_date.isoformat()} ({details})")
    return "; ".join(violations)


def load_actual_surface_frame(gold_dir: Path) -> pl.DataFrame:
    """Load persisted daily surface artifacts."""

    gold_files = sorted_artifact_files(gold_dir, "year=*/*.parquet")
    _require_files(gold_files, "gold surface")
    return (
        scan_parquet_files(gold_files)
        .select(
            "quote_date",
            "maturity_index",
            "maturity_days",
            "moneyness_index",
            "moneyness_point",
            "observed_total_variance",
            "observed_iv",
            "completed_total_variance",
            "completed_iv",
            "observed_mask",
            "completion_status",
            "vega_sum",
        )
        .collect(engine="streaming")
        .sort(["quote_date", "maturity_index", "moneyness_index"])
    )


def load_forecast_frame(forecast_dir: Path) -> pl.DataFrame:
    """Load persisted forecast artifacts."""

    forecast_files = sorted_artifact_files(forecast_dir, "*.parquet")
    _require_files(forecast_files, "forecast")
    return (
        scan_parquet_files(forecast_files)
        .collect(engine="streaming")
        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
    )


def load_daily_spot_frame(silver_dir: Path) -> pl.DataFrame:
    """Load the official stage-08 daily spot from valid active_underlying_price_1545 rows."""

    silver_files = sorted_artifact_files(silver_dir, "year=*/*.parquet")
    _require_files(silver_files, "silver")
    lazy_frame = scan_parquet_files(silver_files)
    spot_frame = (
        lazy_frame.select("quote_date", "active_underlying_price_1545", "is_valid_observation")
        .filter(pl.col("is_valid_observation"))
        .group_by("quote_date")
        .agg(
            pl.len().alias("valid_spot_row_count"),
            pl.col("active_underlying_price_1545").n_unique().alias("active_spot_n_unique"),
            pl.col("active_underlying_price_1545").median().alias("spot_1545"),
            pl.col("active_underlying_price_1545").min().alias("active_spot_min"),
            pl.col("active_underlying_price_1545").max().alias("active_spot_max"),
        )
        .collect(engine="streaming")
        .sort("quote_date")
    )
    if spot_frame.height == 0:
        message = "No valid silver rows available to derive stage-08 daily spot states."
        raise ValueError(message)
    invalid_spot_frames = [
        spot_frame.filter(pl.col("valid_spot_row_count") <= 0),
        spot_frame.filter(
            pl.col("spot_1545").is_null()
            | (~pl.col("spot_1545").is_finite())
            | (pl.col("spot_1545") <= 0.0)
        ),
    ]
    invalid_spots = pl.concat(invalid_spot_frames).unique(subset=["quote_date"]).sort("quote_date")
    if invalid_spots.height > 0:
        message = (
            "Expected strictly positive finite stage-08 daily spot values derived from the "
            "median active_underlying_price_1545 across valid silver rows. Violations: "
            f"{_format_spot_contract_violations(invalid_spots, columns=('spot_1545',))}."
        )
        raise ValueError(message)
    return spot_frame.select("quote_date", "spot_1545")


def assert_forecast_origins_after_hpo_boundary(
    forecast_frame: pl.DataFrame,
    *,
    max_hpo_validation_date: date,
) -> None:
    """Fail fast if forecast artifacts include HPO-contaminated origin dates."""

    contaminated = forecast_frame.filter(pl.col("quote_date") <= pl.lit(max_hpo_validation_date))
    if contaminated.is_empty():
        return
    earliest_quote_date = cast(date, contaminated["quote_date"].min())
    latest_quote_date = cast(date, contaminated["quote_date"].max())
    earliest_target_date = cast(date, contaminated["target_date"].min())
    latest_target_date = cast(date, contaminated["target_date"].max())
    message = (
        "Forecast artifacts include quote_date values that were inside the HPO-used validation "
        f"sample. Found {contaminated.height} contaminated rows with quote_date range "
        f"[{earliest_quote_date.isoformat()}, {latest_quote_date.isoformat()}], "
        f"target_date range "
        f"[{earliest_target_date.isoformat()}, {latest_target_date.isoformat()}], and boundary "
        f"{max_hpo_validation_date.isoformat()}."
    )
    raise ValueError(message)


def build_forecast_realization_panel(
    actual_surface_frame: pl.DataFrame,
    forecast_frame: pl.DataFrame,
    *,
    total_variance_floor: float,
) -> pl.DataFrame:
    """Align forecast artifacts with realized target-day surfaces and origin-day references."""

    actual_target = actual_surface_frame.rename(
        {
            "quote_date": "target_date",
            "observed_total_variance": "actual_observed_total_variance",
            "observed_iv": "actual_observed_iv",
            "completed_total_variance": "actual_completed_total_variance",
            "completed_iv": "actual_completed_iv",
            "observed_mask": "actual_observed_mask",
            "completion_status": "actual_completion_status",
            "vega_sum": "actual_vega_sum",
        }
    )
    origin_surface = actual_surface_frame.rename(
        {
            "completed_total_variance": "origin_completed_total_variance",
            "completed_iv": "origin_completed_iv",
            "completion_status": "origin_completion_status",
        }
    ).select(
        "quote_date",
        "maturity_index",
        "moneyness_index",
        "origin_completed_total_variance",
        "origin_completed_iv",
        "origin_completion_status",
    )

    joined_panel = (
        forecast_frame.join(
            actual_target,
            on=["target_date", "maturity_index", "moneyness_index"],
            how="left",
            validate="m:1",
        )
        .join(
            origin_surface,
            on=["quote_date", "maturity_index", "moneyness_index"],
            how="left",
            validate="m:1",
        )
    )

    _require_non_null_columns(
        joined_panel,
        columns=(
            "actual_completed_total_variance",
            "actual_completed_iv",
            "origin_completed_iv",
            "actual_completion_status",
            "origin_completion_status",
            "predicted_total_variance",
        ),
    )
    _require_finite_float_columns(joined_panel, columns=("predicted_total_variance",))
    _require_non_negative_float_columns(joined_panel, columns=("predicted_total_variance",))

    panel = (
        panel_with_completed_iv(
            joined_panel,
            maturity_days_column="maturity_days",
            total_variance_column="predicted_total_variance",
            output_iv_column="predicted_iv",
            total_variance_floor=total_variance_floor,
        )
        .with_columns(
            (
                pl.col("actual_completed_iv") - pl.col("origin_completed_iv")
            ).alias("actual_iv_change"),
            (
                pl.col("predicted_iv") - pl.col("origin_completed_iv")
            ).alias("predicted_iv_change"),
            # Completed surfaces are the forecast target, while observed-mask x vega defines
            # the official observed-cell evaluation slice used for headline metrics.
            pl.when(pl.col("actual_observed_mask"))
            .then(pl.col("actual_vega_sum"))
            .otherwise(0.0)
            .alias("observed_weight"),
            pl.when(pl.col("actual_completion_status") == COMPLETION_STATUS_INTERPOLATED)
            .then(1.0)
            .otherwise(0.0)
            .alias("interpolated_weight"),
            pl.when(pl.col("actual_completion_status") == COMPLETION_STATUS_EXTRAPOLATED)
            .then(1.0)
            .otherwise(0.0)
            .alias("extrapolated_weight"),
            pl.lit(1.0).alias("full_grid_weight"),
        )
        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
    )

    _require_non_null_columns(
        panel,
        columns=(
            "predicted_iv",
            "predicted_iv_change",
        ),
    )
    _require_finite_float_columns(
        panel,
        columns=("predicted_total_variance", "predicted_iv", "predicted_iv_change"),
    )
    return panel


def panel_with_completed_iv(
    frame: pl.DataFrame,
    maturity_days_column: str,
    total_variance_column: str,
    output_iv_column: str,
    *,
    total_variance_floor: float,
) -> pl.DataFrame:
    """Add an IV column derived from total variance and maturity days."""

    maturity_years = (
        frame[maturity_days_column].to_numpy().astype(np.float64) / 365.0
    ).reshape(-1, 1)
    total_variance = validate_total_variance_array(
        frame[total_variance_column].to_numpy().astype(np.float64),
        context=f"Aligned panel column {total_variance_column}",
        allow_zero=True,
    ).reshape(-1, 1)
    iv = total_variance_to_iv(
        total_variance=total_variance,
        maturity_years=maturity_years,
        total_variance_floor=total_variance_floor,
    ).reshape(-1)
    return frame.with_columns(pl.Series(output_iv_column, iv))

```

### src/ivsurf/evaluation/loss_panels.py

```python
"""Daily loss-panel construction from aligned forecast artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import qlike, weighted_mae, weighted_mse, weighted_rmse
from ivsurf.surfaces.interpolation import (
    COMPLETION_STATUS_EXTRAPOLATED,
    COMPLETION_STATUS_INTERPOLATED,
)


@dataclass(frozen=True, slots=True)
class DailyLossRow:
    """Daily aggregated loss row for one model and one target date."""

    model_name: str
    quote_date: str
    target_date: str
    observed_wrmse_total_variance: float
    observed_wmae_total_variance: float
    observed_mse_total_variance: float
    observed_wrmse_iv: float
    observed_wmae_iv: float
    observed_mse_iv_change: float
    observed_qlike_total_variance: float
    interpolated_cell_count: int
    extrapolated_cell_count: int
    full_wrmse_total_variance: float
    full_wmae_total_variance: float
    full_mse_total_variance: float
    full_wrmse_iv: float
    full_wmae_iv: float
    full_mse_iv_change: float
    full_qlike_total_variance: float
    observed_cell_count: int
    full_grid_cell_count: int


def _full_grid_weights(frame: pl.DataFrame, weighting: str) -> np.ndarray:
    if weighting == "uniform":
        return np.ones(frame.height, dtype=np.float64)
    message = f"Unsupported full-grid weighting mode: {weighting}"
    raise ValueError(message)


def daily_loss_metric_values(
    *,
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_masks: np.ndarray,
    vega_weights: np.ndarray,
    positive_floor: float,
) -> np.ndarray:
    """Return one daily loss value per validation surface using Stage-07 semantics."""

    if y_true.shape != y_pred.shape:
        message = (
            "Validation daily-loss inputs must share the same shape for y_true and y_pred, "
            f"found {y_true.shape!r} != {y_pred.shape!r}."
        )
        raise ValueError(message)
    if observed_masks.shape != y_true.shape or vega_weights.shape != y_true.shape:
        message = (
            "Validation daily-loss inputs must align across y_true, observed_masks, "
            f"and vega_weights, found {y_true.shape!r}, {observed_masks.shape!r}, "
            f"{vega_weights.shape!r}."
        )
        raise ValueError(message)

    rows: list[float] = []
    for row_index in range(y_true.shape[0]):
        actual_row = np.asarray(y_true[row_index], dtype=np.float64)
        predicted_row = np.asarray(y_pred[row_index], dtype=np.float64)
        observed_mask_row = np.asarray(observed_masks[row_index], dtype=np.float64) > 0.5
        observed_weights = np.asarray(
            np.maximum(vega_weights[row_index], 0.0),
            dtype=np.float64,
        )[observed_mask_row]

        match metric_name:
            case "observed_wrmse_total_variance":
                value = weighted_rmse(
                    y_true=actual_row[observed_mask_row],
                    y_pred=predicted_row[observed_mask_row],
                    weights=observed_weights,
                )
            case "observed_wmae_total_variance":
                value = weighted_mae(
                    y_true=actual_row[observed_mask_row],
                    y_pred=predicted_row[observed_mask_row],
                    weights=observed_weights,
                )
            case "observed_mse_total_variance":
                value = weighted_mse(
                    y_true=actual_row[observed_mask_row],
                    y_pred=predicted_row[observed_mask_row],
                    weights=observed_weights,
                )
            case "observed_qlike_total_variance":
                value = qlike(
                    y_true=actual_row[observed_mask_row],
                    y_pred=predicted_row[observed_mask_row],
                    positive_floor=positive_floor,
                )
            case "full_wrmse_total_variance":
                value = weighted_rmse(
                    y_true=actual_row,
                    y_pred=predicted_row,
                    weights=np.ones(actual_row.shape[0], dtype=np.float64),
                )
            case "full_wmae_total_variance":
                value = weighted_mae(
                    y_true=actual_row,
                    y_pred=predicted_row,
                    weights=np.ones(actual_row.shape[0], dtype=np.float64),
                )
            case "full_mse_total_variance":
                value = weighted_mse(
                    y_true=actual_row,
                    y_pred=predicted_row,
                    weights=np.ones(actual_row.shape[0], dtype=np.float64),
                )
            case "full_qlike_total_variance":
                value = qlike(
                    y_true=actual_row,
                    y_pred=predicted_row,
                    positive_floor=positive_floor,
                )
            case _:
                message = (
                    "Stage 05 supports only total-variance daily loss metrics for validation, "
                    f"found {metric_name!r}."
                )
                raise ValueError(message)

        rows.append(float(value))

    return np.asarray(rows, dtype=np.float64)


def mean_daily_loss_metric(
    *,
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_masks: np.ndarray,
    vega_weights: np.ndarray,
    positive_floor: float,
) -> float:
    """Return the mean validation-day loss using Stage-07 daily aggregation semantics."""

    values = daily_loss_metric_values(
        metric_name=metric_name,
        y_true=y_true,
        y_pred=y_pred,
        observed_masks=observed_masks,
        vega_weights=vega_weights,
        positive_floor=positive_floor,
    )
    return float(np.mean(values))


def build_daily_loss_frame(
    panel: pl.DataFrame,
    positive_floor: float,
    full_grid_weighting: str,
) -> pl.DataFrame:
    """Aggregate per-cell aligned panel rows into daily model losses."""

    rows: list[DailyLossRow] = []
    grouped = panel.partition_by(["model_name", "quote_date", "target_date"], maintain_order=True)
    for group in grouped:
        observed_group = group.filter(pl.col("actual_observed_mask"))
        interpolated_group = group.filter(
            pl.col("actual_completion_status") == COMPLETION_STATUS_INTERPOLATED
        )
        extrapolated_group = group.filter(
            pl.col("actual_completion_status") == COMPLETION_STATUS_EXTRAPOLATED
        )
        observed_weights = observed_group["observed_weight"].to_numpy().astype(np.float64)
        full_weights = _full_grid_weights(group, full_grid_weighting)
        rows.append(
            DailyLossRow(
                model_name=str(group["model_name"][0]),
                quote_date=str(group["quote_date"][0]),
                target_date=str(group["target_date"][0]),
                observed_wrmse_total_variance=weighted_rmse(
                    y_true=observed_group["actual_completed_total_variance"].to_numpy(),
                    y_pred=observed_group["predicted_total_variance"].to_numpy(),
                    weights=observed_weights,
                )
                if observed_group.height > 0
                else float("nan"),
                observed_wmae_total_variance=weighted_mae(
                    y_true=observed_group["actual_completed_total_variance"].to_numpy(),
                    y_pred=observed_group["predicted_total_variance"].to_numpy(),
                    weights=observed_weights,
                )
                if observed_group.height > 0
                else float("nan"),
                observed_mse_total_variance=weighted_mse(
                    y_true=observed_group["actual_completed_total_variance"].to_numpy(),
                    y_pred=observed_group["predicted_total_variance"].to_numpy(),
                    weights=observed_weights,
                )
                if observed_group.height > 0
                else float("nan"),
                observed_wrmse_iv=weighted_rmse(
                    y_true=observed_group["actual_completed_iv"].to_numpy(),
                    y_pred=observed_group["predicted_iv"].to_numpy(),
                    weights=observed_weights,
                )
                if observed_group.height > 0
                else float("nan"),
                observed_wmae_iv=weighted_mae(
                    y_true=observed_group["actual_completed_iv"].to_numpy(),
                    y_pred=observed_group["predicted_iv"].to_numpy(),
                    weights=observed_weights,
                )
                if observed_group.height > 0
                else float("nan"),
                observed_mse_iv_change=weighted_mse(
                    y_true=observed_group["actual_iv_change"].to_numpy(),
                    y_pred=observed_group["predicted_iv_change"].to_numpy(),
                    weights=observed_weights,
                )
                if observed_group.height > 0
                else float("nan"),
                observed_qlike_total_variance=qlike(
                    y_true=observed_group["actual_completed_total_variance"].to_numpy(),
                    y_pred=observed_group["predicted_total_variance"].to_numpy(),
                    positive_floor=positive_floor,
                )
                if observed_group.height > 0
                else float("nan"),
                interpolated_cell_count=interpolated_group.height,
                extrapolated_cell_count=extrapolated_group.height,
                full_wrmse_total_variance=weighted_rmse(
                    y_true=group["actual_completed_total_variance"].to_numpy(),
                    y_pred=group["predicted_total_variance"].to_numpy(),
                    weights=full_weights,
                ),
                full_wmae_total_variance=weighted_mae(
                    y_true=group["actual_completed_total_variance"].to_numpy(),
                    y_pred=group["predicted_total_variance"].to_numpy(),
                    weights=full_weights,
                ),
                full_mse_total_variance=weighted_mse(
                    y_true=group["actual_completed_total_variance"].to_numpy(),
                    y_pred=group["predicted_total_variance"].to_numpy(),
                    weights=full_weights,
                ),
                full_wrmse_iv=weighted_rmse(
                    y_true=group["actual_completed_iv"].to_numpy(),
                    y_pred=group["predicted_iv"].to_numpy(),
                    weights=full_weights,
                ),
                full_wmae_iv=weighted_mae(
                    y_true=group["actual_completed_iv"].to_numpy(),
                    y_pred=group["predicted_iv"].to_numpy(),
                    weights=full_weights,
                ),
                full_mse_iv_change=weighted_mse(
                    y_true=group["actual_iv_change"].to_numpy(),
                    y_pred=group["predicted_iv_change"].to_numpy(),
                    weights=full_weights,
                ),
                full_qlike_total_variance=qlike(
                    y_true=group["actual_completed_total_variance"].to_numpy(),
                    y_pred=group["predicted_total_variance"].to_numpy(),
                    positive_floor=positive_floor,
                ),
                observed_cell_count=observed_group.height,
                full_grid_cell_count=group.height,
            )
        )
    return pl.DataFrame(rows).sort(["model_name", "quote_date", "target_date"])

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

### tests/unit/test_alignment.py

```python
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ivsurf.evaluation.alignment import build_forecast_realization_panel, load_daily_spot_frame
from ivsurf.surfaces.interpolation import (
    COMPLETION_STATUS_EXTRAPOLATED,
    COMPLETION_STATUS_INTERPOLATED,
    COMPLETION_STATUS_OBSERVED,
)


def _actual_surface_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for quote_date, sigma in ((date(2021, 1, 4), 0.20), (date(2021, 1, 5), 0.22)):
        for maturity_index, maturity_days in enumerate((30, 90)):
            maturity_years = maturity_days / 365.0
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                total_variance = (sigma * sigma) * maturity_years
                rows.append(
                    {
                        "quote_date": quote_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "observed_total_variance": total_variance,
                        "observed_iv": sigma,
                        "completed_total_variance": total_variance,
                        "completed_iv": sigma,
                        "observed_mask": True,
                        "completion_status": COMPLETION_STATUS_OBSERVED,
                        "vega_sum": 1.0,
                    }
                )
    return pl.DataFrame(rows)


def test_build_forecast_realization_panel_rejects_negative_predicted_total_variance() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            forecast_rows.append(
                {
                    "model_name": "ridge",
                    "quote_date": date(2021, 1, 4),
                    "target_date": date(2021, 1, 5),
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "predicted_total_variance": -1.0e-4 if maturity_index == 0 else 5.0e-3,
                }
            )
    with pytest.raises(ValueError, match="negative total variance"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_build_forecast_realization_panel_preserves_completion_status_masks() -> None:
    actual_surface = _actual_surface_frame().with_columns(
        pl.when((pl.col("quote_date") == date(2021, 1, 5)) & (pl.col("maturity_index") == 0))
        .then(pl.lit(COMPLETION_STATUS_INTERPOLATED))
        .when((pl.col("quote_date") == date(2021, 1, 5)) & (pl.col("maturity_index") == 1))
        .then(pl.lit(COMPLETION_STATUS_EXTRAPOLATED))
        .otherwise(pl.col("completion_status"))
        .alias("completion_status")
    )
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            forecast_rows.append(
                {
                    "model_name": "ridge",
                    "quote_date": date(2021, 1, 4),
                    "target_date": date(2021, 1, 5),
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "predicted_total_variance": 5.0e-3,
                }
            )

    panel = build_forecast_realization_panel(
        actual_surface_frame=actual_surface,
        forecast_frame=pl.DataFrame(forecast_rows),
        total_variance_floor=1.0e-8,
    )

    assert panel["actual_completion_status"].to_list() == [
        COMPLETION_STATUS_INTERPOLATED,
        COMPLETION_STATUS_INTERPOLATED,
        COMPLETION_STATUS_EXTRAPOLATED,
        COMPLETION_STATUS_EXTRAPOLATED,
    ]
    assert panel["origin_completion_status"].to_list() == [COMPLETION_STATUS_OBSERVED] * 4
    assert panel["interpolated_weight"].to_list() == [1.0, 1.0, 0.0, 0.0]
    assert panel["extrapolated_weight"].to_list() == [0.0, 0.0, 1.0, 1.0]


def test_load_daily_spot_frame_uses_active_underlying_price_when_bid_ask_are_zero(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 99.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 99.0,
                "is_valid_observation": True,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    spots = load_daily_spot_frame(tmp_path / "silver")

    assert spots.to_dicts() == [
        {
            "quote_date": date(2020, 8, 10),
            "spot_1545": 97.0,
        },
        {
            "quote_date": date(2020, 8, 11),
            "spot_1545": 99.0,
        },
    ]


def test_load_daily_spot_frame_uses_valid_row_median_when_spot_varies_across_option_rows(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.1,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.2,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 140.0,
                "is_valid_observation": False,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    spots = load_daily_spot_frame(tmp_path / "silver")

    assert spots.to_dicts() == [
        {
            "quote_date": date(2020, 8, 10),
            "spot_1545": 97.1,
        }
    ]


def test_load_daily_spot_frame_rejects_nonpositive_active_underlying_price(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 0.0,
                "is_valid_observation": True,
            }
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    with pytest.raises(
        ValueError,
        match="strictly positive finite stage-08 daily spot values derived from the median",
    ):
        load_daily_spot_frame(tmp_path / "silver")

```

### tests/unit/test_stats.py

```python
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ivsurf.evaluation.alignment import (
    assert_forecast_origins_after_hpo_boundary,
    build_forecast_realization_panel,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.stats.diebold_mariano import diebold_mariano_test
from ivsurf.stats.mcs import model_confidence_set
from ivsurf.stats.spa import superior_predictive_ability_test
from ivsurf.surfaces.interpolation import COMPLETION_STATUS_OBSERVED


def _actual_surface_frame() -> pl.DataFrame:
    dates = [date(2021, 1, 4), date(2021, 1, 5)]
    rows: list[dict[str, object]] = []
    for quote_date in dates:
        for maturity_index, maturity_days in enumerate((30, 90)):
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                total_variance = (
                    0.04 if quote_date == dates[0] else 0.05
                ) * (maturity_days / 365.0)
                iv = float(np.sqrt(total_variance / (maturity_days / 365.0)))
                rows.append(
                    {
                        "quote_date": quote_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "observed_total_variance": total_variance,
                        "observed_iv": iv,
                        "completed_total_variance": total_variance,
                        "completed_iv": iv,
                        "observed_mask": True,
                        "completion_status": COMPLETION_STATUS_OBSERVED,
                        "vega_sum": 1.0 + maturity_index + moneyness_index,
                    }
                )
    return pl.DataFrame(rows)


def _forecast_frame() -> pl.DataFrame:
    target_date = date(2021, 1, 5)
    quote_date = date(2021, 1, 4)
    rows: list[dict[str, object]] = []
    for model_name, sigma in (("good", 0.05), ("bad", 0.07)):
        for maturity_index, maturity_days in enumerate((30, 90)):
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": quote_date,
                        "target_date": target_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "predicted_total_variance": sigma * (maturity_days / 365.0),
                    }
                )
    return pl.DataFrame(rows)


def test_alignment_and_loss_frame_rank_forecasts_correctly() -> None:
    panel = build_forecast_realization_panel(
        actual_surface_frame=_actual_surface_frame(),
        forecast_frame=_forecast_frame(),
        total_variance_floor=1.0e-8,
    )
    loss_frame = build_daily_loss_frame(
        panel=panel,
        positive_floor=1.0e-8,
        full_grid_weighting="uniform",
    )
    assert {
        "observed_mse_total_variance",
        "full_mse_total_variance",
        "observed_qlike_total_variance",
        "full_qlike_total_variance",
        "interpolated_cell_count",
        "extrapolated_cell_count",
    }.issubset(set(loss_frame.columns))
    assert loss_frame["interpolated_cell_count"].to_list() == [0, 0]
    assert loss_frame["extrapolated_cell_count"].to_list() == [0, 0]
    ranked = loss_frame.sort("observed_wrmse_total_variance")
    assert ranked["model_name"].to_list() == ["good", "bad"]


def test_diebold_mariano_detects_better_model() -> None:
    result = diebold_mariano_test(
        loss_a=np.asarray([2.0, 2.0, 2.0, 2.0]),
        loss_b=np.asarray([1.0, 1.0, 1.0, 1.0]),
        model_a="benchmark",
        model_b="candidate",
        alternative="greater",
        max_lag=0,
    )
    assert result.mean_differential > 0.0
    assert result.p_value == 0.0


def test_spa_and_mcs_favor_better_models() -> None:
    benchmark = np.asarray([1.2, 1.1, 1.3, 1.25, 1.15, 1.2])
    candidates = np.column_stack(
        [
            np.asarray([0.9, 0.85, 0.95, 0.9, 0.88, 0.91]),
            np.asarray([1.0, 0.98, 1.02, 1.01, 0.99, 1.0]),
            np.asarray([1.4, 1.45, 1.35, 1.5, 1.42, 1.4]),
        ]
    )
    spa = superior_predictive_ability_test(
        benchmark_losses=benchmark,
        candidate_losses=candidates,
        benchmark_model="benchmark",
        candidate_models=("good", "okay", "bad"),
        alpha=0.10,
        block_size=2,
        bootstrap_reps=200,
        seed=7,
    )
    assert spa.observed_statistic > 0.0
    assert 0.0 <= spa.p_value <= 1.0
    assert spa.alpha == 0.10
    assert "good" in spa.superior_models_by_mean

    losses = np.column_stack([benchmark, candidates])
    mcs = model_confidence_set(
        losses=losses,
        model_names=("benchmark", "good", "okay", "bad"),
        alpha=0.10,
        block_size=2,
        bootstrap_reps=200,
        seed=7,
    )
    assert "bad" not in mcs.superior_models


def test_forecast_origin_guard_rejects_hpo_contaminated_rows() -> None:
    forecast_frame = pl.DataFrame(
        {
            "model_name": ["ridge", "ridge"],
            "quote_date": [date(2021, 1, 4), date(2021, 1, 5)],
            "target_date": [date(2021, 1, 5), date(2021, 1, 6)],
            "maturity_index": [0, 0],
            "maturity_days": [30, 30],
            "moneyness_index": [0, 0],
            "moneyness_point": [0.0, 0.0],
            "predicted_total_variance": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="quote_date values"):
        assert_forecast_origins_after_hpo_boundary(
            forecast_frame,
            max_hpo_validation_date=date(2021, 1, 4),
        )

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
    rows: list[dict[str, object]] = []
    for multiplier, log_moneyness in ((1.0, -0.1), (1.2, 0.1)):
        cell_total_variance = total_variance * multiplier
        rows.append(
            {
                "quote_date": quote_date,
                "tau_years": 30.0 / 365.0,
                "log_moneyness": log_moneyness,
                "total_variance": cell_total_variance,
                "implied_volatility_1545": float(
                    (cell_total_variance / (30.0 / 365.0)) ** 0.5
                ),
                "vega_1545": 1.0,
                "spread_1545": 0.01,
                "is_valid_observation": is_valid,
            }
        )
    if include_outside_grid_row:
        rows.append(
            {
                "quote_date": quote_date,
                "tau_years": 30.0 / 365.0,
                "log_moneyness": 0.3,
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
            "moneyness_points: [-0.2, -0.1, 0.0, 0.1, 0.2]\n"
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
    assert first_built_row["rows_inside_grid_domain"] == 2
    assert first_built_row["rows_outside_grid_domain"] == 1
    assert first_built_row["grid_domain_reason_counts"] == {
        "VALID_GRID_DOMAIN": 2,
        "OUTSIDE_MONEYNESS_GRID_DOMAIN": 1,
    }
    assert first_built_row["completion_status_counts"] == {
        "extrapolated_boundary_fill": 2,
        "interpolated": 1,
        "observed": 2,
    }
    first_gold_surface = pl.read_parquet(
        tmp_path / "data" / "gold" / "year=2021" / "2021-01-04.parquet"
    )
    assert first_gold_surface["completion_status"].to_list() == [
        "extrapolated_boundary_fill",
        "observed",
        "interpolated",
        "observed",
        "extrapolated_boundary_fill",
    ]

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