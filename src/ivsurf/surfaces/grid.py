"""Fixed-grid helpers for daily surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.grid_validation import validate_maturity_days, validate_moneyness_points

MONEYNESS_COORDINATE = "log_spot_moneyness"
MATURITY_COORDINATE = "ACT/365_years_from_effective_decision_to_last_tradable_close"
SURFACE_GRID_SCHEMA_VERSION = "surface_grid_v2"
SURFACE_GRID_CELL_COLUMNS = (
    "maturity_index",
    "maturity_days",
    "moneyness_index",
    "moneyness_point",
)


@dataclass(frozen=True, slots=True)
class SurfaceGrid:
    """Fixed maturity x moneyness grid."""

    maturity_days: tuple[int, ...]
    moneyness_points: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_maturity_days(self.maturity_days)
        validate_moneyness_points(self.moneyness_points)

    @property
    def maturity_years(self) -> np.ndarray:
        return np.asarray(self.maturity_days, dtype=np.float64) / 365.0

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.maturity_days), len(self.moneyness_points))

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
            "maturity_coordinate": MATURITY_COORDINATE,
            "moneyness_coordinate": MONEYNESS_COORDINATE,
            "maturity_days": list(self.maturity_days),
            "moneyness_points": list(self.moneyness_points),
        }

    @property
    def grid_hash(self) -> str:
        payload = json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return sha256(payload).hexdigest()

    @classmethod
    def from_config(cls, config: SurfaceGridConfig) -> SurfaceGrid:
        return cls(maturity_days=config.maturity_days, moneyness_points=config.moneyness_points)


def require_surface_grid_metadata(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    *,
    dataset_name: str,
) -> None:
    """Require persisted artifact metadata to match the configured fixed grid."""

    expected = {
        "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
        "maturity_coordinate": MATURITY_COORDINATE,
        "moneyness_coordinate": MONEYNESS_COORDINATE,
        "surface_grid_hash": grid.grid_hash,
    }
    missing_columns = sorted(name for name in expected if name not in frame.columns)
    if missing_columns:
        message = f"{dataset_name} is missing surface-grid metadata columns: {missing_columns}."
        raise ValueError(message)
    for column_name, expected_value in expected.items():
        observed = frame.select(pl.col(column_name).unique()).to_series().to_list()
        if observed != [expected_value]:
            message = (
                f"{dataset_name} has {column_name}={observed!r}, "
                f"expected [{expected_value!r}]."
            )
            raise ValueError(message)


def expected_surface_grid_cells(grid: SurfaceGrid) -> pl.DataFrame:
    """Return the canonical long-form fixed-grid cell definitions."""

    rows: list[dict[str, int | float]] = []
    for maturity_index, maturity_days in enumerate(grid.maturity_days):
        for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
            rows.append(
                {
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                }
            )
    return pl.DataFrame(rows)


def _require_columns(frame: pl.DataFrame, *, columns: tuple[str, ...], dataset_name: str) -> None:
    missing_columns = sorted(column for column in columns if column not in frame.columns)
    if missing_columns:
        message = f"{dataset_name} is missing required surface-grid columns: {missing_columns}."
        raise ValueError(message)
    for column in columns:
        if frame[column].null_count() > 0:
            message = f"{dataset_name} contains nulls in required surface-grid column {column}."
            raise ValueError(message)


def _format_preview(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> str:
    preview: list[str] = []
    for row in frame.select(columns).head(5).iter_rows(named=True):
        preview.append(", ".join(f"{column}={row[column]!r}" for column in columns))
    return "; ".join(preview)


def require_complete_unique_surface_grid(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    *,
    dataset_name: str,
    group_columns: tuple[str, ...] = (),
) -> None:
    """Fail fast unless each surface group contains one complete fixed-grid cell set."""

    required_columns = (*group_columns, *SURFACE_GRID_CELL_COLUMNS)
    _require_columns(frame, columns=required_columns, dataset_name=dataset_name)
    if frame.is_empty():
        message = f"{dataset_name} must contain at least one surface-grid row."
        raise ValueError(message)

    key_columns = (*group_columns, "maturity_index", "moneyness_index")
    duplicates = (
        frame.group_by(key_columns)
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
        .sort(key_columns)
    )
    if not duplicates.is_empty():
        message = (
            f"{dataset_name} contains duplicate surface-grid cells for key columns "
            f"{key_columns!r}; duplicate_count={duplicates.height}. First duplicates: "
            f"{_format_preview(duplicates, columns=key_columns)}."
        )
        raise ValueError(message)

    expected_cells = expected_surface_grid_cells(grid)
    actual_cell_definitions = frame.select(SURFACE_GRID_CELL_COLUMNS).unique()
    unexpected_cells = actual_cell_definitions.join(
        expected_cells,
        on=SURFACE_GRID_CELL_COLUMNS,
        how="anti",
    )
    missing_cells = expected_cells.join(
        actual_cell_definitions,
        on=SURFACE_GRID_CELL_COLUMNS,
        how="anti",
    )
    if not unexpected_cells.is_empty() or not missing_cells.is_empty():
        message = (
            f"{dataset_name} surface-grid cell coordinates do not match the configured grid; "
            f"unexpected_count={unexpected_cells.height}, missing_count={missing_cells.height}. "
            "First unexpected: "
            f"{_format_preview(unexpected_cells, columns=SURFACE_GRID_CELL_COLUMNS)}. "
            f"First missing: {_format_preview(missing_cells, columns=SURFACE_GRID_CELL_COLUMNS)}."
        )
        raise ValueError(message)

    expected_rows = grid.shape[0] * grid.shape[1]
    if not group_columns:
        if frame.height == expected_rows:
            return
        message = (
            f"{dataset_name} must contain exactly {expected_rows} grid cells; "
            f"found {frame.height}."
        )
        raise ValueError(message)

    invalid_counts = (
        frame.group_by(group_columns)
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") != expected_rows)
        .sort(group_columns)
    )
    if invalid_counts.is_empty():
        return
    message = (
        f"{dataset_name} must contain exactly {expected_rows} grid cells per "
        f"{group_columns!r}; invalid_group_count={invalid_counts.height}. First invalid groups: "
        f"{_format_preview(invalid_counts, columns=(*group_columns, 'row_count'))}."
    )
    raise ValueError(message)


def infer_surface_grid_from_frame(
    frame: pl.DataFrame,
    *,
    dataset_name: str,
) -> SurfaceGrid:
    """Infer a SurfaceGrid after validating index-to-coordinate uniqueness."""

    _require_columns(frame, columns=SURFACE_GRID_CELL_COLUMNS, dataset_name=dataset_name)
    maturity_mapping = (
        frame.group_by("maturity_index")
        .agg(pl.col("maturity_days").n_unique().alias("unique_days"))
        .filter(pl.col("unique_days") != 1)
    )
    if not maturity_mapping.is_empty():
        message = f"{dataset_name} has non-unique maturity_days for at least one maturity_index."
        raise ValueError(message)
    moneyness_mapping = (
        frame.group_by("moneyness_index")
        .agg(pl.col("moneyness_point").n_unique().alias("unique_points"))
        .filter(pl.col("unique_points") != 1)
    )
    if not moneyness_mapping.is_empty():
        message = (
            f"{dataset_name} has non-unique moneyness_point for at least one moneyness_index."
        )
        raise ValueError(message)

    maturity_rows = (
        frame.select("maturity_index", "maturity_days").unique().sort("maturity_index")
    )
    moneyness_rows = (
        frame.select("moneyness_index", "moneyness_point").unique().sort("moneyness_index")
    )
    maturity_indices = maturity_rows["maturity_index"].to_list()
    moneyness_indices = moneyness_rows["moneyness_index"].to_list()
    expected_maturity_indices = list(range(len(maturity_indices)))
    expected_moneyness_indices = list(range(len(moneyness_indices)))
    if maturity_indices != expected_maturity_indices:
        message = (
            f"{dataset_name} maturity_index values must be contiguous from zero; "
            f"found {maturity_indices!r}."
        )
        raise ValueError(message)
    if moneyness_indices != expected_moneyness_indices:
        message = (
            f"{dataset_name} moneyness_index values must be contiguous from zero; "
            f"found {moneyness_indices!r}."
        )
        raise ValueError(message)
    return SurfaceGrid(
        maturity_days=tuple(int(value) for value in maturity_rows["maturity_days"].to_list()),
        moneyness_points=tuple(
            float(value) for value in moneyness_rows["moneyness_point"].to_list()
        ),
    )


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
