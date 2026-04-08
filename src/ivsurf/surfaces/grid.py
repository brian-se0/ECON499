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

    maturity_edges = np.empty(len(grid.maturity_days) + 1, dtype=np.float64)
    maturity_years = grid.maturity_years
    maturity_edges[0] = -np.inf
    maturity_edges[-1] = np.inf
    maturity_edges[1:-1] = (maturity_years[:-1] + maturity_years[1:]) / 2.0

    moneyness_edges = np.empty(len(grid.moneyness_points) + 1, dtype=np.float64)
    money = np.asarray(grid.moneyness_points, dtype=np.float64)
    moneyness_edges[0] = -np.inf
    moneyness_edges[-1] = np.inf
    moneyness_edges[1:-1] = (money[:-1] + money[1:]) / 2.0

    maturity_index = np.searchsorted(
        maturity_edges[1:-1],
        frame["tau_years"].to_numpy(),
        side="right",
    ).astype(np.int64)
    moneyness_index = np.searchsorted(
        moneyness_edges[1:-1],
        frame["log_moneyness"].to_numpy(),
        side="right",
    ).astype(np.int64)

    return frame.with_columns(
        pl.Series("maturity_index", maturity_index),
        pl.Series("moneyness_index", moneyness_index),
    )
