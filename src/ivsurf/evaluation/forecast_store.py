"""Persist model forecasts to parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ivsurf.surfaces.grid import SurfaceGrid


def write_forecasts(
    output_path: Path,
    model_name: str,
    quote_dates: np.ndarray,
    target_dates: np.ndarray,
    predictions: np.ndarray,
    grid: SurfaceGrid,
) -> None:
    """Write long-form forecast artifacts."""

    rows: list[dict[str, object]] = []
    pairs = zip(quote_dates, target_dates, strict=True)
    for row_index, (quote_date, target_date) in enumerate(pairs):
        surface = predictions[row_index].reshape(grid.shape)
        for maturity_index, maturity_day in enumerate(grid.maturity_days):
            for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": quote_date,
                        "target_date": target_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_day,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "predicted_total_variance": float(surface[maturity_index, moneyness_index]),
                    }
                )
    frame = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="zstd", statistics=True)
