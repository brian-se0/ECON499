"""Daily loss-matrix pivoting with Stage-07 alignment semantics.

This mirrors the Stage-07 pivot contract so post-hoc stages can rebuild loss
matrices from the saved daily loss frame without touching the frozen Stage-07
entrypoint.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def daily_loss_matrix(
    loss_frame: pl.DataFrame,
    metric_column: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Pivot one daily loss metric into an aligned dates-by-models matrix."""

    if metric_column not in loss_frame.columns:
        message = f"Daily loss frame is missing metric column {metric_column!r}."
        raise ValueError(message)
    duplicate_keys = (
        loss_frame.group_by(["model_name", "target_date"])
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
    )
    if not duplicate_keys.is_empty():
        message = (
            "Daily loss frame contains duplicate model_name/target_date rows; "
            f"duplicate key count={duplicate_keys.height}."
        )
        raise ValueError(message)
    pivoted = (
        loss_frame.select("model_name", "target_date", metric_column)
        .pivot(on="model_name", index="target_date", values=metric_column)
        .sort("target_date")
    )
    model_columns = tuple(column for column in pivoted.columns if column != "target_date")
    if not model_columns:
        message = "Daily loss matrix must contain at least one model column."
        raise ValueError(message)
    if any(pivoted[column].null_count() > 0 for column in model_columns):
        message = (
            "Daily loss matrix contains missing model/date coverage after pivoting "
            f"metric {metric_column!r}."
        )
        raise ValueError(message)
    matrix = pivoted.select(model_columns).to_numpy().astype(np.float64)
    if not np.isfinite(matrix).all():
        message = f"Daily loss matrix for metric {metric_column!r} contains non-finite values."
        raise ValueError(message)
    target_dates = tuple(str(value) for value in pivoted["target_date"].to_list())
    return matrix, model_columns, target_dates
