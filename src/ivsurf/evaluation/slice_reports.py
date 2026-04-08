"""Slice-level forecast reporting across maturity, moneyness, and stress windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivsurf.config import StressWindowConfig
from ivsurf.evaluation.metrics import qlike, weighted_mae, weighted_rmse


@dataclass(frozen=True, slots=True)
class SliceMetricRow:
    """Aggregated slice-level error summary."""

    model_name: str
    slice_family: str
    slice_label: str
    slice_value_float: float | None
    evaluation_scope: str
    wrmse_total_variance: float
    wmae_total_variance: float
    wrmse_iv: float
    wmae_iv: float
    mse_iv_change: float
    qlike_total_variance: float
    cell_count: int
    target_day_count: int


def _weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    normalized = weights.astype(np.float64, copy=False)
    total = normalized.sum()
    if total <= 0.0:
        return float(np.mean(np.square(y_true - y_pred)))
    normalized = normalized / total
    return float(np.sum(normalized * np.square(y_true - y_pred)))


def _scope_weights(frame: pl.DataFrame, evaluation_scope: str) -> tuple[pl.DataFrame, np.ndarray]:
    if evaluation_scope == "observed":
        scoped = frame.filter(pl.col("actual_observed_mask"))
        weights = scoped["observed_weight"].to_numpy().astype(np.float64)
        return scoped, weights
    if evaluation_scope == "full":
        scoped = frame
        return scoped, np.ones(scoped.height, dtype=np.float64)
    message = f"Unsupported evaluation_scope: {evaluation_scope}"
    raise ValueError(message)


def _build_metric_row(
    frame: pl.DataFrame,
    *,
    slice_family: str,
    slice_label: str,
    slice_value_float: float | None,
    evaluation_scope: str,
    positive_floor: float,
) -> SliceMetricRow:
    scoped, weights = _scope_weights(frame, evaluation_scope)
    if scoped.is_empty():
        nan = float("nan")
        return SliceMetricRow(
            model_name=str(frame["model_name"][0]),
            slice_family=slice_family,
            slice_label=slice_label,
            slice_value_float=slice_value_float,
            evaluation_scope=evaluation_scope,
            wrmse_total_variance=nan,
            wmae_total_variance=nan,
            wrmse_iv=nan,
            wmae_iv=nan,
            mse_iv_change=nan,
            qlike_total_variance=nan,
            cell_count=0,
            target_day_count=0,
        )

    return SliceMetricRow(
        model_name=str(scoped["model_name"][0]),
        slice_family=slice_family,
        slice_label=slice_label,
        slice_value_float=slice_value_float,
        evaluation_scope=evaluation_scope,
        wrmse_total_variance=weighted_rmse(
            y_true=scoped["actual_completed_total_variance"].to_numpy(),
            y_pred=scoped["predicted_total_variance"].to_numpy(),
            weights=weights,
        ),
        wmae_total_variance=weighted_mae(
            y_true=scoped["actual_completed_total_variance"].to_numpy(),
            y_pred=scoped["predicted_total_variance"].to_numpy(),
            weights=weights,
        ),
        wrmse_iv=weighted_rmse(
            y_true=scoped["actual_completed_iv"].to_numpy(),
            y_pred=scoped["predicted_iv"].to_numpy(),
            weights=weights,
        ),
        wmae_iv=weighted_mae(
            y_true=scoped["actual_completed_iv"].to_numpy(),
            y_pred=scoped["predicted_iv"].to_numpy(),
            weights=weights,
        ),
        mse_iv_change=_weighted_mse(
            y_true=scoped["actual_iv_change"].to_numpy(),
            y_pred=scoped["predicted_iv_change"].to_numpy(),
            weights=weights,
        ),
        qlike_total_variance=qlike(
            y_true=scoped["actual_completed_total_variance"].to_numpy(),
            y_pred=scoped["predicted_total_variance"].to_numpy(),
            positive_floor=positive_floor,
        ),
        cell_count=scoped.height,
        target_day_count=int(scoped["target_date"].n_unique()),
    )


def _slice_rows_for_group(
    group: pl.DataFrame,
    *,
    slice_family: str,
    slice_label: str,
    slice_value_float: float | None,
    positive_floor: float,
) -> list[SliceMetricRow]:
    return [
        _build_metric_row(
            group,
            slice_family=slice_family,
            slice_label=slice_label,
            slice_value_float=slice_value_float,
            evaluation_scope=evaluation_scope,
            positive_floor=positive_floor,
        )
        for evaluation_scope in ("observed", "full")
    ]


def build_slice_metric_frame(
    panel: pl.DataFrame,
    *,
    positive_floor: float,
    stress_windows: tuple[StressWindowConfig, ...],
) -> pl.DataFrame:
    """Build maturity, moneyness, and stress-period slice metrics."""

    rows: list[SliceMetricRow] = []

    for group in panel.partition_by(["model_name", "maturity_days"], maintain_order=True):
        maturity_days = int(group["maturity_days"][0])
        rows.extend(
            _slice_rows_for_group(
                group,
                slice_family="maturity",
                slice_label=f"{maturity_days}d",
                slice_value_float=float(maturity_days),
                positive_floor=positive_floor,
            )
        )

    for group in panel.partition_by(["model_name", "moneyness_point"], maintain_order=True):
        moneyness_point = float(group["moneyness_point"][0])
        rows.extend(
            _slice_rows_for_group(
                group,
                slice_family="moneyness",
                slice_label=f"{moneyness_point:+.2f}",
                slice_value_float=moneyness_point,
                positive_floor=positive_floor,
            )
        )

    for window in stress_windows:
        window_frame = panel.filter(
            pl.col("target_date").is_between(window.start_date, window.end_date, closed="both")
        )
        for group in window_frame.partition_by("model_name", maintain_order=True):
            rows.extend(
                _slice_rows_for_group(
                    group,
                    slice_family="stress_window",
                    slice_label=window.label,
                    slice_value_float=None,
                    positive_floor=positive_floor,
                )
            )

    return pl.DataFrame(rows).sort(
        ["slice_family", "slice_label", "evaluation_scope", "wrmse_total_variance", "model_name"]
    )
