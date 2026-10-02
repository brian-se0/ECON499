"""Slice-level forecast reporting across maturity, moneyness, and stress windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivcast.config import StressWindowConfig
from ivcast.evaluation.metric_contracts import (
    require_observed_weight_contract,
    require_positive_observed_weight_sum,
)
from ivcast.evaluation.metrics import (
    validate_total_variance_array,
    weighted_mae,
    weighted_mse,
    weighted_rmse,
)


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
    mse_total_variance: float
    wrmse_iv: float
    wmae_iv: float
    mse_iv_change: float
    qlike_total_variance: float
    cell_count: int
    target_day_count: int


def _day_normalized(target_dates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalize weights within each target date so every date carries unit weight."""

    _, inverse = np.unique(target_dates, return_inverse=True)
    day_totals = np.bincount(inverse, weights=weights)
    return np.asarray(weights / day_totals[inverse], dtype=np.float64)


def _scope_weights(
    frame: pl.DataFrame,
    evaluation_scope: str,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return scoped rows plus day-normalized loss weights and QLIKE weights.

    The official daily metrics normalize weights within each target date and then
    average dates. Slice metrics follow the same rule so raw vega, which grows with the
    index level and strike count, does not let late-sample dates dominate a slice.
    """

    if evaluation_scope == "observed":
        observed_mask, observed_weights = require_observed_weight_contract(
            observed_mask=frame["actual_observed_mask"].to_numpy(),
            observed_weight=frame["observed_weight"].to_numpy(),
            context="Slice report panel",
        )
        scoped = frame.filter(pl.col("actual_observed_mask"))
        weights = observed_weights[observed_mask]
    elif evaluation_scope == "full":
        scoped = frame
        weights = np.ones(scoped.height, dtype=np.float64)
    else:
        message = f"Unsupported evaluation_scope: {evaluation_scope}"
        raise ValueError(message)
    if scoped.is_empty():
        return scoped, weights, weights
    target_dates = scoped["target_date"].to_numpy()
    if (weights <= 0.0).any():
        message = "Slice report weights must be strictly positive on scoped cells."
        raise ValueError(message)
    return (
        scoped,
        _day_normalized(target_dates, weights),
        _day_normalized(target_dates, np.ones(scoped.height, dtype=np.float64)),
    )


def _weighted_qlike(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    weights: np.ndarray,
    positive_floor: float,
) -> float:
    true_values = np.maximum(
        validate_total_variance_array(y_true, context="QLIKE y_true", allow_zero=True),
        positive_floor,
    )
    pred_values = np.maximum(
        validate_total_variance_array(y_pred, context="QLIKE y_pred", allow_zero=True),
        positive_floor,
    )
    ratio = true_values / pred_values
    return float(np.sum(weights * (ratio - np.log(ratio) - 1.0)) / np.sum(weights))


def _nan_metric_row(
    frame: pl.DataFrame,
    scoped: pl.DataFrame,
    *,
    slice_family: str,
    slice_label: str,
    slice_value_float: float | None,
    evaluation_scope: str,
) -> SliceMetricRow:
    nan = float("nan")
    return SliceMetricRow(
        model_name=str(frame["model_name"][0]),
        slice_family=slice_family,
        slice_label=slice_label,
        slice_value_float=slice_value_float,
        evaluation_scope=evaluation_scope,
        wrmse_total_variance=nan,
        wmae_total_variance=nan,
        mse_total_variance=nan,
        wrmse_iv=nan,
        wmae_iv=nan,
        mse_iv_change=nan,
        qlike_total_variance=nan,
        cell_count=scoped.height,
        target_day_count=0 if scoped.is_empty() else int(scoped["target_date"].n_unique()),
    )


def _build_metric_row(
    frame: pl.DataFrame,
    *,
    slice_family: str,
    slice_label: str,
    slice_value_float: float | None,
    evaluation_scope: str,
    positive_floor: float,
) -> SliceMetricRow:
    scoped, weights, qlike_weights = _scope_weights(frame, evaluation_scope)
    if scoped.is_empty():
        return _nan_metric_row(
            frame,
            scoped,
            slice_family=slice_family,
            slice_label=slice_label,
            slice_value_float=slice_value_float,
            evaluation_scope=evaluation_scope,
        )
    if evaluation_scope == "observed":
        require_positive_observed_weight_sum(
            weights,
            metric_name="observed slice metrics",
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
        mse_total_variance=weighted_mse(
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
        mse_iv_change=weighted_mse(
            y_true=scoped["actual_iv_change"].to_numpy(),
            y_pred=scoped["predicted_iv_change"].to_numpy(),
            weights=weights,
        ),
        qlike_total_variance=_weighted_qlike(
            scoped["actual_completed_total_variance"].to_numpy(),
            scoped["predicted_total_variance"].to_numpy(),
            weights=qlike_weights,
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
