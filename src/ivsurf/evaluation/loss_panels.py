"""Daily loss-panel construction from aligned forecast artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivsurf.evaluation.metric_contracts import (
    require_binary_mask_array,
    require_finite_array,
    require_nonnegative_weights,
    require_observed_weight_contract,
    require_positive_observed_weight_sum,
)
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

    actual_values = require_finite_array(y_true, context="Validation daily-loss y_true")
    predicted_values = require_finite_array(y_pred, context="Validation daily-loss y_pred")
    observed_mask_values = require_binary_mask_array(
        observed_masks,
        context="Validation daily-loss observed_masks",
    )
    vega_weight_values = require_nonnegative_weights(
        vega_weights,
        context="Validation daily-loss vega_weights",
    )
    invalid_observed_weights = observed_mask_values & (vega_weight_values <= 0.0)
    if invalid_observed_weights.any():
        message = (
            "Validation daily-loss observed cells must have strictly positive vega_weights; "
            f"invalid_count={int(invalid_observed_weights.sum())}."
        )
        raise ValueError(message)
    if y_true.shape[0] == 0:
        message = "Validation daily-loss inputs must contain at least one validation surface."
        raise ValueError(message)

    rows: list[float] = []
    for row_index in range(y_true.shape[0]):
        actual_row = actual_values[row_index]
        predicted_row = predicted_values[row_index]
        observed_mask_row = observed_mask_values[row_index]
        observed_weights = vega_weight_values[row_index][observed_mask_row]
        if metric_name.startswith("observed_"):
            require_positive_observed_weight_sum(observed_weights, metric_name=metric_name)

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
        observed_mask, observed_weight_values = require_observed_weight_contract(
            observed_mask=group["actual_observed_mask"].to_numpy(),
            observed_weight=group["observed_weight"].to_numpy(),
            context=(
                "Daily loss panel "
                f"model={group['model_name'][0]!r} "
                f"quote_date={group['quote_date'][0]!r} "
                f"target_date={group['target_date'][0]!r}"
            ),
        )
        observed_group = group.filter(pl.col("actual_observed_mask"))
        interpolated_group = group.filter(
            pl.col("actual_completion_status") == COMPLETION_STATUS_INTERPOLATED
        )
        extrapolated_group = group.filter(
            pl.col("actual_completion_status") == COMPLETION_STATUS_EXTRAPOLATED
        )
        observed_weights = observed_weight_values[observed_mask]
        require_positive_observed_weight_sum(
            observed_weights,
            metric_name="observed daily loss metrics",
        )
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
                ,
                observed_wmae_total_variance=weighted_mae(
                    y_true=observed_group["actual_completed_total_variance"].to_numpy(),
                    y_pred=observed_group["predicted_total_variance"].to_numpy(),
                    weights=observed_weights,
                )
                ,
                observed_mse_total_variance=weighted_mse(
                    y_true=observed_group["actual_completed_total_variance"].to_numpy(),
                    y_pred=observed_group["predicted_total_variance"].to_numpy(),
                    weights=observed_weights,
                )
                ,
                observed_wrmse_iv=weighted_rmse(
                    y_true=observed_group["actual_completed_iv"].to_numpy(),
                    y_pred=observed_group["predicted_iv"].to_numpy(),
                    weights=observed_weights,
                )
                ,
                observed_wmae_iv=weighted_mae(
                    y_true=observed_group["actual_completed_iv"].to_numpy(),
                    y_pred=observed_group["predicted_iv"].to_numpy(),
                    weights=observed_weights,
                )
                ,
                observed_mse_iv_change=weighted_mse(
                    y_true=observed_group["actual_iv_change"].to_numpy(),
                    y_pred=observed_group["predicted_iv_change"].to_numpy(),
                    weights=observed_weights,
                )
                ,
                observed_qlike_total_variance=qlike(
                    y_true=observed_group["actual_completed_total_variance"].to_numpy(),
                    y_pred=observed_group["predicted_total_variance"].to_numpy(),
                    positive_floor=positive_floor,
                )
                ,
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
