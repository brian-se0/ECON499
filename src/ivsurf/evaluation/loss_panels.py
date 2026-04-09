"""Daily loss-panel construction from aligned forecast artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import qlike, weighted_mae, weighted_mse, weighted_rmse


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
