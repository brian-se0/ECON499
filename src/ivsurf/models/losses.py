"""Forecast losses for the neural surface model."""

from __future__ import annotations

import torch


def weighted_surface_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    training_weights: torch.Tensor,
    observed_loss_weight: float,
    imputed_loss_weight: float,
) -> torch.Tensor:
    """Weighted MSE for completed-surface supervision in the neural model."""

    observation_weights = torch.where(
        observed_mask > 0.5,
        torch.full_like(predictions, observed_loss_weight),
        torch.full_like(predictions, imputed_loss_weight),
    )
    positive_training_weights = torch.clamp_min(training_weights, 0.0)
    weights = observation_weights * positive_training_weights
    imputed_cells = observed_mask <= 0.5
    if (
        imputed_loss_weight > 0.0
        and bool(torch.any(imputed_cells).detach().item())
        and float(weights[imputed_cells].sum().detach().item()) <= 0.0
    ):
        message = (
            "weighted_surface_mse requires positive training weight on at least one "
            "completed-only target cell when imputed_loss_weight > 0."
        )
        raise ValueError(message)
    weight_sum = weights.sum()
    if float(weight_sum.detach().item()) <= 0.0:
        message = "weighted_surface_mse requires at least one positive supervised target cell."
        raise ValueError(message)
    return ((predictions - targets).square() * weights).sum() / weight_sum
