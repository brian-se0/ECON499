"""Forecast losses for the neural surface model."""

from __future__ import annotations

import torch


def weighted_surface_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    vega_weights: torch.Tensor,
    observed_loss_weight: float,
    imputed_loss_weight: float,
) -> torch.Tensor:
    """Weighted MSE aligned with observed-mask x vega evaluation semantics."""

    observation_weights = torch.where(
        observed_mask > 0.5,
        torch.full_like(predictions, observed_loss_weight),
        torch.full_like(predictions, imputed_loss_weight),
    )
    weights = observation_weights * torch.clamp_min(vega_weights, 0.0)
    weight_sum = weights.sum()
    if float(weight_sum.detach().item()) <= 0.0:
        message = "weighted_surface_mse requires at least one positive vega-weighted target cell."
        raise ValueError(message)
    return ((predictions - targets).square() * weights).sum() / weight_sum
