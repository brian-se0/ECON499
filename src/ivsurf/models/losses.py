"""Forecast losses for the neural surface model."""

from __future__ import annotations

import torch


def weighted_surface_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    observed_loss_weight: float,
    imputed_loss_weight: float,
) -> torch.Tensor:
    """Weighted MSE that distinguishes observed and imputed target cells."""

    weights = torch.where(
        observed_mask > 0.5,
        torch.full_like(predictions, observed_loss_weight),
        torch.full_like(predictions, imputed_loss_weight),
    )
    return ((predictions - targets).square() * weights).mean()

