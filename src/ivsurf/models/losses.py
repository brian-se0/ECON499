"""Forecast losses for the neural surface model."""

from __future__ import annotations

from math import isfinite

import torch


def _require_same_shape(*, expected: torch.Size, tensor: torch.Tensor, name: str) -> None:
    if tensor.shape == expected:
        return
    message = f"weighted_surface_mse expected {name} shape {expected!r}, found {tensor.shape!r}."
    raise ValueError(message)


def _require_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().detach().item()):
        message = f"weighted_surface_mse requires finite {name}."
        raise ValueError(message)


def _require_binary_mask(mask: torch.Tensor) -> None:
    _require_finite_tensor(mask, name="observed_mask")
    valid = (mask == 0) | (mask == 1)
    if bool(valid.all().detach().item()):
        return
    message = "weighted_surface_mse requires observed_mask to be binary 0/1."
    raise ValueError(message)


def _require_nonnegative_tensor(tensor: torch.Tensor, *, name: str) -> None:
    _require_finite_tensor(tensor, name=name)
    if bool((tensor >= 0.0).all().detach().item()):
        return
    message = f"weighted_surface_mse requires nonnegative {name}."
    raise ValueError(message)


def _require_nonnegative_finite_loss_weight(value: float, *, name: str) -> None:
    if isfinite(value) and value >= 0.0:
        return
    message = f"weighted_surface_mse requires finite nonnegative {name}; found {value!r}."
    raise ValueError(message)


def weighted_surface_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    training_weights: torch.Tensor,
    observed_loss_weight: float,
    imputed_loss_weight: float,
) -> torch.Tensor:
    """Weighted MSE for completed-surface supervision in the neural model."""

    if predictions.ndim == 0:
        message = "weighted_surface_mse requires non-scalar prediction and target tensors."
        raise ValueError(message)
    _require_same_shape(expected=predictions.shape, tensor=targets, name="targets")
    _require_same_shape(expected=predictions.shape, tensor=observed_mask, name="observed_mask")
    _require_same_shape(
        expected=predictions.shape,
        tensor=training_weights,
        name="training_weights",
    )
    _require_finite_tensor(predictions, name="predictions")
    _require_finite_tensor(targets, name="targets")
    _require_binary_mask(observed_mask)
    _require_nonnegative_tensor(training_weights, name="training_weights")
    _require_nonnegative_finite_loss_weight(
        observed_loss_weight,
        name="observed_loss_weight",
    )
    _require_nonnegative_finite_loss_weight(
        imputed_loss_weight,
        name="imputed_loss_weight",
    )

    observed_cells = observed_mask == 1
    observation_weights = torch.where(
        observed_cells,
        torch.full_like(predictions, observed_loss_weight),
        torch.full_like(predictions, imputed_loss_weight),
    )
    weights = observation_weights * training_weights
    if (
        observed_loss_weight > 0.0
        and bool(torch.any(observed_cells).detach().item())
        and float(weights[observed_cells].sum().detach().item()) <= 0.0
    ):
        message = (
            "weighted_surface_mse requires positive training weight on at least one "
            "observed target cell when observed_loss_weight > 0."
        )
        raise ValueError(message)
    imputed_cells = ~observed_cells
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
    loss = ((predictions - targets).square() * weights).sum() / weight_sum
    if loss.ndim != 0 or not bool(torch.isfinite(loss).detach().item()):
        message = "weighted_surface_mse produced a non-finite scalar loss."
        raise RuntimeError(message)
    return loss
