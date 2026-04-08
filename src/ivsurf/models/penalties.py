"""Arbitrage-aware neural penalties."""

from __future__ import annotations

import torch


def _reshape_surface(predictions: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    return predictions.view(predictions.shape[0], grid_shape[0], grid_shape[1])


def calendar_monotonicity_penalty(
    predictions: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Penalize decreases across maturity."""

    surface = _reshape_surface(predictions, grid_shape)
    diffs = surface[:, 1:, :] - surface[:, :-1, :]
    return torch.relu(-diffs).mean()


def convexity_penalty(predictions: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Penalize negative second differences across moneyness."""

    surface = _reshape_surface(predictions, grid_shape)
    second_diffs = surface[:, :, 2:] - (2.0 * surface[:, :, 1:-1]) + surface[:, :, :-2]
    return torch.relu(-second_diffs).mean()


def roughness_penalty(predictions: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Discourage large local curvature in either axis."""

    surface = _reshape_surface(predictions, grid_shape)
    maturity_curvature = surface[:, 2:, :] - (2.0 * surface[:, 1:-1, :]) + surface[:, :-2, :]
    moneyness_curvature = surface[:, :, 2:] - (2.0 * surface[:, :, 1:-1]) + surface[:, :, :-2]
    return maturity_curvature.square().mean() + moneyness_curvature.square().mean()

