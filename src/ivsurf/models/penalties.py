"""Arbitrage-aware neural penalties."""

from __future__ import annotations

import math

import torch


def _reshape_surface(predictions: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    if predictions.ndim != 2:
        message = f"penalty inputs must be rank-2 [samples, cells], found {predictions.ndim}."
        raise ValueError(message)
    if len(grid_shape) != 2:
        message = f"grid_shape must contain maturity and moneyness sizes, found {grid_shape!r}."
        raise ValueError(message)
    if grid_shape[0] <= 0 or grid_shape[1] <= 0:
        message = f"grid_shape sizes must be strictly positive, found {grid_shape!r}."
        raise ValueError(message)
    expected_cells = grid_shape[0] * grid_shape[1]
    if predictions.shape[1] != expected_cells:
        message = (
            "penalty inputs must match grid_shape cell count: "
            f"{predictions.shape[1]} != {expected_cells}."
        )
        raise ValueError(message)
    if not bool(torch.isfinite(predictions).all().detach().item()):
        message = "penalty inputs must contain only finite predictions."
        raise ValueError(message)
    return predictions.view(predictions.shape[0], grid_shape[0], grid_shape[1])


def _require_min_axis_length(
    grid_shape: tuple[int, int],
    *,
    maturity_min: int,
    moneyness_min: int,
    penalty_name: str,
) -> None:
    if grid_shape[0] >= maturity_min and grid_shape[1] >= moneyness_min:
        return
    message = (
        f"{penalty_name} requires grid_shape with at least {maturity_min} maturity "
        f"points and {moneyness_min} moneyness points; found {grid_shape!r}."
    )
    raise ValueError(message)


def _require_finite_scalar(value: torch.Tensor, *, penalty_name: str) -> torch.Tensor:
    if value.ndim == 0 and bool(torch.isfinite(value).detach().item()):
        return value
    message = f"{penalty_name} produced a non-finite scalar penalty."
    raise RuntimeError(message)


def _coordinate_tensor(
    coordinates: tuple[float, ...],
    *,
    expected_size: int,
    device: torch.device,
    dtype: torch.dtype,
    coordinate_name: str,
) -> torch.Tensor:
    if len(coordinates) != expected_size:
        message = (
            f"{coordinate_name} length {len(coordinates)} does not match expected size "
            f"{expected_size}."
        )
        raise ValueError(message)
    tensor = torch.as_tensor(coordinates, dtype=dtype, device=device)
    if not bool(torch.isfinite(tensor).all().detach().item()):
        message = f"{coordinate_name} must contain only finite values."
        raise ValueError(message)
    if not bool((tensor[1:] > tensor[:-1]).all().detach().item()):
        message = f"{coordinate_name} must be strictly increasing."
        raise ValueError(message)
    return tensor


def _normal_cdf(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def _normalized_call_prices(
    total_variance: torch.Tensor,
    *,
    log_moneyness: torch.Tensor,
) -> torch.Tensor:
    if not bool(torch.isfinite(total_variance).all().detach().item()):
        message = "total_variance must contain only finite values."
        raise ValueError(message)
    if not bool((total_variance > 0.0).all().detach().item()):
        message = "total_variance must be strictly positive for price convexity penalties."
        raise ValueError(message)
    strikes = torch.exp(log_moneyness)
    sqrt_variance = torch.sqrt(total_variance)
    d1 = ((-log_moneyness).view(1, 1, -1) + (0.5 * total_variance)) / sqrt_variance
    d2 = d1 - sqrt_variance
    return _normal_cdf(d1) - (strikes.view(1, 1, -1) * _normal_cdf(d2))


def _second_derivative_nonuniform(
    values: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    left_spacing = coordinates[1:-1] - coordinates[:-2]
    right_spacing = coordinates[2:] - coordinates[1:-1]
    full_spacing = coordinates[2:] - coordinates[:-2]
    left_slope = (values[:, :, 1:-1] - values[:, :, :-2]) / left_spacing.view(1, 1, -1)
    right_slope = (values[:, :, 2:] - values[:, :, 1:-1]) / right_spacing.view(1, 1, -1)
    return 2.0 * (right_slope - left_slope) / full_spacing.view(1, 1, -1)


def calendar_monotonicity_penalty(
    predictions: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Penalize decreases across maturity."""

    _require_min_axis_length(
        grid_shape,
        maturity_min=2,
        moneyness_min=1,
        penalty_name="calendar_monotonicity_penalty",
    )
    surface = _reshape_surface(predictions, grid_shape)
    diffs = surface[:, 1:, :] - surface[:, :-1, :]
    return _require_finite_scalar(
        torch.relu(-diffs).mean(),
        penalty_name="calendar_monotonicity_penalty",
    )


def convexity_penalty(
    predictions: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    moneyness_points: tuple[float, ...],
) -> torch.Tensor:
    """Penalize negative strike-space call-price curvature across moneyness."""

    _require_min_axis_length(
        grid_shape,
        maturity_min=1,
        moneyness_min=3,
        penalty_name="convexity_penalty",
    )
    surface = _reshape_surface(predictions, grid_shape)
    log_moneyness = _coordinate_tensor(
        moneyness_points,
        expected_size=grid_shape[1],
        device=surface.device,
        dtype=surface.dtype,
        coordinate_name="moneyness_points",
    )
    strikes = torch.exp(log_moneyness)
    call_prices = _normalized_call_prices(surface, log_moneyness=log_moneyness)
    curvature = _second_derivative_nonuniform(call_prices, strikes)
    return _require_finite_scalar(torch.relu(-curvature).mean(), penalty_name="convexity_penalty")


def roughness_penalty(predictions: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Discourage large local curvature in either axis."""

    _require_min_axis_length(
        grid_shape,
        maturity_min=3,
        moneyness_min=3,
        penalty_name="roughness_penalty",
    )
    surface = _reshape_surface(predictions, grid_shape)
    maturity_curvature = surface[:, 2:, :] - (2.0 * surface[:, 1:-1, :]) + surface[:, :-2, :]
    moneyness_curvature = surface[:, :, 2:] - (2.0 * surface[:, :, 1:-1]) + surface[:, :, :-2]
    penalty = maturity_curvature.square().mean() + moneyness_curvature.square().mean()
    return _require_finite_scalar(penalty, penalty_name="roughness_penalty")
