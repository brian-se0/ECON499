"""Surface-shape diagnostics for calendar monotonicity and butterfly convexity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ArbitrageDiagnostics:
    """Violation counts and magnitudes."""

    calendar_violation_count: int
    calendar_violation_magnitude: float
    convexity_violation_count: int
    convexity_violation_magnitude: float


def calendar_monotonicity_violations(surface: np.ndarray) -> tuple[int, float]:
    """Count decreases across maturity."""

    diffs = np.diff(surface, axis=0)
    violations = np.minimum(diffs, 0.0)
    return int(np.count_nonzero(violations < 0.0)), float(np.abs(violations).sum())


def _validate_coordinate_vector(
    coordinates: np.ndarray,
    *,
    expected_size: int,
    coordinate_name: str,
) -> np.ndarray:
    vector = np.asarray(coordinates, dtype=np.float64)
    if vector.ndim != 1:
        message = f"{coordinate_name} must be one-dimensional."
        raise ValueError(message)
    if vector.shape[0] != expected_size:
        message = (
            f"{coordinate_name} length {vector.shape[0]} does not match expected size "
            f"{expected_size}."
        )
        raise ValueError(message)
    if not np.isfinite(vector).all():
        message = f"{coordinate_name} must contain only finite values."
        raise ValueError(message)
    if not np.all(np.diff(vector) > 0.0):
        message = f"{coordinate_name} must be strictly increasing."
        raise ValueError(message)
    return vector


def _validate_positive_total_variance(surface: np.ndarray) -> np.ndarray:
    values = np.asarray(surface, dtype=np.float64)
    if values.ndim != 2:
        message = "surface must be a two-dimensional maturity x moneyness array."
        raise ValueError(message)
    if not np.isfinite(values).all():
        message = "surface must contain only finite total-variance values."
        raise ValueError(message)
    if not np.all(values > 0.0):
        message = "surface total variance must be strictly positive for price diagnostics."
        raise ValueError(message)
    return values


def _normalized_call_prices(
    total_variance: np.ndarray,
    *,
    log_moneyness: np.ndarray,
) -> np.ndarray:
    strikes = np.exp(log_moneyness)
    sqrt_variance = np.sqrt(total_variance)
    d1 = ((-log_moneyness)[None, :] + (0.5 * total_variance)) / sqrt_variance
    d2 = d1 - sqrt_variance
    return cast(FloatArray, ndtr(d1) - (strikes[None, :] * ndtr(d2)))


def _second_derivative_nonuniform(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    left_spacing = coordinates[1:-1] - coordinates[:-2]
    right_spacing = coordinates[2:] - coordinates[1:-1]
    full_spacing = coordinates[2:] - coordinates[:-2]
    left_slope = (values[:, 1:-1] - values[:, :-2]) / left_spacing[None, :]
    right_slope = (values[:, 2:] - values[:, 1:-1]) / right_spacing[None, :]
    return cast(FloatArray, 2.0 * (right_slope - left_slope) / full_spacing[None, :])


def convexity_violations(
    surface: np.ndarray,
    *,
    moneyness_points: np.ndarray,
) -> tuple[int, float]:
    """Count negative strike-space call-price curvature across moneyness."""

    values = _validate_positive_total_variance(surface)
    log_moneyness = _validate_coordinate_vector(
        moneyness_points,
        expected_size=values.shape[1],
        coordinate_name="moneyness_points",
    )
    strikes = np.exp(log_moneyness)
    call_prices = _normalized_call_prices(values, log_moneyness=log_moneyness)
    curvature = _second_derivative_nonuniform(call_prices, strikes)
    violations = np.minimum(curvature, 0.0)
    return int(np.count_nonzero(violations < 0.0)), float(np.abs(violations).sum())


def summarize_diagnostics(
    surface: np.ndarray,
    *,
    moneyness_points: np.ndarray,
) -> ArbitrageDiagnostics:
    """Summarize daily arbitrage-aware diagnostics."""

    calendar_count, calendar_magnitude = calendar_monotonicity_violations(surface)
    convexity_count, convexity_magnitude = convexity_violations(
        surface,
        moneyness_points=moneyness_points,
    )
    return ArbitrageDiagnostics(
        calendar_violation_count=calendar_count,
        calendar_violation_magnitude=calendar_magnitude,
        convexity_violation_count=convexity_count,
        convexity_violation_magnitude=convexity_magnitude,
    )
