"""Deterministic sequential axis-wise surface completion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from ivcast.exceptions import InterpolationError

COMPLETION_STATUS_OBSERVED = "observed"
COMPLETION_STATUS_INTERPOLATED = "interpolated"
COMPLETION_STATUS_EXTRAPOLATED = "extrapolated_boundary_fill"
COMPLETION_STATUS_MISSING = "missing"
COMPLETED_SURFACE_SCHEMA_VERSION = "completed_surface_v2"
MATURITY_BOUNDARY_FILL_FLAT_IV = "flat_implied_volatility"
MATURITY_BOUNDARY_FILL_FLAT_TOTAL_VARIANCE = "flat_total_variance"
MATURITY_BOUNDARY_FILL_MODES = (
    MATURITY_BOUNDARY_FILL_FLAT_IV,
    MATURITY_BOUNDARY_FILL_FLAT_TOTAL_VARIANCE,
)


@dataclass(frozen=True, slots=True)
class CompletedSurface:
    """Completed daily surface with mask information."""

    completed_total_variance: np.ndarray
    observed_mask: np.ndarray
    completion_status: np.ndarray


def _fill_axis(
    values: np.ndarray,
    coordinates: np.ndarray,
    completion_status: np.ndarray,
    *,
    proportional_boundary_fill: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill one axis by PCHIP interpolation and nearest-boundary carry-forward.

    With `proportional_boundary_fill`, the carried boundary value is scaled by the
    coordinate ratio. On the maturity axis this holds the boundary implied volatility
    constant (total variance proportional to maturity) instead of holding total
    variance constant, which would imply that all variance accrues before the shortest
    observed maturity.
    """

    result = values.copy()
    status = completion_status.copy()
    finite_mask = np.isfinite(values)
    count = int(finite_mask.sum())
    if count == 0:
        return result, status
    missing_positions = np.flatnonzero(~finite_mask)
    if missing_positions.size == 0:
        return result, status

    observed_x = coordinates[finite_mask]
    observed_y = values[finite_mask]
    target_x = coordinates[missing_positions]
    if proportional_boundary_fill:
        low_fill = observed_y[0] * (target_x / observed_x[0])
        high_fill = observed_y[-1] * (target_x / observed_x[-1])
    else:
        low_fill = np.full(target_x.shape, observed_y[0], dtype=np.float64)
        high_fill = np.full(target_x.shape, observed_y[-1], dtype=np.float64)
    if count == 1:
        result[missing_positions] = np.where(target_x < observed_x[0], low_fill, high_fill)
        status[missing_positions] = COMPLETION_STATUS_EXTRAPOLATED
        return result, status

    interpolator = PchipInterpolator(observed_x, observed_y, extrapolate=False)
    predicted = interpolator(target_x)
    extrapolated = (target_x < observed_x.min()) | (target_x > observed_x.max())
    predicted = np.where(
        target_x < observed_x.min(),
        low_fill,
        np.where(target_x > observed_x.max(), high_fill, predicted),
    )
    result[missing_positions] = predicted
    status[missing_positions] = np.where(
        extrapolated,
        COMPLETION_STATUS_EXTRAPOLATED,
        COMPLETION_STATUS_INTERPOLATED,
    )
    return result, status


def complete_surface(
    observed_total_variance: np.ndarray,
    observed_mask: np.ndarray,
    maturity_coordinates: np.ndarray,
    moneyness_coordinates: np.ndarray,
    interpolation_order: tuple[str, ...],
    interpolation_cycles: int,
    total_variance_floor: float,
    maturity_boundary_fill: str = MATURITY_BOUNDARY_FILL_FLAT_IV,
) -> CompletedSurface:
    """Complete a surface by fixed-order sequential one-dimensional interpolation."""

    if maturity_boundary_fill not in MATURITY_BOUNDARY_FILL_MODES:
        message = (
            f"Unsupported maturity_boundary_fill {maturity_boundary_fill!r}; "
            f"expected one of {MATURITY_BOUNDARY_FILL_MODES!r}."
        )
        raise ValueError(message)
    proportional_maturity_fill = maturity_boundary_fill == MATURITY_BOUNDARY_FILL_FLAT_IV

    completed = observed_total_variance.astype(np.float64, copy=True)
    normalized_observed_mask = np.asarray(observed_mask, dtype=bool)
    if normalized_observed_mask.shape != completed.shape:
        message = (
            "observed_mask must have the same shape as observed_total_variance, "
            f"found {normalized_observed_mask.shape} != {completed.shape}."
        )
        raise ValueError(message)
    completed[~normalized_observed_mask] = np.nan
    completion_status = np.full(
        completed.shape,
        COMPLETION_STATUS_MISSING,
        dtype=object,
    )
    completion_status[normalized_observed_mask] = COMPLETION_STATUS_OBSERVED

    for _ in range(interpolation_cycles):
        for axis_name in interpolation_order:
            if axis_name == "maturity":
                for money_idx in range(completed.shape[1]):
                    completed[:, money_idx], completion_status[:, money_idx] = _fill_axis(
                        completed[:, money_idx],
                        maturity_coordinates,
                        completion_status[:, money_idx],
                        proportional_boundary_fill=proportional_maturity_fill,
                    )
            elif axis_name == "moneyness":
                for maturity_idx in range(completed.shape[0]):
                    completed[maturity_idx, :], completion_status[maturity_idx, :] = _fill_axis(
                        completed[maturity_idx, :],
                        moneyness_coordinates,
                        completion_status[maturity_idx, :],
                    )
            else:
                message = f"Unsupported interpolation axis: {axis_name}"
                raise ValueError(message)

    if not np.isfinite(completed).all():
        message = (
            "Surface completion left NaN or infinite values "
            "after deterministic interpolation."
        )
        raise InterpolationError(message)

    completed = np.maximum(completed, total_variance_floor)
    return CompletedSurface(
        completed_total_variance=completed,
        observed_mask=normalized_observed_mask,
        completion_status=completion_status,
    )
