"""Deterministic sequential axis-wise surface completion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from ivsurf.exceptions import InterpolationError

COMPLETION_STATUS_OBSERVED = "observed"
COMPLETION_STATUS_INTERPOLATED = "interpolated"
COMPLETION_STATUS_EXTRAPOLATED = "extrapolated_boundary_fill"
COMPLETION_STATUS_MISSING = "missing"
COMPLETED_SURFACE_SCHEMA_VERSION = "completed_surface_v1"


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
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    status = completion_status.copy()
    finite_mask = np.isfinite(values)
    count = int(finite_mask.sum())
    if count == 0:
        return result, status
    missing_positions = np.flatnonzero(~finite_mask)
    if missing_positions.size == 0:
        return result, status
    if count == 1:
        result[missing_positions] = values[finite_mask][0]
        status[missing_positions] = COMPLETION_STATUS_EXTRAPOLATED
        return result, status

    observed_x = coordinates[finite_mask]
    observed_y = values[finite_mask]
    interpolator = PchipInterpolator(observed_x, observed_y, extrapolate=False)

    target_x = coordinates[missing_positions]
    predicted = interpolator(target_x)
    extrapolated = (target_x < observed_x.min()) | (target_x > observed_x.max())
    predicted = np.where(
        target_x < observed_x.min(),
        observed_y[0],
        np.where(target_x > observed_x.max(), observed_y[-1], predicted),
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
) -> CompletedSurface:
    """Complete a surface by fixed-order sequential one-dimensional interpolation."""

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
