"""Shared fail-fast contracts for observed-cell evaluation inputs."""

from __future__ import annotations

import numpy as np


def require_finite_array(values: np.ndarray, *, context: str) -> np.ndarray:
    """Return a float64 array after requiring all entries to be finite."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        message = f"{context} must contain only finite values."
        raise ValueError(message)
    return array


def require_binary_mask_array(values: np.ndarray, *, context: str) -> np.ndarray:
    """Return a boolean mask after requiring explicit 0/1 mask values."""

    mask_values = require_finite_array(values, context=context)
    valid = (mask_values == 0.0) | (mask_values == 1.0)
    if not valid.all():
        invalid_count = int((~valid).sum())
        message = f"{context} must be binary 0/1; invalid_count={invalid_count}."
        raise ValueError(message)
    return mask_values.astype(bool, copy=False)


def require_nonnegative_weights(values: np.ndarray, *, context: str) -> np.ndarray:
    """Return float64 weights after requiring finite nonnegative values."""

    weights = require_finite_array(values, context=context)
    if (weights < 0.0).any():
        minimum = float(weights.min())
        message = f"{context} must be nonnegative; minimum_value={minimum}."
        raise ValueError(message)
    return weights


def require_observed_weight_contract(
    *,
    observed_mask: np.ndarray,
    observed_weight: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Require observed weights to match an explicit observed-cell mask."""

    if observed_mask.shape != observed_weight.shape:
        message = (
            f"{context} observed mask and observed weight shapes must match; "
            f"found {observed_mask.shape!r} and {observed_weight.shape!r}."
        )
        raise ValueError(message)
    mask = require_binary_mask_array(observed_mask, context=f"{context} observed mask")
    weights = require_nonnegative_weights(observed_weight, context=f"{context} observed weight")
    unobserved_nonzero = (~mask) & (weights != 0.0)
    if unobserved_nonzero.any():
        message = (
            f"{context} observed_weight must be exactly zero for unobserved cells; "
            f"invalid_count={int(unobserved_nonzero.sum())}."
        )
        raise ValueError(message)
    observed_nonpositive = mask & (weights <= 0.0)
    if observed_nonpositive.any():
        message = (
            f"{context} observed cells must have strictly positive observed_weight; "
            f"invalid_count={int(observed_nonpositive.sum())}."
        )
        raise ValueError(message)
    return mask, weights


def require_positive_observed_weight_sum(
    weights: np.ndarray,
    *,
    metric_name: str,
) -> None:
    """Require a positive observed-weight sum for an observed-cell metric."""

    if weights.size > 0 and float(weights.sum()) > 0.0:
        return
    message = (
        f"{metric_name} requires at least one observed cell with positive observed weight; "
        "refusing to fall back to full-grid scoring."
    )
    raise ValueError(message)
