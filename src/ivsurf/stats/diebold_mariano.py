"""Diebold-Mariano forecast-comparison test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class DieboldMarianoResult:
    """Pairwise DM test output."""

    model_a: str
    model_b: str
    n_obs: int
    mean_loss_a: float
    mean_loss_b: float
    mean_differential: float
    statistic: float
    p_value: float
    alternative: str
    max_lag: int


def _autocovariance(values: np.ndarray, lag: int) -> float:
    centered = values - values.mean()
    if lag == 0:
        return float(np.dot(centered, centered) / values.shape[0])
    return float(np.dot(centered[lag:], centered[:-lag]) / values.shape[0])


def long_run_variance(values: np.ndarray, max_lag: int) -> float:
    """Newey-West style long-run variance estimate."""

    gamma0 = _autocovariance(values, lag=0)
    variance = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - (lag / (max_lag + 1))
        gamma = _autocovariance(values, lag=lag)
        variance += 2.0 * weight * gamma
    return max(variance, 0.0)


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    model_a: str,
    model_b: str,
    alternative: str = "two-sided",
    max_lag: int = 0,
) -> DieboldMarianoResult:
    """Compute the DM test on two aligned daily loss series."""

    if loss_a.shape != loss_b.shape:
        message = "loss_a and loss_b must have the same shape."
        raise ValueError(message)
    if loss_a.ndim != 1:
        message = "DM test expects one-dimensional loss series."
        raise ValueError(message)
    if alternative not in {"two-sided", "greater", "less"}:
        message = f"Unsupported DM alternative: {alternative}"
        raise ValueError(message)

    differential = loss_a.astype(np.float64) - loss_b.astype(np.float64)
    mean_diff = float(differential.mean())
    lrv = long_run_variance(differential, max_lag=max_lag)
    n_obs = differential.shape[0]
    if lrv == 0.0:
        if mean_diff > 0.0:
            statistic = float("inf")
        elif mean_diff < 0.0:
            statistic = float("-inf")
        else:
            statistic = 0.0
    else:
        statistic = mean_diff / np.sqrt(lrv / n_obs)

    if alternative == "two-sided":
        p_value = 2.0 * (1.0 - norm.cdf(abs(statistic)))
    elif alternative == "greater":
        p_value = 1.0 - norm.cdf(statistic)
    else:
        p_value = norm.cdf(statistic)

    return DieboldMarianoResult(
        model_a=model_a,
        model_b=model_b,
        n_obs=n_obs,
        mean_loss_a=float(loss_a.mean()),
        mean_loss_b=float(loss_b.mean()),
        mean_differential=mean_diff,
        statistic=float(statistic),
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        alternative=alternative,
        max_lag=max_lag,
    )

