"""Superior Predictive Ability test against a benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivsurf.stats.bootstrap import moving_block_bootstrap_indices


@dataclass(frozen=True, slots=True)
class SpaResult:
    """SPA summary output."""

    benchmark_model: str
    candidate_models: tuple[str, ...]
    observed_statistic: float
    p_value: float
    mean_differentials: tuple[float, ...]
    superior_models_by_mean: tuple[str, ...]
    alpha: float
    block_size: int
    bootstrap_reps: int


def superior_predictive_ability_test(
    benchmark_losses: np.ndarray,
    candidate_losses: np.ndarray,
    benchmark_model: str,
    candidate_models: tuple[str, ...],
    alpha: float,
    block_size: int,
    bootstrap_reps: int,
    seed: int,
) -> SpaResult:
    """Run a block-bootstrap SPA test for a set of candidate models."""

    if benchmark_losses.ndim != 1:
        message = "benchmark_losses must be one-dimensional."
        raise ValueError(message)
    if candidate_losses.ndim != 2:
        message = "candidate_losses must be two-dimensional."
        raise ValueError(message)
    if benchmark_losses.shape[0] != candidate_losses.shape[0]:
        message = "benchmark and candidate losses must align in time."
        raise ValueError(message)
    if benchmark_losses.size == 0:
        message = "SPA test requires at least one aligned loss observation."
        raise ValueError(message)
    if candidate_losses.shape[1] == 0:
        message = "SPA test requires at least one candidate model."
        raise ValueError(message)
    if candidate_losses.shape[1] != len(candidate_models):
        message = "candidate_models length must match candidate_losses columns."
        raise ValueError(message)
    if not np.isfinite(benchmark_losses).all() or not np.isfinite(candidate_losses).all():
        message = "SPA test losses must contain only finite values."
        raise ValueError(message)

    benchmark_losses = benchmark_losses.astype(np.float64)
    candidate_losses = candidate_losses.astype(np.float64)
    differentials = benchmark_losses[:, None] - candidate_losses
    means = differentials.mean(axis=0)
    scales = differentials.std(axis=0, ddof=1)
    safe_scales = np.where(scales > 0.0, scales, 1.0)
    observed_stat = float(np.max(np.sqrt(differentials.shape[0]) * means / safe_scales))

    recentered = differentials - np.maximum(means, 0.0)
    bootstrap_indices = moving_block_bootstrap_indices(
        n_obs=differentials.shape[0],
        block_size=block_size,
        reps=bootstrap_reps,
        seed=seed,
    )
    bootstrap_stats = np.empty(bootstrap_reps, dtype=np.float64)
    for rep in range(bootstrap_reps):
        sample = recentered[bootstrap_indices[rep]]
        sample_means = sample.mean(axis=0)
        bootstrap_stats[rep] = np.max(np.sqrt(sample.shape[0]) * sample_means / safe_scales)

    p_value = float(np.mean(bootstrap_stats >= observed_stat))
    superior_models = tuple(
        model_name
        for model_name, mean_value in zip(candidate_models, means, strict=True)
        if mean_value > 0.0
    )
    return SpaResult(
        benchmark_model=benchmark_model,
        candidate_models=candidate_models,
        observed_statistic=observed_stat,
        p_value=p_value,
        mean_differentials=tuple(float(value) for value in means),
        superior_models_by_mean=superior_models,
        alpha=alpha,
        block_size=block_size,
        bootstrap_reps=bootstrap_reps,
    )
