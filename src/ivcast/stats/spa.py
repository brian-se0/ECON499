"""Superior Predictive Ability test against a benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivcast.stats.bootstrap import moving_block_bootstrap_indices


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
    recentering: str = "consistent"
    scale_method: str = "bootstrap_long_run_standard_deviation"


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
    """Run Hansen's consistently recentered SPA with circular-block resampling.

    Each loss differential is studentized by a consistent estimate of its long-run
    standard deviation, omega_k^2 = n * var*(dbar_k), the bootstrap variance of the
    mean differential (Hansen, 2005). Loss differentials are serially dependent, so an
    iid sample standard deviation is not a consistent scale. The same scales enter the
    observed and bootstrap statistics, and the consistent null recentering uses the
    log-log threshold. Bootstrap ties are included conservatively, and a zero observed
    statistic has p-value one.
    """

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
    if benchmark_losses.size < 3:
        message = "SPA test requires at least three observations for its log-log threshold."
        raise ValueError(message)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        message = "SPA test alpha must be finite and strictly between zero and one."
        raise ValueError(message)
    if block_size <= 0 or block_size >= benchmark_losses.size:
        message = "SPA block_size must be positive and smaller than the observation count."
        raise ValueError(message)
    if bootstrap_reps <= 0:
        message = "SPA bootstrap_reps must be positive."
        raise ValueError(message)

    benchmark_losses = benchmark_losses.astype(np.float64)
    candidate_losses = candidate_losses.astype(np.float64)
    bootstrap_indices = moving_block_bootstrap_indices(
        n_obs=benchmark_losses.shape[0],
        block_size=block_size,
        reps=bootstrap_reps,
        seed=seed,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        differentials = benchmark_losses[:, None] - candidate_losses
        means = differentials.mean(axis=0)
        centered_bootstrap_means = np.stack(
            [differentials[indices].mean(axis=0) - means for indices in bootstrap_indices]
        )
        scales = np.sqrt(
            differentials.shape[0] * np.mean(np.square(centered_bootstrap_means), axis=0)
        )
    if not all(np.isfinite(values).all() for values in (differentials, means, scales)):
        message = "SPA loss differentials, means, and scales must remain finite."
        raise ValueError(message)
    identical = np.all(differentials == 0.0, axis=0)
    constant = np.all(differentials == differentials[0], axis=0)
    unestimable = ~identical & (constant | (scales <= 0.0))
    if unestimable.any():
        names = tuple(np.asarray(candidate_models)[unestimable])
        message = (
            "SPA cannot estimate variation for nonzero constant or zero-variance "
            "loss differentials: "
            f"{names!r}."
        )
        raise ValueError(message)
    # An exact duplicate of the benchmark has zero statistic and no evidence
    # against the null; unit scaling only avoids 0/0 for these zero columns.
    safe_scales = np.where(identical, 1.0, scales)
    sqrt_n = np.sqrt(differentials.shape[0])
    with np.errstate(over="ignore", invalid="ignore"):
        standardized_means = sqrt_n * (means / safe_scales)
    if not np.isfinite(standardized_means).all():
        message = "SPA standardized mean loss differentials must remain finite."
        raise ValueError(message)
    observed_stat = max(0.0, float(np.max(standardized_means)))

    # Hansen (2005), pp. 368 and 372: g_c(mean) recenters every candidate
    # except those below the negative log-log threshold. max(mean, 0) instead
    # implements his liberal lower bound and is not the consistent SPA test.
    threshold = np.sqrt(2.0 * np.log(np.log(differentials.shape[0])))
    recentering_means = np.where(standardized_means >= -threshold, means, 0.0)
    recentered = differentials - recentering_means
    bootstrap_stats = np.empty(bootstrap_reps, dtype=np.float64)
    for rep in range(bootstrap_reps):
        sample = recentered[bootstrap_indices[rep]]
        sample_means = sample.mean(axis=0)
        bootstrap_stats[rep] = np.max(sqrt_n * (sample_means / safe_scales))

    if not np.isfinite(bootstrap_stats).all():
        message = "SPA bootstrap statistics must remain finite."
        raise ValueError(message)
    bootstrap_stats = np.maximum(bootstrap_stats, 0.0)
    p_value = 1.0 if observed_stat == 0.0 else float(np.mean(bootstrap_stats >= observed_stat))
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
