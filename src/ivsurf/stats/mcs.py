"""Simplified model-confidence-set procedure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivsurf.stats.bootstrap import moving_block_bootstrap_indices


@dataclass(frozen=True, slots=True)
class McsIteration:
    """One elimination step in the MCS procedure."""

    included_models: tuple[str, ...]
    test_statistic: float
    p_value: float
    eliminated_model: str | None


@dataclass(frozen=True, slots=True)
class McsResult:
    """Final MCS output."""

    superior_models: tuple[str, ...]
    iterations: tuple[McsIteration, ...]
    alpha: float
    block_size: int
    bootstrap_reps: int


def _tmax_statistic(losses: np.ndarray, bootstrap_indices: np.ndarray) -> tuple[float, np.ndarray]:
    n_obs, n_models = losses.shape
    pairwise_diff = losses[:, :, None] - losses[:, None, :]
    pairwise_mean = pairwise_diff.mean(axis=0)

    bootstrap_pairwise_means = np.empty(
        (bootstrap_indices.shape[0], n_models, n_models),
        dtype=np.float64,
    )
    centered_diff = pairwise_diff - pairwise_mean[None, :, :]
    for rep in range(bootstrap_indices.shape[0]):
        sample = centered_diff[bootstrap_indices[rep]]
        bootstrap_pairwise_means[rep] = sample.mean(axis=0)
    pairwise_var = bootstrap_pairwise_means.var(axis=0, ddof=1)
    safe_var = np.where(pairwise_var > 0.0, pairwise_var, 1.0)
    observed_t = np.abs(pairwise_mean / np.sqrt(safe_var / n_obs))
    bootstrap_t = np.abs(
        bootstrap_pairwise_means / np.sqrt(safe_var[None, :, :] / n_obs)
    )
    observed_stat = float(np.max(observed_t))
    bootstrap_stats = bootstrap_t.reshape(bootstrap_t.shape[0], -1).max(axis=1)
    return observed_stat, bootstrap_stats


def model_confidence_set(
    losses: np.ndarray,
    model_names: tuple[str, ...],
    alpha: float,
    block_size: int,
    bootstrap_reps: int,
    seed: int,
) -> McsResult:
    """Run a Tmax-style model-confidence-set elimination routine."""

    if losses.ndim != 2:
        message = "losses must be a two-dimensional array."
        raise ValueError(message)
    if losses.shape[1] != len(model_names):
        message = "model_names length must match losses columns."
        raise ValueError(message)

    remaining_losses = losses.astype(np.float64)
    remaining_models = list(model_names)
    iterations: list[McsIteration] = []
    seed_offset = seed

    while len(remaining_models) > 1:
        bootstrap_indices = moving_block_bootstrap_indices(
            n_obs=remaining_losses.shape[0],
            block_size=block_size,
            reps=bootstrap_reps,
            seed=seed_offset,
        )
        observed_stat, bootstrap_stats = _tmax_statistic(
            remaining_losses,
            bootstrap_indices,
        )
        p_value = float(np.mean(bootstrap_stats >= observed_stat))
        if p_value >= alpha:
            iterations.append(
                McsIteration(
                    included_models=tuple(remaining_models),
                    test_statistic=observed_stat,
                    p_value=p_value,
                    eliminated_model=None,
                )
            )
            break

        average_losses = remaining_losses.mean(axis=0)
        worst_index = int(np.argmax(average_losses))
        eliminated_model = remaining_models.pop(worst_index)
        remaining_losses = np.delete(remaining_losses, worst_index, axis=1)
        iterations.append(
            McsIteration(
                included_models=tuple(remaining_models),
                test_statistic=observed_stat,
                p_value=p_value,
                eliminated_model=eliminated_model,
            )
        )
        seed_offset += 1

    return McsResult(
        superior_models=tuple(remaining_models),
        iterations=tuple(iterations),
        alpha=alpha,
        block_size=block_size,
        bootstrap_reps=bootstrap_reps,
    )

