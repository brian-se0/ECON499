"""Model confidence set with the range statistic of Hansen, Lunde and Nason (2011)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivcast.stats.bootstrap import moving_block_bootstrap_indices

MCS_PROCEDURE_NAME = "hln_2011_range_statistic_elimination"


@dataclass(frozen=True, slots=True)
class McsIteration:
    """One equivalence test and, when rejected, one elimination step.

    `included_models` is the candidate set tested at this step, before elimination.
    `mcs_p_value` is the running maximum of the step p-values (Hansen, Lunde and Nason,
    2011, Section 3.1), the MCS p-value of the model eliminated at this step.
    """

    included_models: tuple[str, ...]
    test_statistic: float
    p_value: float
    mcs_p_value: float
    eliminated_model: str | None


@dataclass(frozen=True, slots=True)
class McsResult:
    """Final model confidence set output."""

    superior_models: tuple[str, ...]
    iterations: tuple[McsIteration, ...]
    alpha: float
    block_size: int
    bootstrap_reps: int
    procedure_name: str


def _range_statistic(
    losses: np.ndarray,
    bootstrap_indices: np.ndarray,
    model_names: list[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return T_R, its bootstrap distribution, and the pairwise t-statistics.

    t_ij = dbar_ij / sqrt(var*(dbar_ij)), with var* the bootstrap variance of the mean
    loss differential (HLN 2011, Section 3.1.2), and T_R = max_ij |t_ij|.
    """

    pairwise_diff = losses[:, :, None] - losses[:, None, :]
    pairwise_mean = pairwise_diff.mean(axis=0)
    centered_bootstrap_means = np.empty(
        (bootstrap_indices.shape[0], losses.shape[1], losses.shape[1]),
        dtype=np.float64,
    )
    for rep in range(bootstrap_indices.shape[0]):
        centered_bootstrap_means[rep] = (
            pairwise_diff[bootstrap_indices[rep]].mean(axis=0) - pairwise_mean
        )
    pairwise_var = np.mean(np.square(centered_bootstrap_means), axis=0)

    off_diagonal = ~np.eye(losses.shape[1], dtype=bool)
    identical = np.all(pairwise_diff == 0.0, axis=0)
    unestimable = off_diagonal & ~identical & ~(pairwise_var > 0.0)
    if unestimable.any():
        first, second = np.argwhere(unestimable)[0]
        message = (
            "MCS cannot standardize a nonzero loss differential with zero bootstrap "
            f"variance: {model_names[int(first)]!r} vs {model_names[int(second)]!r}."
        )
        raise ValueError(message)
    # Exact duplicates have zero differential in every sample; unit scale avoids 0/0.
    safe_scale = np.sqrt(np.where(pairwise_var > 0.0, pairwise_var, 1.0))
    t_matrix = pairwise_mean / safe_scale
    observed_stat = float(np.max(np.abs(t_matrix)))
    bootstrap_stats = np.abs(centered_bootstrap_means / safe_scale[None, :, :]).reshape(
        bootstrap_indices.shape[0],
        -1,
    ).max(axis=1)
    return observed_stat, bootstrap_stats, t_matrix


def model_confidence_set(
    losses: np.ndarray,
    model_names: tuple[str, ...],
    alpha: float,
    block_size: int,
    bootstrap_reps: int,
    seed: int,
) -> McsResult:
    """Run the HLN (2011) model confidence set with the range statistic T_R.

    Each step tests equal predictive ability of the remaining models with T_R and a
    circular block bootstrap. On rejection it eliminates e_R = argmax_i max_j t_ij, the
    model whose standardized loss excess over another remaining model is largest.
    """

    if losses.ndim != 2:
        message = "losses must be a two-dimensional array."
        raise ValueError(message)
    if losses.shape[0] == 0:
        message = "MCS requires at least one aligned loss observation."
        raise ValueError(message)
    if losses.shape[1] == 0:
        message = "MCS requires at least one model."
        raise ValueError(message)
    if losses.shape[1] != len(model_names):
        message = "model_names length must match losses columns."
        raise ValueError(message)
    if not np.isfinite(losses).all():
        message = "MCS losses must contain only finite values."
        raise ValueError(message)

    remaining_losses = losses.astype(np.float64)
    remaining_models = list(model_names)
    iterations: list[McsIteration] = []
    seed_offset = seed
    running_p_value = 0.0

    while len(remaining_models) > 1:
        bootstrap_indices = moving_block_bootstrap_indices(
            n_obs=remaining_losses.shape[0],
            block_size=block_size,
            reps=bootstrap_reps,
            seed=seed_offset,
        )
        observed_stat, bootstrap_stats, t_matrix = _range_statistic(
            remaining_losses,
            bootstrap_indices,
            remaining_models,
        )
        p_value = float(np.mean(bootstrap_stats >= observed_stat))
        running_p_value = max(running_p_value, p_value)
        tested_models = tuple(remaining_models)
        if p_value >= alpha:
            iterations.append(
                McsIteration(
                    included_models=tested_models,
                    test_statistic=observed_stat,
                    p_value=p_value,
                    mcs_p_value=running_p_value,
                    eliminated_model=None,
                )
            )
            break

        worst_index = int(np.argmax(t_matrix.max(axis=1)))
        eliminated_model = remaining_models.pop(worst_index)
        remaining_losses = np.delete(remaining_losses, worst_index, axis=1)
        iterations.append(
            McsIteration(
                included_models=tested_models,
                test_statistic=observed_stat,
                p_value=p_value,
                mcs_p_value=running_p_value,
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
        procedure_name=MCS_PROCEDURE_NAME,
    )
