"""Dependent bootstrap utilities for forecast-comparison tests."""

from __future__ import annotations

import numpy as np


def moving_block_bootstrap_indices(
    n_obs: int,
    block_size: int,
    reps: int,
    seed: int,
) -> np.ndarray:
    """Generate circular moving-block bootstrap indices."""

    if n_obs <= 0:
        message = "n_obs must be positive."
        raise ValueError(message)
    if block_size <= 0:
        message = "block_size must be positive."
        raise ValueError(message)
    if reps <= 0:
        message = "reps must be positive."
        raise ValueError(message)

    rng = np.random.default_rng(seed)
    blocks_per_rep = int(np.ceil(n_obs / block_size))
    indices = np.empty((reps, n_obs), dtype=np.int64)
    base = np.arange(block_size, dtype=np.int64)
    for rep in range(reps):
        starts = rng.integers(0, n_obs, size=blocks_per_rep, endpoint=False)
        sample = np.concatenate([(start + base) % n_obs for start in starts])[:n_obs]
        indices[rep, :] = sample
    return indices

