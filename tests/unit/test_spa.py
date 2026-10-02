from __future__ import annotations

import numpy as np
import pytest
from arch.bootstrap import SPA

from ivcast.stats.bootstrap import moving_block_bootstrap_indices
from ivcast.stats.spa import superior_predictive_ability_test


def _losses(n_obs: int = 400) -> tuple[np.ndarray, np.ndarray]:
    shocks = np.random.default_rng(42).normal(size=(n_obs, 2))
    shocks = (shocks - shocks.mean(axis=0)) / shocks.std(axis=0, ddof=1)
    differentials = shocks + np.asarray([1.5, -0.5]) / np.sqrt(n_obs)
    benchmark = np.full(n_obs, 10.0)
    return benchmark, benchmark[:, None] - differentials


def test_spa_consistent_recentering_reverses_liberal_rejection_like_arch() -> None:
    benchmark, candidates = _losses()
    result = superior_predictive_ability_test(
        benchmark, candidates, "benchmark", ("a", "b"), 0.10, 1, 10_000, 7
    )
    # IID resampling isolates recentering: arch's lower bound rejects at 10% while the
    # consistent test does not. The bootstrap long-run scale differs from arch's analytic
    # scale only by Monte Carlo error, so the consistent p-values agree approximately.
    reference = SPA(benchmark, candidates, block_size=1, reps=10_000, bootstrap="cbb", seed=7)
    reference.compute()
    assert reference.pvalues["lower"] == pytest.approx(0.0877)
    assert result.p_value == pytest.approx(reference.pvalues["consistent"], abs=0.01)
    assert reference.pvalues["lower"] < result.alpha < result.p_value
    assert result.recentering == "consistent"
    assert result.scale_method == "bootstrap_long_run_standard_deviation"


@pytest.mark.parametrize("block_size", [1, 5, 17])
def test_spa_matches_hansen_equations_with_the_same_circular_resamples(block_size: int) -> None:
    benchmark, candidates = _losses(120)
    # A third, clearly inferior candidate exercises threshold exclusion.
    worse_candidate = 12.0 + np.random.default_rng(81).normal(0.0, 0.2, size=120)
    candidates = np.column_stack([candidates, worse_candidate])
    result = superior_predictive_ability_test(
        benchmark, candidates, "benchmark", ("a", "b", "worse"), 0.10, block_size, 500, 19
    )

    # Independently evaluate Hansen (2005): studentize by the bootstrap long-run
    # standard deviation of the mean differential, recenter consistently, and compare
    # the bootstrap maxima with the observed statistic on the same resamples.
    differences = benchmark[:, None] - candidates
    n_obs = len(differences)
    sample_means = differences.mean(axis=0)
    indices = moving_block_bootstrap_indices(n_obs, block_size, 500, 19)
    resampled_means = np.stack([differences[rows].mean(axis=0) for rows in indices])
    omegas = np.sqrt(n_obs * np.mean(np.square(resampled_means - sample_means), axis=0))
    threshold = -np.sqrt(2.0 * np.log(np.log(n_obs))) * omegas / np.sqrt(n_obs)
    null_means = np.where(sample_means < threshold, sample_means, 0.0)
    assert null_means[:2] == pytest.approx([0.0, 0.0])
    assert null_means[2] < 0.0
    observed = max(0.0, float(np.max(np.sqrt(n_obs) * sample_means / omegas)))
    simulated = np.maximum(
        np.max(np.sqrt(n_obs) * (resampled_means - sample_means + null_means) / omegas, axis=1),
        0.0,
    )
    assert result.observed_statistic == pytest.approx(observed)
    assert result.p_value == pytest.approx(float(np.mean(simulated >= observed)))


def test_spa_returns_no_evidence_when_all_candidates_lose_by_mean() -> None:
    benchmark = np.ones(6)
    candidates = np.asarray([1.5, 1.8, 2.0, 2.1, 1.7, 1.6])[:, None]
    result = superior_predictive_ability_test(
        benchmark, candidates, "benchmark", ("worse",), 0.10, 2, 100, 7
    )
    assert result.observed_statistic == 0.0
    assert result.p_value == 1.0
    assert result.superior_models_by_mean == ()


def test_spa_handles_identical_forecasts_without_nan_or_false_significance() -> None:
    benchmark = np.arange(6, dtype=np.float64)
    result = superior_predictive_ability_test(
        benchmark, benchmark[:, None], "benchmark", ("duplicate",), 0.10, 2, 100, 7
    )
    assert result.observed_statistic == 0.0
    assert result.p_value == 1.0
    assert result.mean_differentials == (0.0,)


@pytest.mark.parametrize("difference", [-1.0, 1.0])
def test_spa_rejects_nonzero_constant_differentials(difference: float) -> None:
    benchmark = np.ones(6)
    with pytest.raises(ValueError, match="cannot estimate variation"):
        superior_predictive_ability_test(
            benchmark, (benchmark - difference)[:, None],
            "benchmark", ("constant",), 0.10, 2, 100, 7,
        )


@pytest.mark.parametrize("n_obs", [1, 2])
def test_spa_requires_enough_observations_for_log_log_threshold(n_obs: int) -> None:
    with pytest.raises(ValueError, match="at least three"):
        superior_predictive_ability_test(
            np.ones(n_obs), np.ones((n_obs, 1)), "benchmark", ("a",), 0.10, 1, 100, 7
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan"), float("inf")])
def test_spa_rejects_invalid_alpha(alpha: float) -> None:
    benchmark, candidates = _losses(10)
    with pytest.raises(ValueError, match="alpha"):
        superior_predictive_ability_test(
            benchmark, candidates, "benchmark", ("a", "b"), alpha, 1, 100, 7
        )


@pytest.mark.parametrize("block_size", [0, 10, 11])
def test_spa_rejects_invalid_or_degenerate_block_sizes(block_size: int) -> None:
    benchmark, candidates = _losses(10)
    with pytest.raises(ValueError, match="block_size"):
        superior_predictive_ability_test(
            benchmark, candidates, "benchmark", ("a", "b"), 0.10, block_size, 100, 7
        )


def test_spa_rejects_nonpositive_repetitions() -> None:
    benchmark, candidates = _losses(10)
    with pytest.raises(ValueError, match="bootstrap_reps"):
        superior_predictive_ability_test(
            benchmark, candidates, "benchmark", ("a", "b"), 0.10, 1, 0, 7
        )


def test_spa_rejects_finite_inputs_whose_differences_overflow() -> None:
    with pytest.raises(ValueError, match="must remain finite"):
        superior_predictive_ability_test(
            np.full(6, 1.0e308), np.full((6, 1), -1.0e308),
            "benchmark", ("a",), 0.10, 1, 100, 7,
        )
