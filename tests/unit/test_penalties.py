from __future__ import annotations

import pytest
import torch

from ivsurf.models.penalties import (
    calendar_monotonicity_penalty,
    convexity_penalty,
    roughness_penalty,
)


def test_penalties_are_zero_for_monotone_convex_surface() -> None:
    surface = torch.tensor(
        [[0.04, 0.04, 0.04], [0.05, 0.05, 0.05]],
        dtype=torch.float32,
    ).reshape(1, -1)
    assert float(calendar_monotonicity_penalty(surface, (2, 3))) == 0.0
    assert float(convexity_penalty(surface, (2, 3), moneyness_points=(-0.1, 0.0, 0.1))) == 0.0


def test_penalties_increase_for_violations() -> None:
    surface = torch.tensor(
        [[0.05, 0.05, 0.05], [0.01, 1.00, 0.01]],
        dtype=torch.float32,
    ).reshape(1, -1)
    assert float(calendar_monotonicity_penalty(surface, (2, 3))) > 0.0
    assert float(convexity_penalty(surface, (2, 3), moneyness_points=(-0.1, 0.0, 0.1))) > 0.0


def test_price_convexity_penalty_respects_nonuniform_moneyness_grid() -> None:
    surface = torch.tensor(
        [[0.04, 0.04, 0.04, 0.04]],
        dtype=torch.float64,
    )
    penalty = convexity_penalty(
        surface,
        (1, 4),
        moneyness_points=(-0.30, -0.10, -0.05, 0.30),
    )
    assert float(penalty) == 0.0


def test_calendar_penalty_rejects_grid_without_maturity_difference() -> None:
    surface = torch.tensor([[0.04, 0.05, 0.06]], dtype=torch.float32)

    with pytest.raises(ValueError, match="calendar_monotonicity_penalty requires"):
        calendar_monotonicity_penalty(surface, (1, 3))


def test_convexity_penalty_rejects_grid_without_second_moneyness_difference() -> None:
    surface = torch.tensor([[0.04, 0.05], [0.06, 0.07]], dtype=torch.float32).reshape(1, -1)

    with pytest.raises(ValueError, match="convexity_penalty requires"):
        convexity_penalty(surface, (2, 2), moneyness_points=(-0.1, 0.0))


def test_roughness_penalty_rejects_grid_without_second_differences() -> None:
    surface = torch.tensor([[0.04, 0.05], [0.06, 0.07]], dtype=torch.float32).reshape(1, -1)

    with pytest.raises(ValueError, match="roughness_penalty requires"):
        roughness_penalty(surface, (2, 2))


def test_roughness_penalty_is_finite_on_minimum_valid_grid() -> None:
    surface = torch.tensor(
        [
            [0.04, 0.05, 0.06],
            [0.05, 0.06, 0.07],
            [0.06, 0.07, 0.08],
        ],
        dtype=torch.float32,
    ).reshape(1, -1)

    penalty = roughness_penalty(surface, (3, 3))

    assert penalty.ndim == 0
    assert bool(torch.isfinite(penalty).item())
