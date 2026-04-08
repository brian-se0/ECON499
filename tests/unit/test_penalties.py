from __future__ import annotations

import torch

from ivsurf.models.penalties import calendar_monotonicity_penalty, convexity_penalty


def test_penalties_are_zero_for_monotone_convex_surface() -> None:
    surface = torch.tensor(
        [[0.04, 0.03, 0.04], [0.05, 0.04, 0.05]],
        dtype=torch.float32,
    ).reshape(1, -1)
    assert float(calendar_monotonicity_penalty(surface, (2, 3))) == 0.0
    assert float(convexity_penalty(surface, (2, 3))) == 0.0


def test_penalties_increase_for_violations() -> None:
    surface = torch.tensor(
        [[0.05, 0.04, 0.05], [0.04, 0.06, 0.04]],
        dtype=torch.float32,
    ).reshape(1, -1)
    assert float(calendar_monotonicity_penalty(surface, (2, 3))) > 0.0
    assert float(convexity_penalty(surface, (2, 3))) > 0.0

