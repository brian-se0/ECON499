from __future__ import annotations

import numpy as np
import pytest

from ivsurf.surfaces.arbitrage_diagnostics import convexity_violations, summarize_diagnostics


def test_price_convexity_diagnostic_respects_nonuniform_moneyness_grid() -> None:
    surface = np.asarray([[0.04, 0.04, 0.04, 0.04]], dtype=np.float64)

    count, magnitude = convexity_violations(
        surface,
        moneyness_points=np.asarray([-0.30, -0.10, -0.05, 0.30], dtype=np.float64),
    )

    assert count == 0
    assert magnitude == 0.0


def test_price_convexity_diagnostic_flags_butterfly_violation() -> None:
    surface = np.asarray([[0.01, 1.00, 0.01]], dtype=np.float64)

    count, magnitude = convexity_violations(
        surface,
        moneyness_points=np.asarray([-0.10, 0.00, 0.10], dtype=np.float64),
    )

    assert count > 0
    assert magnitude > 0.0


def test_diagnostic_rejects_unsorted_moneyness_points() -> None:
    surface = np.asarray([[0.04, 0.04, 0.04]], dtype=np.float64)

    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_diagnostics(
            surface,
            moneyness_points=np.asarray([-0.10, 0.10, 0.00], dtype=np.float64),
        )

