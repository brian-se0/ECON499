from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from ivsurf.config import StressWindowConfig
from ivsurf.evaluation.slice_reports import build_slice_metric_frame


def _panel_rows(model_name: str, scale: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for quote_date, target_date, date_shift in (
        (date(2021, 1, 4), date(2021, 1, 5), 0.0),
        (date(2021, 1, 5), date(2021, 1, 6), 0.002),
    ):
        for maturity_days in (30, 90):
            maturity_years = maturity_days / 365.0
            for moneyness_point, money_shift in ((-0.1, 0.001), (0.0, 0.0)):
                actual_total_variance = 0.04 * maturity_years + money_shift + date_shift
                predicted_total_variance = actual_total_variance * scale
                origin_iv = 0.18
                actual_iv = float(np.sqrt(actual_total_variance / maturity_years))
                predicted_iv = float(np.sqrt(predicted_total_variance / maturity_years))
                observed = not (
                    target_date == date(2021, 1, 6)
                    and maturity_days == 90
                    and moneyness_point == -0.1
                )
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": quote_date,
                        "target_date": target_date,
                        "maturity_days": maturity_days,
                        "moneyness_point": moneyness_point,
                        "actual_completed_total_variance": actual_total_variance,
                        "predicted_total_variance": predicted_total_variance,
                        "actual_completed_iv": actual_iv,
                        "predicted_iv": predicted_iv,
                        "actual_iv_change": actual_iv - origin_iv,
                        "predicted_iv_change": predicted_iv - origin_iv,
                        "actual_observed_mask": observed,
                        "observed_weight": 2.0 if observed else 0.0,
                    }
                )
    return rows


def test_build_slice_metric_frame_produces_expected_slices() -> None:
    panel = pl.DataFrame(_panel_rows("good", 1.0) + _panel_rows("bad", 1.10))
    stress_windows = (
        StressWindowConfig(
            label="jan_2021",
            start_date=date(2021, 1, 5),
            end_date=date(2021, 1, 6),
        ),
    )

    frame = build_slice_metric_frame(
        panel=panel,
        positive_floor=1.0e-8,
        stress_windows=stress_windows,
    )

    assert frame.height == 20
    assert set(frame["slice_family"].to_list()) == {"maturity", "moneyness", "stress_window"}
    assert set(frame["evaluation_scope"].to_list()) == {"observed", "full"}

    good_rows = frame.filter(pl.col("model_name") == "good")
    bad_rows = frame.filter(pl.col("model_name") == "bad")
    joined = good_rows.join(
        bad_rows,
        on=["slice_family", "slice_label", "evaluation_scope"],
        how="inner",
        suffix="_bad",
        validate="1:1",
    )
    assert (
        joined["wrmse_total_variance"] < joined["wrmse_total_variance_bad"]
    ).all()

    observed_stress = frame.filter(
        (pl.col("slice_family") == "stress_window")
        & (pl.col("slice_label") == "jan_2021")
        & (pl.col("evaluation_scope") == "observed")
        & (pl.col("model_name") == "good")
    )
    assert observed_stress["cell_count"][0] == 7
    assert observed_stress["target_day_count"][0] == 2
