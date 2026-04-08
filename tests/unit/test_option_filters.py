from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.cleaning.option_filters import apply_option_quality_flags
from ivsurf.config import CleaningConfig


def test_zero_iv_is_flagged_explicitly() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4)],
            "option_type": ["C"],
            "bid_1545": [1.0],
            "ask_1545": [1.2],
            "implied_volatility_1545": [0.0],
            "vega_1545": [1.0],
            "active_underlying_price_1545": [100.0],
            "mid_1545": [1.1],
            "tau_years": [0.1],
            "log_moneyness": [0.0],
        }
    )
    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())
    assert flagged["invalid_reason"][0] == "NON_POSITIVE_IV"
    assert flagged["is_valid_observation"][0] is False

