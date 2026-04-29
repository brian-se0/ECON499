from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ivsurf.cleaning.option_filters import apply_option_quality_flags, valid_option_rows
from ivsurf.config import CleaningConfig


def _base_cleaning_frame(*, missing_column: str | None = None) -> pl.DataFrame:
    row = {
        "quote_date": date(2021, 1, 4),
        "option_type": "C",
        "bid_1545": 1.0,
        "ask_1545": 1.2,
        "implied_volatility_1545": 0.2,
        "vega_1545": 1.0,
        "active_underlying_price_1545": 100.0,
        "strike": 100.0,
        "mid_1545": 1.1,
        "tau_years": 0.1,
        "log_moneyness": 0.0,
        "total_variance": 0.004,
    }
    rows = [dict(row), dict(row)]
    if missing_column is not None:
        rows[0][missing_column] = None
    return pl.DataFrame(rows)


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
            "strike": [100.0],
            "mid_1545": [1.1],
            "tau_years": [0.1],
            "log_moneyness": [0.0],
            "total_variance": [0.0],
        }
    )
    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())
    assert flagged["invalid_reason"][0] == "NON_POSITIVE_IV"
    assert flagged["is_valid_observation"][0] is False


def test_cleaning_threshold_fields_are_explicitly_exclusive() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "option_type": ["C", "C"],
            "bid_1545": [0.0, 0.1],
            "ask_1545": [0.0, 0.2],
            "implied_volatility_1545": [0.2, 0.2],
            "vega_1545": [1.0, 1.0],
            "active_underlying_price_1545": [100.0, 100.0],
            "strike": [100.0, 100.0],
            "mid_1545": [0.0, 0.15],
            "tau_years": [0.1, 0.1],
            "log_moneyness": [0.0, 0.0],
            "total_variance": [0.004, 0.004],
        }
    )
    flagged = apply_option_quality_flags(
        frame=frame,
        config=CleaningConfig(
            min_valid_bid_exclusive=0.0,
            min_valid_ask_exclusive=0.0,
            min_valid_mid_price_exclusive=0.0,
        ),
    )

    assert flagged["invalid_reason"].to_list() == ["NON_POSITIVE_BID", None]
    assert flagged["is_valid_observation"].to_list() == [False, True]


def test_critical_null_1545_fields_receive_explicit_reason_codes() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "option_type": ["C", "C"],
            "bid_1545": [None, 1.0],
            "ask_1545": [1.2, 1.2],
            "implied_volatility_1545": [0.2, None],
            "vega_1545": [1.0, 1.0],
            "active_underlying_price_1545": [100.0, 100.0],
            "strike": [100.0, 100.0],
            "mid_1545": [1.1, 1.1],
            "tau_years": [0.1, 0.1],
            "log_moneyness": [0.0, 0.0],
            "total_variance": [0.004, 0.004],
        }
    )

    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())

    assert flagged["invalid_reason"].to_list() == [
        "MISSING_BID_1545",
        "MISSING_IMPLIED_VOLATILITY_1545",
    ]
    assert flagged["is_valid_observation"].to_list() == [False, False]


@pytest.mark.parametrize(
    ("missing_column", "expected_reason"),
    (
        ("option_type", "MISSING_OPTION_TYPE"),
        ("bid_1545", "MISSING_BID_1545"),
        ("ask_1545", "MISSING_ASK_1545"),
        ("implied_volatility_1545", "MISSING_IMPLIED_VOLATILITY_1545"),
        ("vega_1545", "MISSING_VEGA_1545"),
        ("active_underlying_price_1545", "MISSING_ACTIVE_UNDERLYING_PRICE_1545"),
        ("strike", "MISSING_STRIKE"),
        ("mid_1545", "MISSING_MID_1545"),
        ("tau_years", "MISSING_TAU_YEARS"),
        ("total_variance", "MISSING_TOTAL_VARIANCE"),
        ("log_moneyness", "MISSING_LOG_MONEYNESS"),
    ),
)
def test_every_critical_null_field_is_invalid_with_explicit_reason(
    missing_column: str,
    expected_reason: str,
) -> None:
    flagged = apply_option_quality_flags(
        frame=_base_cleaning_frame(missing_column=missing_column),
        config=CleaningConfig(),
    )

    assert flagged["invalid_reason"].to_list() == [expected_reason, None]
    assert flagged["is_valid_observation"].to_list() == [False, True]
    assert valid_option_rows(flagged).height == 1


@pytest.mark.parametrize(
    ("column_name", "bad_value", "expected_reason"),
    (
        ("bid_1545", float("nan"), "NONFINITE_BID_1545"),
        ("bid_1545", float("inf"), "NONFINITE_BID_1545"),
        ("bid_1545", float("-inf"), "NONFINITE_BID_1545"),
        ("ask_1545", float("nan"), "NONFINITE_ASK_1545"),
        ("implied_volatility_1545", float("inf"), "NONFINITE_IV_1545"),
        ("vega_1545", float("-inf"), "NONFINITE_VEGA_1545"),
        ("active_underlying_price_1545", float("nan"), "NONFINITE_UNDERLYING_1545"),
        ("strike", float("inf"), "NONFINITE_STRIKE"),
        ("mid_1545", float("nan"), "NONFINITE_MID_1545"),
        ("tau_years", float("inf"), "NONFINITE_TAU_YEARS"),
        ("total_variance", float("nan"), "NONFINITE_TOTAL_VARIANCE"),
        ("log_moneyness", float("-inf"), "NONFINITE_LOG_MONEYNESS"),
    ),
)
def test_every_critical_nonfinite_field_is_invalid_with_explicit_reason(
    column_name: str,
    bad_value: float,
    expected_reason: str,
) -> None:
    frame = _base_cleaning_frame()
    frame = frame.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(bad_value).otherwise(
        pl.col(column_name)
    ).alias(column_name))

    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())

    assert flagged["invalid_reason"].to_list() == [expected_reason, None]
    assert flagged["is_valid_observation"].to_list() == [False, True]
    assert valid_option_rows(flagged).height == 1


@pytest.mark.parametrize("bad_strike", (0.0, -1.0))
def test_non_positive_strike_is_invalid_before_log_moneyness_reason(bad_strike: float) -> None:
    frame = _base_cleaning_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(bad_strike))
        .otherwise(pl.col("strike"))
        .alias("strike"),
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(float("-inf")))
        .otherwise(pl.col("log_moneyness"))
        .alias("log_moneyness"),
    )

    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())

    assert flagged["invalid_reason"].to_list() == ["NON_POSITIVE_STRIKE", None]
    assert flagged["is_valid_observation"].to_list() == [False, True]
