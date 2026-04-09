from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.qc.raw_checks import assert_target_symbol_only


def test_assert_target_symbol_only_accepts_spx_underlying_scope_across_roots() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "underlying_symbol": ["^SPX", "^SPX"],
            "root": ["SPX", "SPXW"],
        }
    )

    assert_target_symbol_only(
        frame,
        symbol_column="underlying_symbol",
        expected_symbol="^SPX",
        dataset_name="synthetic_spx_scope",
    )
