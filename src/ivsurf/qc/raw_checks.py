"""Raw-file and bronze-ingestion validation helpers."""

from __future__ import annotations

import polars as pl

from ivsurf.exceptions import DataValidationError


def assert_single_quote_date(frame: pl.DataFrame, *, dataset_name: str) -> None:
    """Require exactly one quote_date in one daily artifact."""

    unique_count = frame.select(pl.col("quote_date").n_unique()).item()
    if unique_count != 1:
        message = f"{dataset_name} must contain exactly one quote_date, found {unique_count}."
        raise DataValidationError(message)


def assert_target_symbol_only(
    frame: pl.DataFrame,
    *,
    symbol_column: str,
    expected_symbol: str,
    dataset_name: str,
) -> None:
    """Require one explicit symbol after early pipeline filtering."""

    symbols = frame.select(pl.col(symbol_column).unique()).to_series().to_list()
    if symbols != [expected_symbol]:
        message = (
            f"{dataset_name} must contain only {expected_symbol!r} in {symbol_column}, "
            f"found {symbols}."
        )
        raise DataValidationError(message)
