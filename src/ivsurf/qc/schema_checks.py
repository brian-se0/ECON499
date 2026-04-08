"""Explicit schema and column-level validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from ivsurf.exceptions import DataValidationError, SchemaDriftError


def assert_required_columns(
    columns: Iterable[str],
    *,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Fail fast when required columns are absent."""

    actual = tuple(columns)
    required = tuple(required_columns)
    missing = [name for name in required if name not in actual]
    if missing:
        message = f"{dataset_name} is missing required columns: {missing}"
        raise SchemaDriftError(message)


def assert_non_null_columns(
    frame: pl.DataFrame,
    *,
    columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Fail fast when critical columns contain nulls."""

    null_columns = [name for name in columns if frame[name].null_count() > 0]
    if null_columns:
        message = f"{dataset_name} contains nulls in critical columns: {null_columns}"
        raise DataValidationError(message)
