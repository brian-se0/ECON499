from __future__ import annotations

import pytest

from ivsurf.exceptions import SchemaDriftError
from ivsurf.schemas import RAW_ALLOWED_EXTRA_COLUMNS, RAW_COLUMNS, validate_raw_columns


def test_validate_raw_columns_accepts_expected_schema() -> None:
    validate_raw_columns(RAW_COLUMNS)


def test_validate_raw_columns_accepts_audited_extra_columns() -> None:
    validate_raw_columns((*RAW_COLUMNS, *RAW_ALLOWED_EXTRA_COLUMNS))


def test_validate_raw_columns_accepts_audited_implied_underlying_price_column() -> None:
    validate_raw_columns((*RAW_COLUMNS, "implied_underlying_price_1545"))


def test_validate_raw_columns_rejects_unexpected_extra_vendor_column() -> None:
    with pytest.raises(
        SchemaDriftError,
        match=r"unexpected=\['vendor_added_column_1545'\]",
    ):
        validate_raw_columns((*RAW_COLUMNS, "vendor_added_column_1545"))


def test_validate_raw_columns_rejects_missing_required_column() -> None:
    with pytest.raises(
        SchemaDriftError,
        match=r"missing=\['open_interest'\] unexpected=\[\]",
    ):
        validate_raw_columns(RAW_COLUMNS[:-1])
