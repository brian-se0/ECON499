from __future__ import annotations

import pytest

from ivsurf.exceptions import SchemaDriftError
from ivsurf.schemas import RAW_ALLOWED_EXTRA_COLUMNS, RAW_COLUMNS, validate_raw_columns


def test_validate_raw_columns_accepts_expected_schema() -> None:
    validate_raw_columns(RAW_COLUMNS)


def test_validate_raw_columns_accepts_audited_extra_columns() -> None:
    validate_raw_columns((*RAW_COLUMNS, *RAW_ALLOWED_EXTRA_COLUMNS))


def test_validate_raw_columns_rejects_drift() -> None:
    with pytest.raises(SchemaDriftError):
        validate_raw_columns((*RAW_COLUMNS[:-1], "unexpected_field"))
