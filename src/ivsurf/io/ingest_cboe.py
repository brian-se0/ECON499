"""Raw Cboe daily-zip ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from zipfile import ZipFile

import polars as pl

from ivsurf.config import RawDataConfig
from ivsurf.exceptions import DataValidationError, SchemaDriftError
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.qc.raw_checks import assert_single_quote_date, assert_target_symbol_only
from ivsurf.qc.sample_window import quote_date_in_sample_window
from ivsurf.qc.schema_checks import assert_non_null_columns
from ivsurf.schemas import RAW_COLUMNS, RAW_POLARS_SCHEMA, validate_raw_columns


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary for one ingested raw zip."""

    source_zip: Path
    quote_date: date
    bronze_path: Path | None
    raw_row_count: int
    target_symbol_row_count: int
    non_target_symbol_row_count: int
    row_count: int
    status: str


def list_raw_zip_files(config: RawDataConfig) -> list[Path]:
    """List raw zip files in deterministic order."""

    return sorted(config.raw_options_dir.glob(config.raw_file_glob))


def _extract_single_csv(zip_path: Path, temp_dir: Path) -> Path:
    with ZipFile(zip_path) as archive:
        entries = archive.namelist()
        if len(entries) != 1:
            message = f"{zip_path} must contain exactly one CSV member, found {len(entries)}."
            raise SchemaDriftError(message)
        entry_name = entries[0]
        destination = temp_dir / entry_name
        archive.extract(entry_name, path=temp_dir)
    return destination


def ingest_one_zip(zip_path: Path, config: RawDataConfig) -> IngestionResult:
    """Read one daily zip, filter to SPX, and write partitioned parquet."""

    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extracted_csv = _extract_single_csv(zip_path, temp_dir)
        header = pl.read_csv(extracted_csv, n_rows=0).columns
        validate_raw_columns(header)

        lazy_frame = pl.scan_csv(
            extracted_csv,
            schema_overrides=cast(dict[str, Any], RAW_POLARS_SCHEMA),
            null_values={"open_interest": ""},
            quote_char='"',
            infer_schema=True,
        )
        selected = lazy_frame.select(RAW_COLUMNS)
        target_filter_counts = (
            selected.select(
                pl.len().alias("raw_row_count"),
                (pl.col("underlying_symbol") == config.target_symbol)
                .sum()
                .alias("target_symbol_row_count"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
        raw_row_count = int(target_filter_counts["raw_row_count"])
        target_symbol_row_count = int(target_filter_counts["target_symbol_row_count"])
        non_target_symbol_row_count = raw_row_count - target_symbol_row_count
        frame = (
            selected
            .filter(pl.col("underlying_symbol") == config.target_symbol)
            .with_columns(
                pl.col("quote_date").str.strptime(pl.Date, strict=True),
                pl.col("expiration").str.strptime(pl.Date, strict=True),
                pl.lit(str(zip_path)).alias("source_zip"),
            )
            .collect(engine="streaming")
        )

    if frame.is_empty():
        message = f"No rows for symbol {config.target_symbol} found in {zip_path.name}."
        raise DataValidationError(message)

    assert_target_symbol_only(
        frame,
        symbol_column="underlying_symbol",
        expected_symbol=config.target_symbol,
        dataset_name=zip_path.name,
    )
    assert_single_quote_date(frame, dataset_name=zip_path.name)
    assert_non_null_columns(
        frame,
        columns=("quote_date", "expiration", "root", "strike", "option_type"),
        dataset_name=zip_path.name,
    )

    quote_date = frame["quote_date"][0]
    if not isinstance(quote_date, date):
        message = f"{zip_path.name} quote_date must be parsed as a Polars Date."
        raise TypeError(message)
    if not quote_date_in_sample_window(quote_date, config):
        return IngestionResult(
            source_zip=zip_path,
            quote_date=quote_date,
            bronze_path=None,
            raw_row_count=raw_row_count,
            target_symbol_row_count=target_symbol_row_count,
            non_target_symbol_row_count=non_target_symbol_row_count,
            row_count=frame.height,
            status="skipped_out_of_sample_window",
        )

    output_dir = config.bronze_dir / f"year={quote_date.year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    bronze_path = output_dir / f"{zip_path.stem}.parquet"
    write_parquet_frame(frame, bronze_path)
    return IngestionResult(
        source_zip=zip_path,
        quote_date=quote_date,
        bronze_path=bronze_path,
        raw_row_count=raw_row_count,
        target_symbol_row_count=target_symbol_row_count,
        non_target_symbol_row_count=non_target_symbol_row_count,
        row_count=frame.height,
        status="written",
    )


def ingest_all(config: RawDataConfig, limit: int | None = None) -> list[IngestionResult]:
    """Ingest all raw zips in deterministic order."""

    zip_paths = list_raw_zip_files(config)
    if limit is not None:
        zip_paths = zip_paths[:limit]
    return [ingest_one_zip(path, config) for path in zip_paths]
