from __future__ import annotations

import csv
from datetime import date
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import polars as pl

from ivsurf.config import RawDataConfig
from ivsurf.io.ingest_cboe import ingest_one_zip
from ivsurf.schemas import RAW_COLUMNS


def _raw_row(symbol: str) -> dict[str, object]:
    return {
        "underlying_symbol": symbol,
        "quote_date": "2021-01-04",
        "root": "SPXW",
        "expiration": "2021-01-15",
        "strike": 3700.0,
        "option_type": "C",
        "trade_volume": 10,
        "bid_size_1545": 5,
        "bid_1545": 1.0,
        "ask_size_1545": 5,
        "ask_1545": 1.2,
        "underlying_bid_1545": 3699.0,
        "underlying_ask_1545": 3701.0,
        "active_underlying_price_1545": 3700.0,
        "implied_volatility_1545": 0.2,
        "delta_1545": 0.5,
        "gamma_1545": 0.1,
        "theta_1545": -0.01,
        "vega_1545": 1.0,
        "rho_1545": 0.01,
        "open_interest": 100,
    }


def _write_raw_zip(path: Path) -> None:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RAW_COLUMNS)
    writer.writeheader()
    writer.writerow(_raw_row("^SPX"))
    writer.writerow(_raw_row("^NDX"))
    with ZipFile(path, "w") as archive:
        archive.writestr("daily.csv", buffer.getvalue())


def test_ingestion_result_records_target_symbol_filter_counts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "data" / "bronze"
    raw_dir.mkdir(parents=True)
    zip_path = raw_dir / "UnderlyingOptionsEODCalcs_20210104.zip"
    _write_raw_zip(zip_path)
    config = RawDataConfig(
        raw_options_dir=raw_dir,
        bronze_dir=bronze_dir,
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        manifests_dir=tmp_path / "data" / "manifests",
        sample_start_date=date(2021, 1, 4),
        sample_end_date=date(2021, 1, 4),
    )

    result = ingest_one_zip(zip_path=zip_path, config=config)

    assert result.raw_row_count == 2
    assert result.target_symbol_row_count == 1
    assert result.non_target_symbol_row_count == 1
    assert result.row_count == 1
    assert result.bronze_path is not None
    bronze = pl.read_parquet(result.bronze_path)
    assert bronze["underlying_symbol"].to_list() == ["^SPX"]
