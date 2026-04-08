"""Shared parquet IO helpers with explicit defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from ivsurf.io.atomic import cleanup_atomic_temp_files, write_text_atomic


def write_parquet_frame(
    frame: pl.DataFrame,
    output_path: Path,
    *,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
    statistics: bool = True,
) -> None:
    """Write a parquet artifact with the project's explicit defaults."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temp_files(output_path)
    temp_path = output_path.with_name(f"{output_path.name}.write_tmp")
    try:
        if temp_path.exists():
            temp_path.unlink()
        frame.write_parquet(temp_path, compression=compression, statistics=statistics)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_csv_frame(frame: pl.DataFrame, output_path: Path) -> None:
    """Write a CSV artifact atomically."""

    csv_text = frame.write_csv()
    if csv_text is None:
        message = f"Polars returned no CSV text while writing {output_path}."
        raise ValueError(message)
    write_text_atomic(output_path, csv_text, encoding="utf-8")


def read_parquet_files(paths: list[Path]) -> pl.DataFrame:
    """Read a list of parquet files into one concatenated frame."""

    if not paths:
        message = "read_parquet_files requires at least one parquet path."
        raise ValueError(message)
    return pl.concat([pl.read_parquet(path) for path in paths])


def scan_parquet_files(paths: list[Path]) -> pl.LazyFrame:
    """Create one lazy parquet scan from explicit file paths."""

    if not paths:
        message = "scan_parquet_files requires at least one parquet path."
        raise ValueError(message)
    return pl.scan_parquet([str(path) for path in paths])
